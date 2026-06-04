//! Fragmented-MP4 relay over HTTP. Spawns a single ffmpeg.exe subprocess
//! that takes raw NV12 video and f32 audio over two TCP localhost sockets,
//! encodes to H.264 + AAC inside a streamable fragmented MP4 container, and
//! emits the container bytes on stdout. vicash parses those bytes into an
//! `init segment` (ftyp + moov) plus a sequence of `media segments` (moof
//! + mdat pairs) and broadcasts them to every client connected to the
//! `/stream.mp4` HTTP endpoint.
//!
//! The local preview path is not touched: it keeps reading from the same
//! `SharedFrame` latest-frame slot at full capture rate, and the cpal audio
//! callbacks keep filling the speaker ringbuf. The relay is purely an
//! additional consumer that the user can turn on or off from the F1 panel.

use anyhow::{Context, Result, anyhow};
use parking_lot::{Condvar, Mutex};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::audio::AudioState;
use crate::frame::{FrameData, SharedFrame};

/// Live state shared between the ffmpeg-stdout parser and every connected
/// HTTP client. The parser sets `init_segment` once and then publishes a
/// fresh `latest_media` for each fMP4 fragment it sees on stdout.
pub struct BroadcastState {
    /// Init bytes (ftyp + moov). Set once after ffmpeg's first output and
    /// kept forever. Late-joining clients receive this verbatim before any
    /// media segment so the browser's MSE buffer can be initialised.
    init_segment: Mutex<Option<Arc<Vec<u8>>>>,
    /// Most recent media segment (moof + mdat). Each new segment bumps
    /// `latest_seq` so clients can detect that a new chunk is ready
    /// without copying the buffer on every wake-up.
    latest_media: Mutex<Option<Arc<Vec<u8>>>>,
    latest_seq: AtomicU64,
    cv: Condvar,
    /// Set when the relay is shutting down so client loops can exit
    /// cleanly even if no more segments are produced.
    pub shutdown: AtomicBool,
}

impl BroadcastState {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            init_segment: Mutex::new(None),
            latest_media: Mutex::new(None),
            latest_seq: AtomicU64::new(0),
            cv: Condvar::new(),
            shutdown: AtomicBool::new(false),
        })
    }

    pub fn init_segment(&self) -> Option<Arc<Vec<u8>>> {
        self.init_segment.lock().clone()
    }

    /// Block until either a new media segment is published or `shutdown`
    /// is asserted. Returns the new segment plus its sequence number, or
    /// None if the relay shut down while we were waiting.
    pub fn wait_for_next(&self, last_seen: u64, timeout: Duration) -> Option<(Arc<Vec<u8>>, u64)> {
        let mut guard = self.latest_media.lock();
        let deadline = Instant::now() + timeout;
        loop {
            if self.shutdown.load(Ordering::Relaxed) {
                return None;
            }
            let cur = self.latest_seq.load(Ordering::Relaxed);
            if cur != last_seen {
                if let Some(buf) = guard.as_ref() {
                    return Some((buf.clone(), cur));
                }
            }
            let now = Instant::now();
            if now >= deadline {
                return None;
            }
            self.cv.wait_for(&mut guard, deadline - now);
        }
    }
}

/// Handle to a running ffmpeg relay. Drop it (or call `shutdown`) to tear
/// the subprocess and threads down cleanly.
pub struct Fmp4Relay {
    pub state: Arc<BroadcastState>,
    child: Mutex<Option<Child>>,
    threads: Mutex<Vec<std::thread::JoinHandle<()>>>,
    /// Held so we can detach the relay sink hook from the cpal input
    /// callback on shutdown. Without this the producer half stays alive
    /// in `AudioState.relay_sink` forever and the input callback keeps
    /// trying to push into a consumer that nobody is draining.
    audio: Arc<AudioState>,
}

impl Fmp4Relay {
    /// Build a new relay. Spawns ffmpeg, two TCP listeners, three worker
    /// threads (video writer, audio writer, stdout parser). `width`,
    /// `height`, `fps` describe what the capture thread is currently
    /// publishing; `audio` provides the live sample rate + the relay sink
    /// hook into the cpal input callback.
    pub fn spawn(
        shared_frame: SharedFrame,
        audio: Arc<AudioState>,
        width: u32,
        height: u32,
        fps: u32,
    ) -> Result<Self> {
        if width == 0 || height == 0 || fps == 0 {
            return Err(anyhow!("relay needs an active capture (got {width}x{height}@{fps})"));
        }

        // Bind two listeners on a random free localhost port. ffmpeg will
        // connect to both as a client; we accept and then keep the sockets
        // alive for the lifetime of the subprocess.
        let video_listener = TcpListener::bind(("127.0.0.1", 0))
            .context("failed to bind video TCP listener")?;
        let audio_listener = TcpListener::bind(("127.0.0.1", 0))
            .context("failed to bind audio TCP listener")?;
        let video_addr: SocketAddr = video_listener
            .local_addr()
            .context("could not read video listener port")?;
        let audio_addr: SocketAddr = audio_listener
            .local_addr()
            .context("could not read audio listener port")?;

        let sample_rate = audio.sample_rate().max(1);
        let in_channels = (audio.channels() as u32).max(1);

        log::info!(
            "fmp4 relay: video {width}x{height}@{fps} on {video_addr}, audio f32 {sample_rate}Hz x{in_channels} on {audio_addr}"
        );

        let mut child = build_ffmpeg(
            video_addr,
            audio_addr,
            width,
            height,
            fps,
            sample_rate,
            in_channels,
        )
        .context("failed to spawn ffmpeg.exe for fMP4 relay")?;

        // Take stdout BEFORE we move child into the Mutex; the parser
        // thread reads from this for the entire run.
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("ffmpeg child had no stdout"))?;

        // Accept the video connection first, then spin up the video writer
        // BEFORE waiting on the audio accept. ffmpeg opens inputs in order
        // and reads a few bytes from input 1 before it bothers opening
        // input 2, so if the video socket stays silent, ffmpeg never tries
        // to connect to the audio port and our audio accept times out.
        let video_sock = accept_with_timeout(&video_listener, Duration::from_secs(5))
            .context("ffmpeg never connected to the video input port")?;
        drop(video_listener);

        let state = BroadcastState::new();
        let mut threads = Vec::new();

        let state_v = state.clone();
        let shared_v = shared_frame.clone();
        threads.push(
            std::thread::Builder::new()
                .name("fmp4-relay-video".into())
                .spawn(move || {
                    video_writer_loop(video_sock, shared_v, fps, state_v);
                })
                .context("failed to spawn fmp4 video writer thread")?,
        );

        // Now ffmpeg is being fed video bytes and will proceed to opening
        // input 2 (audio). Accept that connection. Give it a longer
        // window because ffmpeg has to demux a few seconds of video first
        // on slow systems.
        let audio_sock = accept_with_timeout(&audio_listener, Duration::from_secs(15))
            .context("ffmpeg never connected to the audio input port")?;
        drop(audio_listener);

        let state_a = state.clone();
        let audio_cons = audio.install_relay_sink((sample_rate as usize) * 2 * (in_channels as usize));
        threads.push(
            std::thread::Builder::new()
                .name("fmp4-relay-audio".into())
                .spawn(move || {
                    audio_writer_loop(audio_sock, audio_cons, state_a);
                })
                .context("failed to spawn fmp4 audio writer thread")?,
        );

        // Reader: parses MP4 boxes off ffmpeg stdout, splits them into
        // init + media segments, and publishes to BroadcastState.
        let state_r = state.clone();
        threads.push(
            std::thread::Builder::new()
                .name("fmp4-relay-reader".into())
                .spawn(move || {
                    reader_loop(stdout, state_r);
                })
                .context("failed to spawn fmp4 reader thread")?,
        );

        Ok(Self {
            state,
            child: Mutex::new(Some(child)),
            threads: Mutex::new(threads),
            audio,
        })
    }

    /// Stop the subprocess and all worker threads.
    pub fn shutdown(&self) {
        self.state.shutdown.store(true, Ordering::Relaxed);
        self.state.cv.notify_all();
        if let Some(mut child) = self.child.lock().take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        let handles: Vec<_> = self.threads.lock().drain(..).collect();
        for h in handles {
            let _ = h.join();
        }
        // Detach the producer half from the cpal input callback so the
        // hot path stops fanning out into a now-dead ringbuf.
        self.audio.remove_relay_sink();
    }
}

impl Drop for Fmp4Relay {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn build_ffmpeg(
    video_addr: SocketAddr,
    audio_addr: SocketAddr,
    width: u32,
    height: u32,
    fps: u32,
    sample_rate: u32,
    channels: u32,
) -> Result<Child> {
    // ffmpeg flags chosen for a streamable fragmented MP4:
    //   -movflags +empty_moov         emit init box (moov) up front
    //   +default_base_moof            offsets relative to fragment
    //   +frag_keyframe                split at each H.264 keyframe
    //   +separate_moof                independent moof per track
    //   +omit_tfhd_offset             omit absolute offsets (live)
    //   -frag_duration 500000         500ms target fragment duration
    //   -g <fps>                      keyframe every second
    //   -tune zerolatency             skip lookahead, minimise queueing
    //   -ac 2                         upmix mono input to stereo AAC
    let frag_duration_us = 500_000u32;
    let gop = fps.max(1);
    let video_url = format!("tcp://{video_addr}");
    let audio_url = format!("tcp://{audio_addr}");
    let mut cmd = Command::new("ffmpeg");
    cmd.args([
        "-y",
        "-loglevel", "warning",
        "-fflags", "+nobuffer+genpts",
    ]);
    cmd.args([
        "-f", "rawvideo",
        "-pix_fmt", "nv12",
        "-video_size", &format!("{width}x{height}"),
        "-framerate", &format!("{fps}"),
        "-i", &video_url,
    ]);
    cmd.args([
        "-f", "f32le",
        "-ar", &format!("{sample_rate}"),
        "-ac", &format!("{channels}"),
        "-i", &audio_url,
    ]);
    cmd.args([
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-g", &format!("{gop}"),
        "-pix_fmt", "yuv420p",
    ]);
    cmd.args([
        "-c:a", "aac",
        "-b:a", "128k",
        "-ac", "2",
    ]);
    cmd.args([
        "-movflags",
        "+empty_moov+default_base_moof+frag_keyframe+separate_moof+omit_tfhd_offset",
        "-frag_duration", &format!("{frag_duration_us}"),
        "-f", "mp4",
        "pipe:1",
    ]);
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    cmd.spawn().map_err(|e| e.into())
}

fn accept_with_timeout(listener: &TcpListener, timeout: Duration) -> Result<TcpStream> {
    listener.set_nonblocking(true)?;
    let deadline = Instant::now() + timeout;
    loop {
        match listener.accept() {
            Ok((sock, _)) => {
                sock.set_nonblocking(false)?;
                sock.set_nodelay(true)?;
                return Ok(sock);
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    return Err(anyhow!("timed out waiting for ffmpeg to connect"));
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(e) => return Err(e.into()),
        }
    }
}

fn video_writer_loop(
    mut sock: TcpStream,
    shared: SharedFrame,
    fps: u32,
    state: Arc<BroadcastState>,
) {
    let frame_duration = Duration::from_secs_f64(1.0 / fps as f64);
    let mut next_due = Instant::now();
    let mut last_seq: u64 = 0;
    let mut held: Option<Arc<Vec<u8>>> = None;
    loop {
        if state.shutdown.load(Ordering::Relaxed) {
            break;
        }
        // Pull the latest frame; if it's new, take a strong reference to
        // its NV12 bytes; if not, repeat the last one we held so ffmpeg
        // continues to see a steady stream.
        if let Some(f) = shared.get() {
            if f.seq != last_seq {
                last_seq = f.seq;
                if let FrameData::Nv12(bytes) = &f.data {
                    held = Some(bytes.clone());
                }
            }
        }
        if let Some(buf) = held.as_deref() {
            if let Err(e) = sock.write_all(buf) {
                log::warn!("fmp4 video writer: socket write failed: {e}");
                break;
            }
        }
        // Pace to the requested fps. If we fell behind we don't try to
        // catch up - that would burst-flush the encoder and trash sync.
        next_due += frame_duration;
        let now = Instant::now();
        if next_due > now {
            std::thread::sleep(next_due - now);
        } else if now - next_due > Duration::from_millis(200) {
            next_due = now;
        }
    }
    let _ = sock.shutdown(std::net::Shutdown::Write);
}

fn audio_writer_loop(
    mut sock: TcpStream,
    mut cons: ringbuf::HeapCons<f32>,
    state: Arc<BroadcastState>,
) {
    use ringbuf::traits::Consumer;
    let mut buf: Vec<f32> = Vec::with_capacity(4096);
    loop {
        if state.shutdown.load(Ordering::Relaxed) {
            break;
        }
        buf.clear();
        while let Some(s) = cons.try_pop() {
            buf.push(s);
            if buf.len() >= 4096 {
                break;
            }
        }
        if buf.is_empty() {
            std::thread::sleep(Duration::from_millis(5));
            continue;
        }
        let bytes: &[u8] = bytemuck::cast_slice(&buf);
        if let Err(e) = sock.write_all(bytes) {
            log::warn!("fmp4 audio writer: socket write failed: {e}");
            break;
        }
    }
    let _ = sock.shutdown(std::net::Shutdown::Write);
}

/// Pull MP4 boxes off ffmpeg's stdout and route them into the broadcast
/// state. The first chunk we collect (everything up to the first `moof`)
/// becomes the init segment. Every `moof` followed by its `mdat` becomes
/// one media segment.
fn reader_loop(mut stdout: std::process::ChildStdout, state: Arc<BroadcastState>) {
    let mut pending = Vec::<u8>::with_capacity(64 * 1024);
    let mut init_done = false;
    let mut chunk = [0u8; 64 * 1024];
    while !state.shutdown.load(Ordering::Relaxed) {
        let n = match stdout.read(&mut chunk) {
            Ok(0) => {
                log::info!("fmp4 reader: ffmpeg stdout closed");
                break;
            }
            Ok(n) => n,
            Err(e) => {
                log::warn!("fmp4 reader: stdout read failed: {e}");
                break;
            }
        };
        pending.extend_from_slice(&chunk[..n]);

        // Walk the buffer and slice out complete boxes. We track the byte
        // offset of the start of each top-level box; when we see a moof,
        // the bytes BEFORE it constitute either the init segment (first
        // time) or a tail of the previous media segment (subsequent
        // times), and the moof plus its following mdat is the new media
        // segment.
        loop {
            let Some((init_end, media_end)) = next_segment_split(&pending, init_done) else {
                break;
            };

            if !init_done {
                // init_end is the offset where the first moof begins.
                let init = pending[..init_end].to_vec();
                *state.init_segment.lock() = Some(Arc::new(init));
                init_done = true;
                log::info!("fmp4 reader: init segment cached ({} bytes)", init_end);
            }
            let media: Vec<u8> = pending[init_end..media_end].to_vec();
            let media_arc = Arc::new(media);
            {
                let mut guard = state.latest_media.lock();
                *guard = Some(media_arc);
                state.latest_seq.fetch_add(1, Ordering::Relaxed);
            }
            state.cv.notify_all();
            pending.drain(..media_end);
        }
    }
    state.shutdown.store(true, Ordering::Relaxed);
    state.cv.notify_all();
}

/// Identify the start of the next moof and the end of the segment that
/// follows it (moof + mdat). Returns Some((moof_start, segment_end)) when
/// a complete segment is present in `buf`; None when more bytes are
/// needed. If `init_done` is false the function also requires the bytes
/// before the moof to be available (the ftyp + moov + first chunk).
fn next_segment_split(buf: &[u8], init_done: bool) -> Option<(usize, usize)> {
    let mut pos = 0usize;
    let mut moof_start: Option<usize> = None;
    while pos + 8 <= buf.len() {
        let size = u32::from_be_bytes(buf[pos..pos + 4].try_into().ok()?) as usize;
        let kind = &buf[pos + 4..pos + 8];
        if size < 8 {
            // Malformed or extended (size=1) box. fMP4 streaming from
            // ffmpeg should never emit either, so bail and wait for more
            // bytes; if it persists the reader will surface stderr.
            return None;
        }
        if pos + size > buf.len() {
            // Incomplete box; need more bytes.
            return None;
        }
        if kind == b"moof" {
            if moof_start.is_none() {
                moof_start = Some(pos);
                pos += size;
                continue;
            }
            // Hit the NEXT moof while still expecting an mdat. Treat the
            // previous moof's end as a complete segment (ffmpeg sometimes
            // emits empty fragments at the very start).
            if !init_done {
                return None;
            }
            return Some((moof_start.unwrap(), pos));
        }
        if kind == b"mdat" {
            if let Some(start) = moof_start {
                let end = pos + size;
                return Some((start, end));
            }
            // mdat without a preceding moof: skip it as part of the init
            // (rare, but the moov has its own mdat in some encoders).
            pos += size;
            continue;
        }
        pos += size;
    }
    None
}
