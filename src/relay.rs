use crate::audio::AudioState;
use crate::audio_relay::{self, AudioChunk, AudioRelay, AudioSubscription};
use crate::fmp4_relay::Fmp4Relay;
use crate::relay_video::{Pacer, VideoVariants};
use crate::frame::{FrameData, SharedFrame};
use crate::settings::Settings;
#[cfg(windows)]
use crate::video_stream::{self, TsBroadcaster};
use anyhow::{Context, Result};
use image::codecs::jpeg::JpegEncoder;
use image::{ColorType, ImageEncoder};
use parking_lot::{Condvar, Mutex};
use std::io::Write;
use std::net::{IpAddr, SocketAddr, TcpListener, UdpSocket};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;
use tiny_http::{Header, Method, Response, Server};

const BOUNDARY: &str = "vcshareframe";

/// Live state of the relay server, surfaced into the F1 panel so the user can
/// see at a glance where to point a second PC.
pub struct RelayInfo {
    pub bind_addr: SocketAddr,
    pub lan_url: String,
    pub local_url: String,
    pub active_clients: AtomicUsize,
    pub total_clients: AtomicUsize,
    /// Flipped to true to ask all relay threads (accept loop, encoder, every
    /// client streamer) to wind down before their next iteration. Wrapped in
    /// an Arc so we can hand the flag to side threads (the H.264 pipeline)
    /// without making them depend on RelayInfo itself.
    pub shutdown: Arc<AtomicBool>,
    /// Broadcast fan-out for the H.264 over MPEG-TS pipeline. None on
    /// platforms without an MSMF encoder (anything but Windows for now).
    #[cfg(windows)]
    pub ts: Arc<TsBroadcaster>,
    /// Optional fragmented-MP4 + AAC relay backed by an ffmpeg subprocess.
    /// Set when the user enables "Audio im Relay" in F1; cleared on
    /// toggle-off. Behind a Mutex so the HTTP /stream.mp4 handler can
    /// read it at request time.
    pub fmp4: Mutex<Option<Arc<Fmp4Relay>>>,
    /// Raw PCM fan-out behind /audio.wav and the /live page. Always present;
    /// it only taps the audio callback while someone is listening.
    pub audio: Arc<AudioRelay>,
    /// Downscaled encoders for the /live quality menu.
    pub video: Arc<VideoVariants>,
}

impl RelayInfo {
    pub fn stop(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
        // Wake any waiting encoder / clients so they notice the shutdown.
    }
}

/// Single encoded JPEG snapshot kept in memory, refreshed by the encoder
/// thread on every new capture frame. Every connected client serialises the
/// same bytes, so JPEG cost is paid once per frame regardless of audience
/// size.
pub(crate) struct LatestJpeg {
    pub bytes: Arc<Vec<u8>>,
    pub seq: u64,
    /// Capture time of the source frame on the relay clock, for /live sync.
    pub pts_us: u64,
}

pub(crate) struct SharedJpeg {
    state: Mutex<Option<LatestJpeg>>,
    notify: Condvar,
}

impl SharedJpeg {
    pub fn new() -> Self {
        Self { state: Mutex::new(None), notify: Condvar::new() }
    }

    pub fn publish(&self, snapshot: LatestJpeg) {
        *self.state.lock() = Some(snapshot);
        self.notify.notify_all();
    }

    /// Block (with timeout) until a snapshot newer than `since` exists or the
    /// shutdown flag is raised. Returns None if shutdown.
    fn wait_for_new(&self, since: u64, shutdown: &AtomicBool) -> Option<LatestJpeg> {
        let mut guard = self.state.lock();
        loop {
            if shutdown.load(Ordering::Relaxed) {
                return None;
            }
            if let Some(ref snap) = *guard {
                if snap.seq != since {
                    return Some(LatestJpeg {
                        bytes: snap.bytes.clone(),
                        seq: snap.seq,
                        pts_us: snap.pts_us,
                    });
                }
            }
            // Short timeout so the shutdown flag is also re-checked.
            self.notify
                .wait_for(&mut guard, Duration::from_millis(100));
        }
    }

    fn latest(&self) -> Option<LatestJpeg> {
        self.state.lock().as_ref().map(|s| LatestJpeg {
            bytes: s.bytes.clone(),
            seq: s.seq,
            pts_us: s.pts_us,
        })
    }
}

pub fn spawn(
    addr: SocketAddr,
    shared: SharedFrame,
    settings: Arc<Mutex<Settings>>,
    audio_source: Arc<Mutex<Option<Arc<AudioState>>>>,
) -> Result<Arc<RelayInfo>> {
    let listener = TcpListener::bind(addr).with_context(|| format!("failed to bind {addr}"))?;
    set_listener_nodelay(&listener);
    let server = Server::from_listener(listener, None)
        .map_err(|e| anyhow::anyhow!("failed to bind {addr}: {e}"))?;
    let actual = server.server_addr().to_ip().unwrap_or(addr);
    let lan_url = build_lan_url(actual);
    let local_url = format!("http://127.0.0.1:{}", actual.port());
    #[cfg(windows)]
    let ts = TsBroadcaster::new();
    let shutdown = Arc::new(AtomicBool::new(false));
    let audio = AudioRelay::spawn(audio_source, shutdown.clone())
        .context("failed to spawn relay audio thread")?;
    let info = Arc::new(RelayInfo {
        bind_addr: actual,
        lan_url,
        local_url,
        active_clients: AtomicUsize::new(0),
        total_clients: AtomicUsize::new(0),
        shutdown: shutdown.clone(),
        #[cfg(windows)]
        ts: ts.clone(),
        fmp4: Mutex::new(None),
        audio,
        video: VideoVariants::new(shared.clone(), settings.clone(), shutdown.clone()),
    });
    let jpeg = Arc::new(SharedJpeg::new());

    // NOTE: The MSMF software H.264 encoder behind /stream.ts is wip and
    // does not honour IDR / GOP requests reliably (see CLAUDE.md notes).
    // Worse, when it was auto-spawned at every relay start the encoder
    // ran continuously even with 0 subscribers, leaked Media Foundation
    // buffers on long sessions, and the OS eventually returned
    // E_OUTOFMEMORY (0x8007000E) on the next encode call, which surfaced
    // as a fast-fail abort (Windows 0xc0000409) with no Rust panic
    // hook to catch it. Repro on the dev machine: ~30 min of normal use
    // with relay running. fMP4 audio+video via ffmpeg has superseded this
    // path; keep the module compiled but stop spawning the encoder thread.
    let _ = shutdown.clone();

    // One encoder thread per relay; pays the NV12 -> RGB -> JPEG cost once
    // per frame and hands the result out to every connected client unchanged.
    let enc_shared = shared.clone();
    let enc_settings = settings.clone();
    let enc_jpeg = jpeg.clone();
    let enc_info = info.clone();
    std::thread::Builder::new()
        .name("relay-encoder".into())
        .spawn(move || encoder_loop(enc_shared, enc_settings, enc_jpeg, enc_info))
        .context("failed to spawn relay encoder thread")?;

    let info_for_thread = info.clone();
    let jpeg_for_thread = jpeg.clone();
    std::thread::Builder::new()
        .name("relay-accept".into())
        .spawn(move || accept_loop(server, jpeg_for_thread, info_for_thread))
        .context("failed to spawn relay accept thread")?;
    Ok(info)
}

/// Turn off Nagle on the listening socket; accepted sockets inherit it.
/// tiny_http never exposes the per-connection stream, and without this the
/// 10 ms audio chunks (and the tail of every JPEG) can sit in the send
/// buffer until the client's delayed ACK fires, up to 200 ms on Windows.
#[cfg(windows)]
fn set_listener_nodelay(listener: &TcpListener) {
    use std::os::windows::io::AsRawSocket;
    use windows_sys::Win32::Networking::WinSock::{IPPROTO_TCP, TCP_NODELAY, setsockopt};
    let on: i32 = 1;
    let rc = unsafe {
        setsockopt(
            listener.as_raw_socket() as usize,
            IPPROTO_TCP,
            TCP_NODELAY,
            &on as *const i32 as *const u8,
            std::mem::size_of::<i32>() as i32,
        )
    };
    if rc != 0 {
        log::warn!("relay: could not set TCP_NODELAY on listener");
    }
}

#[cfg(not(windows))]
fn set_listener_nodelay(_listener: &TcpListener) {}

fn build_lan_url(addr: SocketAddr) -> String {
    let port = addr.port();
    if !addr.ip().is_unspecified() {
        return format!("http://{}", addr);
    }
    let ip = local_ip().unwrap_or_else(|| IpAddr::V4(std::net::Ipv4Addr::LOCALHOST));
    if ip.is_loopback() {
        format!("http://localhost:{port}")
    } else {
        format!("http://{}:{}", ip, port)
    }
}

fn local_ip() -> Option<IpAddr> {
    let sock = UdpSocket::bind("0.0.0.0:0").ok()?;
    sock.connect("8.8.8.8:80").ok()?;
    sock.local_addr().ok().map(|a| a.ip())
}

fn encoder_loop(
    shared: SharedFrame,
    settings: Arc<Mutex<Settings>>,
    jpeg: Arc<SharedJpeg>,
    info: Arc<RelayInfo>,
) {
    let mut last_seq: u64 = 0;
    let mut last_quality: u8 = 0;
    let mut fps_window = std::time::Instant::now();
    let mut fps_frames = 0u32;
    while !info.shutdown.load(Ordering::Relaxed) {
        let frame = match shared.get() {
            Some(f) => f,
            None => {
                std::thread::sleep(Duration::from_millis(5));
                continue;
            }
        };
        let quality = settings.lock().jpeg_quality.clamp(1, 100);
        // Skip if the frame is the same one we already encoded at the same
        // quality. Quality changes force a re-encode of the latest frame.
        if frame.seq == last_seq && quality == last_quality {
            std::thread::sleep(Duration::from_millis(2));
            continue;
        }
        if frame.seq != last_seq {
            fps_frames += 1;
            let elapsed = fps_window.elapsed();
            if elapsed >= Duration::from_secs(2) {
                let fps = (fps_frames as f64 / elapsed.as_secs_f64()).round() as u32;
                info.video.source_fps.store(fps, Ordering::Relaxed);
                fps_frames = 0;
                fps_window = std::time::Instant::now();
            }
        }
        let rgb = frame_to_rgb(&frame.data, frame.width, frame.height);
        let bytes = match encode_jpeg(&rgb, frame.width, frame.height, quality) {
            Ok(b) => b,
            Err(e) => {
                log::warn!("relay encoder: {e}");
                std::thread::sleep(Duration::from_millis(50));
                continue;
            }
        };
        last_seq = frame.seq;
        last_quality = quality;
        jpeg.publish(LatestJpeg {
            bytes: Arc::new(bytes),
            seq: frame.seq,
            pts_us: audio_relay::pts_us(frame.captured_at),
        });
    }
    log::info!("relay encoder loop exiting");
}

fn accept_loop(server: Server, jpeg: Arc<SharedJpeg>, info: Arc<RelayInfo>) {
    while !info.shutdown.load(Ordering::Relaxed) {
        let request = match server.recv_timeout(Duration::from_millis(200)) {
            Ok(Some(r)) => r,
            Ok(None) => continue,
            Err(e) => {
                log::warn!("relay accept error: {e}");
                continue;
            }
        };
        let jpeg = jpeg.clone();
        let info = info.clone();
        let url = request.url().to_string();
        let method = request.method().clone();
        std::thread::Builder::new()
            .name(format!(
                "relay-{}",
                request.remote_addr().map(|a| a.to_string()).unwrap_or_default()
            ))
            .spawn(move || {
                if let Err(e) = handle(request, &url, &method, jpeg, info) {
                    log::debug!("client gone: {e}");
                }
            })
            .ok();
    }
    log::info!("relay accept loop exiting");
}

fn handle(
    request: tiny_http::Request,
    url: &str,
    method: &Method,
    jpeg: Arc<SharedJpeg>,
    info: Arc<RelayInfo>,
) -> Result<()> {
    if method != &Method::Get {
        let _ = request.respond(Response::from_string("method not allowed").with_status_code(405));
        return Ok(());
    }
    let (url, query) = url.split_once('?').unwrap_or((url, ""));
    match url {
        "/" | "/index.html" => serve_index(request),
        "/live" | "/live.html" => serve_live(request),
        "/stream" | "/stream.mjpg" => serve_mjpeg(request, jpeg, info),
        "/snapshot.jpg" => serve_snapshot(request, jpeg),
        "/audio.wav" => serve_wav(request, info),
        "/live/audio" => serve_live_audio(request, info),
        "/live/video" => serve_live_video(request, query, jpeg, info),
        "/live/info" => serve_live_info(request, info),
        #[cfg(windows)]
        "/stream.ts" => serve_mpegts(request, info),
        "/stream.mp4" => serve_fmp4(request, info),
        "/player" | "/player.html" => serve_fmp4_player(request),
        _ => {
            let _ = request.respond(Response::from_string("not found").with_status_code(404));
            Ok(())
        }
    }
}

#[cfg(windows)]
fn serve_mpegts(request: tiny_http::Request, info: Arc<RelayInfo>) -> Result<()> {
    use std::io::Write as IoWrite;
    let rx = info.ts.subscribe();
    let mut writer = request.into_writer();
    write!(writer, "HTTP/1.1 200 OK\r\n")?;
    write!(writer, "Content-Type: video/mp2t\r\n")?;
    write!(writer, "Cache-Control: no-store, no-cache, must-revalidate, max-age=0\r\n")?;
    write!(writer, "Pragma: no-cache\r\n")?;
    write!(writer, "Connection: close\r\n")?;
    write!(writer, "\r\n")?;
    writer.flush()?;

    info.active_clients.fetch_add(1, Ordering::Relaxed);
    info.total_clients.fetch_add(1, Ordering::Relaxed);
    let _guard = ClientGuard(info.clone());

    while !info.shutdown.load(Ordering::Relaxed) {
        match rx.recv_timeout(Duration::from_millis(500)) {
            Ok(chunk) => {
                writer.write_all(chunk.as_ref())?;
                writer.flush()?;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    Ok(())
}

/// Wait for the first audio chunk of a fresh subscription; clients need its
/// format before they can write a header. None if nothing arrives in time.
fn first_audio_chunk(info: &RelayInfo, sub: &mut AudioSubscription) -> Option<Arc<AudioChunk>> {
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    loop {
        if let Some(c) = sub.recv(Duration::from_millis(200)) {
            return Some(c);
        }
        if std::time::Instant::now() >= deadline || info.shutdown.load(Ordering::Relaxed) {
            return None;
        }
    }
}

fn audio_unavailable(request: tiny_http::Request, msg: &str) -> Result<()> {
    let _ = request.respond(Response::from_string(msg).with_status_code(503));
    Ok(())
}

const AUDIO_OFF_MSG: &str = "audio passthrough is off (enable audio in the vicash F1 panel)";
const AUDIO_SILENT_MSG: &str = "no audio samples arriving from the capture input";

fn write_stream_headers(writer: &mut dyn Write, content_type: &str) -> std::io::Result<()> {
    write!(writer, "HTTP/1.1 200 OK\r\n")?;
    write!(writer, "Content-Type: {content_type}\r\n")?;
    write!(writer, "Cache-Control: no-store, no-cache, must-revalidate, max-age=0\r\n")?;
    write!(writer, "Pragma: no-cache\r\n")?;
    write!(writer, "Connection: close\r\n")?;
    write!(writer, "\r\n")
}

/// Live capture audio as an endless 16-bit PCM WAV. Plays in VLC, ffplay or
/// an OBS Media Source, but carries no timestamps, so it is not synced to
/// the picture; /live is. The connection is closed when the sample format
/// changes (different input device); clients just reconnect.
fn serve_wav(request: tiny_http::Request, info: Arc<RelayInfo>) -> Result<()> {
    if !info.audio.is_available() {
        return audio_unavailable(request, AUDIO_OFF_MSG);
    }
    let mut sub = info.audio.subscribe();
    let Some(first) = first_audio_chunk(&info, &mut sub) else {
        return audio_unavailable(request, AUDIO_SILENT_MSG);
    };
    let (rate, channels) = (first.sample_rate, first.channels);

    let mut writer = request.into_writer();
    write_stream_headers(&mut writer, "audio/wav")?;
    writer.write_all(&audio_relay::wav_header(rate, channels))?;
    writer.write_all(&first.pcm)?;
    writer.flush()?;

    info.active_clients.fetch_add(1, Ordering::Relaxed);
    info.total_clients.fetch_add(1, Ordering::Relaxed);
    let _guard = ClientGuard(info.clone());

    while !info.shutdown.load(Ordering::Relaxed) {
        let Some(chunk) = sub.recv(Duration::from_millis(500)) else {
            continue;
        };
        if chunk.sample_rate != rate || chunk.channels != channels {
            log::info!("relay audio: format changed, closing /audio.wav client");
            break;
        }
        writer.write_all(&chunk.pcm)?;
        writer.flush()?;
    }
    Ok(())
}

/// Timestamped audio for the /live player. A 12 byte stream header
/// (`VCA1`, sample rate u32, channels u16, reserved u16), then one frame per
/// chunk: pts u64 in microseconds on the relay clock, byte length u32, s16le
/// interleaved PCM. Everything little endian. Closed on format change.
fn serve_live_audio(request: tiny_http::Request, info: Arc<RelayInfo>) -> Result<()> {
    if !info.audio.is_available() {
        return audio_unavailable(request, AUDIO_OFF_MSG);
    }
    let mut sub = info.audio.subscribe();
    let Some(first) = first_audio_chunk(&info, &mut sub) else {
        return audio_unavailable(request, AUDIO_SILENT_MSG);
    };
    let (rate, channels) = (first.sample_rate, first.channels);

    let mut writer = request.into_writer();
    write_stream_headers(&mut writer, "application/octet-stream")?;
    writer.write_all(b"VCA1")?;
    writer.write_all(&rate.to_le_bytes())?;
    writer.write_all(&channels.to_le_bytes())?;
    writer.write_all(&0u16.to_le_bytes())?;

    info.active_clients.fetch_add(1, Ordering::Relaxed);
    info.total_clients.fetch_add(1, Ordering::Relaxed);
    let _guard = ClientGuard(info.clone());

    let mut next = Some(first);
    while !info.shutdown.load(Ordering::Relaxed) {
        let chunk = match next.take() {
            Some(c) => c,
            None => match sub.recv(Duration::from_millis(500)) {
                Some(c) => c,
                None => continue,
            },
        };
        if chunk.sample_rate != rate || chunk.channels != channels {
            log::info!("relay audio: format changed, closing /live/audio client");
            break;
        }
        writer.write_all(&chunk.pts_us.to_le_bytes())?;
        writer.write_all(&(chunk.pcm.len() as u32).to_le_bytes())?;
        writer.write_all(&chunk.pcm)?;
        writer.flush()?;
    }
    Ok(())
}

/// Timestamped MJPEG for the /live player: per frame pts u64 in
/// microseconds on the relay clock (capture time), byte length u32, JPEG.
/// Little endian. `?h=720` picks a downscaled variant (anything at or above
/// the capture height is the full-resolution stream), `?fps=N` caps the
/// rate; frames in between are skipped, never queued.
fn serve_live_video(
    request: tiny_http::Request,
    query: &str,
    jpeg: Arc<SharedJpeg>,
    info: Arc<RelayInfo>,
) -> Result<()> {
    let param = |name: &str| {
        query
            .split('&')
            .find_map(|kv| kv.strip_prefix(name)?.strip_prefix('='))
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| *v > 0.0)
    };
    let fps = param("fps");
    let mut pacer = fps.map(Pacer::new);

    let src_h = info.video.source_size().map_or(0, |(_, h)| h);
    let lease = param("h")
        .map(|h| h as u32)
        .filter(|h| src_h > 0 && *h < src_h)
        .map(|h| info.video.lease(h, fps));
    let jpeg = lease.as_ref().map_or(jpeg, |l| l.variant.jpeg.clone());

    let mut writer = request.into_writer();
    write_stream_headers(&mut writer, "application/octet-stream")?;
    writer.flush()?;

    info.active_clients.fetch_add(1, Ordering::Relaxed);
    info.total_clients.fetch_add(1, Ordering::Relaxed);
    let _guard = ClientGuard(info.clone());

    let mut last_seq: u64 = 0;
    while !info.shutdown.load(Ordering::Relaxed) {
        let Some(snap) = jpeg.wait_for_new(last_seq, &info.shutdown) else {
            return Ok(());
        };
        last_seq = snap.seq;
        if pacer.as_mut().is_some_and(|p| !p.admit()) {
            continue;
        }
        writer.write_all(&snap.pts_us.to_le_bytes())?;
        writer.write_all(&(snap.bytes.len() as u32).to_le_bytes())?;
        writer.write_all(snap.bytes.as_ref())?;
        writer.flush()?;
    }
    Ok(())
}

/// Capture size and rate, so the /live player only offers qualities the
/// source can actually deliver.
fn serve_live_info(request: tiny_http::Request, info: Arc<RelayInfo>) -> Result<()> {
    let (w, h) = info.video.source_size().unwrap_or((0, 0));
    let fps = info.video.source_fps.load(Ordering::Relaxed);
    let body = format!(
        r#"{{"width":{w},"height":{h},"fps":{fps},"audio":{}}}"#,
        info.audio.is_available()
    );
    let header = Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap();
    let no_store = Header::from_bytes(&b"Cache-Control"[..], &b"no-store"[..]).unwrap();
    request.respond(Response::from_string(body).with_header(header).with_header(no_store))?;
    Ok(())
}

/// Synced video + audio player for phones, browsers and OBS Browser Sources.
///
/// Both streams carry timestamps on the same relay clock. Audio is the
/// master: chunks are scheduled back to back on the Web Audio timeline, so
/// the page always knows which capture instant is coming out of the speaker
/// right now (output latency of the device included), and draws the video
/// frame with the matching timestamp. If frames arrive too late for that
/// (slow WiFi, slow JPEG decode on a phone) the audio buffer grows until
/// they fit, and shrinks again once there is slack.
///
/// Deliberately no AudioWorklet: browsers only expose it on HTTPS or
/// localhost, and a phone opening http://192.168.x.x is neither.
///
/// A gear button (hidden until the pointer moves, so it never shows up in an
/// OBS capture) offers YouTube style qualities from the capture resolution
/// down to 360p, each at 60 or 30 fps. The choice is remembered per browser.
///
/// Query options: `q=720p30` fixed quality (for OBS), `ui=0` no gear,
/// `buffer=<ms>` minimum audio buffer (default 80), `offset=<ms>` manual
/// nudge (positive shows the picture later), `video=0` / `audio=0`,
/// `stats` overlay.
fn serve_live(request: tiny_http::Request) -> Result<()> {
    let html = r#"<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>vicash live</title>
<style>
  html,body{margin:0;background:#000;height:100%;overflow:hidden;font-family:system-ui,sans-serif}
  canvas{display:block;width:100%;height:100%;object-fit:contain}
  #tap{position:fixed;inset:0;display:none;align-items:center;justify-content:center;text-align:center;
       color:#cfe;font-size:22px;line-height:1.5;background:rgba(0,0,0,.45);cursor:pointer}
  #stats{position:fixed;left:8px;top:8px;display:none;white-space:pre;color:#9ad;
         font:12px/1.4 ui-monospace,monospace;background:rgba(0,0,0,.6);padding:4px 8px;border-radius:4px}
  #gear{position:fixed;right:14px;bottom:14px;z-index:5;width:46px;height:46px;border:0;border-radius:50%;
        background:rgba(0,0,0,.65);color:#fff;font-size:24px;line-height:46px;cursor:pointer;
        opacity:0;transition:opacity .3s;-webkit-tap-highlight-color:transparent}
  #gear.show{opacity:1}
  #menu{position:fixed;right:14px;bottom:70px;z-index:5;display:none;min-width:220px;max-height:70vh;overflow:auto;
        background:rgba(22,22,26,.94);border-radius:10px;padding:6px 0;color:#eee;font-size:15px;
        box-shadow:0 6px 24px rgba(0,0,0,.5)}
  #menu.open{display:block}
  #menu .h{padding:6px 16px 8px;color:#99a;font-size:12px;text-transform:uppercase;letter-spacing:.06em}
  #menu button{display:flex;justify-content:space-between;align-items:baseline;gap:18px;width:100%;
               padding:10px 16px;border:0;background:none;color:inherit;font:inherit;text-align:left;cursor:pointer}
  #menu button:hover{background:rgba(255,255,255,.08)}
  #menu button.on{color:#6cf;font-weight:600}
  #menu small{color:#889;font-weight:400}
</style></head>
<body>
<canvas id="v"></canvas>
<div id="tap">Tippen f&uuml;r Ton<br><small>tap for sound</small></div>
<div id="stats"></div>
<div id="menu"></div>
<button id="gear" aria-label="Qualit&auml;t">&#9881;</button>
<script>
'use strict';
const q = new URLSearchParams(location.search);
const baseBuffer = Math.min(1, Math.max(0.02, (+q.get('buffer') || 80) / 1000));
const offsetUs = (+q.get('offset') || 0) * 1000;
const wantVideo = q.get('video') !== '0';
const wantAudio = q.get('audio') !== '0';
const sleep = ms => new Promise(r => setTimeout(r, ms));
const canvas = document.getElementById('v');
const g = canvas.getContext('2d');
const tap = document.getElementById('tap');
const stats = document.getElementById('stats');
if (q.has('stats')) stats.style.display = 'block';
if (!wantVideo) canvas.style.display = 'none';
// iOS: play through the silent switch like a video would.
try { if (navigator.audioSession) navigator.audioSession.type = 'playback'; } catch (e) {}

class ByteReader {
  constructor(body) { this.r = body.getReader(); this.chunks = []; this.len = 0; }
  async take(n) {
    while (this.len < n) {
      const { done, value } = await this.r.read();
      if (done) throw new Error('stream ended');
      this.chunks.push(value); this.len += value.length;
    }
    const out = new Uint8Array(n);
    let o = 0;
    while (o < n) {
      const c = this.chunks[0], k = Math.min(c.length, n - o);
      out.set(c.subarray(0, k), o); o += k;
      if (k === c.length) this.chunks.shift(); else this.chunks[0] = c.subarray(k);
    }
    this.len -= n;
    return out;
  }
  cancel() { this.r.cancel().catch(() => {}); }
}
const dv = b => new DataView(b.buffer, b.byteOffset, b.byteLength);
const u64 = (b, o) => dv(b).getUint32(o, true) + dv(b).getUint32(o + 4, true) * 4294967296;

// ---------- audio: master clock ----------
let actx = null;
let nextTime = 0;          // context time where the next chunk starts, 0 = not running
let target = baseBuffer;   // wanted lead of scheduled audio, grows when video lags
const marks = [];          // scheduled chunks: {at, end, pts}
let leadMin = Infinity, leadSince = 0;

const outLatency = () => actx ? (actx.outputLatency || 0) + (actx.baseLatency || 0) : 0;

function updateTap() {
  tap.style.display = actx && actx.state !== 'running' ? 'flex' : 'none';
}

function unlock() {
  if (!actx) return;
  actx.resume().catch(() => {});
  const b = actx.createBuffer(1, 1, actx.sampleRate);
  const s = actx.createBufferSource();
  s.buffer = b; s.connect(actx.destination); s.start();
}
tap.addEventListener('click', unlock);

function ensureCtx(rate) {
  if (actx && actx.sampleRate === rate) return;
  if (actx) actx.close();
  try { actx = new AudioContext({ sampleRate: rate, latencyHint: 'interactive' }); }
  catch (e) { actx = new AudioContext({ latencyHint: 'interactive' }); }
  marks.length = 0; nextTime = 0;
  actx.onstatechange = updateTap;
  actx.resume().catch(() => {});
  setTimeout(updateTap, 300);
}

// Relay-clock timestamp of the sound leaving the speaker right now.
function audioClock() {
  if (!actx || !marks.length) return null;
  const ct = actx.currentTime - outLatency();
  while (marks.length > 1 && marks[0].end <= ct) marks.shift();
  const m = marks[0];
  return m.pts + (Math.min(ct, m.end) - m.at) * 1e6;
}

function audioLive() {
  return !!actx && actx.state === 'running' && marks.length > 0
    && actx.currentTime - marks[marks.length - 1].end < 0.5;
}

function schedule(pts, pcm, rate, ch) {
  if (actx.state !== 'running') { nextTime = 0; marks.length = 0; return; }
  const frames = (pcm.length / (2 * ch)) | 0;
  if (!frames) return;
  const now = actx.currentTime, dur = frames / rate;
  if (!nextTime || nextTime - now < 0.003) {
    // Start or underrun: rebuffer to the target lead.
    nextTime = now + target;
    leadMin = Infinity; leadSince = now;
  }
  const lead = nextTime - now;
  if (lead > target + 0.3) return;          // way behind after a stall: drop
  leadMin = Math.min(leadMin, lead);
  if (now - leadSince > 2) {
    // Once per window, nudge the lowest lead seen towards the target.
    // Network jitter makes the lead wobble; only its floor matters.
    const floor = leadMin;
    leadMin = Infinity; leadSince = now;
    if (floor > target + 0.03) return;      // drop one chunk
    if (floor < target - 0.03) nextTime += target - floor;
  }
  const s16 = new Int16Array(pcm.buffer, pcm.byteOffset, frames * ch);
  const buf = actx.createBuffer(ch, frames, rate);
  for (let c = 0; c < ch; c++) {
    const d = buf.getChannelData(c);
    for (let i = 0, j = c; i < frames; i++, j += ch) d[i] = s16[j] / 32768;
  }
  const src = actx.createBufferSource();
  src.buffer = buf;
  src.connect(actx.destination);
  src.start(nextTime);
  marks.push({ at: nextTime, end: nextTime + dur, pts });
  nextTime += dur;
}

async function runAudio() {
  const res = await fetch('/live/audio', { cache: 'no-store' });
  if (!res.ok) throw new Error('audio http ' + res.status);
  const br = new ByteReader(res.body);
  try {
    const h = await br.take(12);
    if (String.fromCharCode(h[0], h[1], h[2], h[3]) !== 'VCA1') throw new Error('bad audio header');
    const rate = dv(h).getUint32(4, true), ch = dv(h).getUint16(8, true);
    ensureCtx(rate);
    nextTime = 0;
    while (true) {
      const hd = await br.take(12);
      const pcm = await br.take(dv(hd).getUint32(8, true));
      schedule(u64(hd, 0), pcm, rate, ch);
    }
  } finally {
    br.cancel();
  }
}

// ---------- video: follows the audio clock ----------
const frames = [];         // decoded, waiting for their moment: {pts, bmp}
let lastLate = 0, lastRaise = 0, slackMax = -Infinity, slackSince = 0, vCount = 0, vFps = 0, vBytes = 0, vMbit = 0;

// ---------- quality menu ----------
const NAMES = { 2160: '4K', 1440: '2K', 1080: 'Full HD', 720: 'HD' };
const STEPS = [1440, 1080, 720, 480, 360];
const gear = document.getElementById('gear');
const menu = document.getElementById('menu');
let src = null;            // capture {width, height, fps} from /live/info
let current = null;        // quality the running video stream was opened with
let videoAbort = null;

const parseQuality = s => {
  const m = /^(\d+)p(\d+)?$/.exec(s || '');
  return m ? { h: +m[1], fps: +(m[2] || 60) } : null;
};
let quality = parseQuality(q.get('q'));
if (!quality) { try { quality = parseQuality(localStorage.getItem('vicash.quality')); } catch (e) {} }

// Everything the source can deliver, best first. No upscaling.
function options() {
  if (!src || !src.height) return [];
  const heights = [src.height].concat(STEPS.filter(h => h < src.height));
  const rates = src.fps && src.fps < 45 ? [30] : [60, 30];
  const out = [];
  for (const h of heights) for (const fps of rates) out.push({ h, fps });
  return out;
}

// The stored wish mapped onto what this source offers: same or next lower
// resolution, same rate if available.
function effective() {
  const opts = options();
  if (!opts.length) return { h: 0, fps: 60 };
  if (!quality) return opts[0];
  const fit = opts.filter(o => o.h <= quality.h);
  const pool = fit.length ? fit : opts.slice(-2);
  return pool.find(o => o.h === pool[0].h && o.fps === quality.fps) || pool[0];
}

function buildMenu() {
  menu.innerHTML = '<div class="h">Qualit&auml;t</div>';
  const cur = current || effective();
  for (const o of options()) {
    const b = document.createElement('button');
    if (o.h === cur.h && o.fps === cur.fps) b.className = 'on';
    const tag = o.h === src.height ? 'Original' : (NAMES[o.h] || '');
    const fps = src.fps ? Math.min(o.fps, src.fps) : o.fps;
    b.innerHTML = '<span>' + o.h + 'p ' + (tag ? '<small>' + tag + '</small>' : '') + '</span><span>' + fps + ' fps</span>';
    b.onclick = e => {
      e.stopPropagation();
      quality = o;
      try { localStorage.setItem('vicash.quality', o.h + 'p' + o.fps); } catch (err) {}
      menu.classList.remove('open');
      if (videoAbort) videoAbort.abort();
    };
    menu.appendChild(b);
  }
}

let hideTimer = 0;
function poke() {
  gear.classList.add('show');
  clearTimeout(hideTimer);
  hideTimer = setTimeout(() => { if (!menu.classList.contains('open')) gear.classList.remove('show'); }, 3000);
}
if (q.get('ui') === '0' || !wantVideo) {
  gear.style.display = 'none';
} else {
  for (const ev of ['pointermove', 'pointerdown', 'touchstart']) document.addEventListener(ev, poke, { passive: true });
  gear.addEventListener('click', async e => {
    e.stopPropagation();
    unlock();
    await refreshInfo();
    buildMenu();
    menu.classList.toggle('open');
    poke();
  });
  document.addEventListener('click', () => menu.classList.remove('open'));
}

async function refreshInfo() {
  try {
    const r = await fetch('/live/info', { cache: 'no-store' });
    if (r.ok) src = await r.json();
  } catch (e) {}
}

// How late a freshly decoded frame is against the audio it belongs to.
// Late frames make the audio wait longer; lots of slack lets it catch up.
function noteLateness(pts) {
  if (!audioLive()) return;
  const t = audioClock();
  if (t === null) return;
  const late = (t - (pts + offsetUs)) / 1e6;
  const now = performance.now();
  lastLate = late;
  if (late > -0.01 && now - lastRaise > 1000 && target < 1) {
    const add = Math.min(1 - target, late + 0.03);
    target += add;
    if (nextTime) nextTime += add;
    leadMin = Infinity; leadSince = actx.currentTime;
    lastRaise = now; slackSince = now; slackMax = -Infinity;
    return;
  }
  slackMax = Math.max(slackMax, late);
  if (now - slackSince > 5000) {
    if (slackMax < -0.08 && target > baseBuffer) {
      target = Math.max(baseBuffer, target - Math.min(0.05, -slackMax - 0.04));
    }
    slackSince = now; slackMax = -Infinity;
  }
}

async function runVideo() {
  await refreshInfo();
  const want = effective();
  let url = '/live/video?h=' + want.h;
  if (want.fps === 30 && !(src && src.fps && src.fps < 45)) url += '&fps=30';
  videoAbort = new AbortController();
  const res = await fetch(url, { cache: 'no-store', signal: videoAbort.signal });
  if (!res.ok) throw new Error('video http ' + res.status);
  current = want;
  const br = new ByteReader(res.body);
  try {
    while (true) {
      const hd = await br.take(12);
      const pts = u64(hd, 0);
      const len = dv(hd).getUint32(8, true);
      const jpg = await br.take(len);
      vBytes += len + 12;
      let bmp;
      try { bmp = await createImageBitmap(new Blob([jpg], { type: 'image/jpeg' })); } catch (e) { continue; }
      noteLateness(pts);
      frames.push({ pts, bmp });
      if (frames.length > 180) frames.shift().bmp.close();
      vCount++;
    }
  } finally {
    br.cancel();
  }
}

function render() {
  requestAnimationFrame(render);
  let pick = null;
  const take = () => { if (pick) pick.bmp.close(); pick = frames.shift(); };
  const t = audioLive() ? audioClock() : null;
  if (t !== null) {
    while (frames.length && frames[0].pts + offsetUs <= t) take();
    // Clocks far apart (audio restarted, host restarted): resync on newest.
    if (!pick && frames.length && frames[0].pts + offsetUs - t > 2e6) while (frames.length) take();
  } else {
    // No sound yet: show the picture as fast as possible.
    while (frames.length) take();
  }
  if (!pick) return;
  if (canvas.width !== pick.bmp.width || canvas.height !== pick.bmp.height) {
    canvas.width = pick.bmp.width; canvas.height = pick.bmp.height;
  }
  g.drawImage(pick.bmp, 0, 0);
  pick.bmp.close();
}

function loop(fn, name) {
  (async () => {
    while (true) {
      try { await fn(); } catch (e) {
        // A quality switch aborts the stream on purpose: reconnect at once.
        if (e && e.name === 'AbortError') continue;
        console.warn('vicash ' + name + ':', e);
      }
      await sleep(1000);
    }
  })();
}

if (wantVideo) { requestAnimationFrame(render); loop(runVideo, 'video'); }
if (wantAudio) loop(runAudio, 'audio');

setInterval(() => {
  vFps = vCount * 2; vCount = 0;
  vMbit = vBytes * 2 * 8 / 1e6; vBytes = 0;
  if (stats.style.display === 'none') return;
  const ms = s => (s * 1000).toFixed(0);
  stats.textContent =
    'qualitaet ' + (current ? current.h + 'p' + current.fps : '-') + '   ' + vMbit.toFixed(1) + ' Mbit/s\n' +
    'sync      ' + (audioLive() ? 'an' : 'aus') + '   audio ' + (actx ? actx.state : '-') + '\n' +
    'puffer    ' + ms(target) + ' ms   lead ' + (actx && nextTime ? ms(nextTime - actx.currentTime) : '-') + ' ms\n' +
    'ausgabe   ' + ms(outLatency()) + ' ms\n' +
    'video     ' + vFps + ' fps   spaet ' + ms(lastLate) + ' ms';
}, 500);
</script>
</body></html>"#;
    let header = Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=utf-8"[..]).unwrap();
    request.respond(Response::from_string(html).with_header(header))?;
    Ok(())
}

/// Stream the fragmented-MP4 produced by the ffmpeg subprocess. Sends the
/// cached init segment first so the browser's MSE buffer can warm up, then
/// loops on the broadcaster's condvar pushing each fresh media segment to
/// the client. Returns when the client disconnects or the relay shuts down.
fn serve_fmp4(request: tiny_http::Request, info: Arc<RelayInfo>) -> Result<()> {
    use std::io::Write as IoWrite;
    // Snapshot the Arc into a local so we keep the relay alive for the
    // duration of this response even if the user toggles it off mid-stream.
    let relay = info.fmp4.lock().clone();
    let Some(relay) = relay else {
        let _ = request.respond(
            Response::from_string("fMP4 relay not enabled (toggle 'Audio im Relay' in F1, needs ffmpeg.exe)")
                .with_status_code(503),
        );
        return Ok(());
    };

    // The init segment may not be ready yet if the encoder just started.
    // Briefly wait for it; if it never arrives, return a clean error so
    // browsers do not see truncated MP4 bytes.
    let mut init = relay.state.init_segment();
    let init_deadline = std::time::Instant::now() + Duration::from_secs(3);
    while init.is_none() && std::time::Instant::now() < init_deadline {
        std::thread::sleep(Duration::from_millis(50));
        if relay.state.shutdown.load(Ordering::Relaxed) {
            break;
        }
        init = relay.state.init_segment();
    }
    let Some(init) = init else {
        let _ = request.respond(
            Response::from_string("fMP4 init segment not ready - is ffmpeg still warming up?")
                .with_status_code(503),
        );
        return Ok(());
    };

    let mut writer = request.into_writer();
    write!(writer, "HTTP/1.1 200 OK\r\n")?;
    write!(writer, "Content-Type: video/mp4\r\n")?;
    write!(writer, "Cache-Control: no-store, no-cache, must-revalidate, max-age=0\r\n")?;
    write!(writer, "Pragma: no-cache\r\n")?;
    write!(writer, "Connection: close\r\n")?;
    write!(writer, "\r\n")?;
    writer.write_all(&init)?;
    writer.flush()?;

    info.active_clients.fetch_add(1, Ordering::Relaxed);
    info.total_clients.fetch_add(1, Ordering::Relaxed);
    let _guard = ClientGuard(info.clone());

    let mut last_seen: u64 = 0;
    while !info.shutdown.load(Ordering::Relaxed)
        && !relay.state.shutdown.load(Ordering::Relaxed)
    {
        match relay.state.wait_for_next(last_seen, Duration::from_millis(500)) {
            Some((segment, seq)) => {
                last_seen = seq;
                writer.write_all(&segment)?;
                writer.flush()?;
            }
            None => continue,
        }
    }
    Ok(())
}

/// Serve a tiny HTML page that pulls /stream.mp4 through Media Source
/// Extensions and aggressively trims its own buffer so latency stays near
/// the fragment duration (~0.5-1s) instead of the multi-second default a
/// raw `<video src="/stream.mp4">` would buffer. Use this URL in OBS
/// Browser Source for the lowest practical end-to-end delay.
fn serve_fmp4_player(request: tiny_http::Request) -> Result<()> {
    let html = r#"<!doctype html>
<html><head><meta charset="utf-8"><title>vicash fMP4</title>
<style>
  html,body{margin:0;background:#000;height:100%}
  video{display:block;width:100%;height:100%;object-fit:contain;background:#000}
</style></head>
<body>
<video id="v" autoplay muted playsinline></video>
<script>
const v = document.getElementById('v');
const ms = new MediaSource();
v.src = URL.createObjectURL(ms);
ms.addEventListener('sourceopen', async () => {
  const sb = ms.addSourceBuffer('video/mp4; codecs="avc1.42E01E,mp4a.40.2"');
  sb.mode = 'sequence';
  const res = await fetch('/stream.mp4');
  const reader = res.body.getReader();
  const wait = () => new Promise(r => sb.updating ? sb.addEventListener('updateend', r, {once:true}) : r());
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    await wait();
    sb.appendBuffer(value);
    if (sb.buffered.length) {
      const start = sb.buffered.start(0);
      const end = sb.buffered.end(0);
      if (end - start > 3) {
        await wait();
        sb.remove(start, end - 1);
      }
    }
  }
});
setInterval(() => {
  if (v.buffered.length) {
    const live = v.buffered.end(v.buffered.length - 1);
    if (live - v.currentTime > 1.2) v.currentTime = live - 0.3;
  }
}, 400);
v.play().catch(() => {});
</script>
</body></html>"#;
    let header = Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=utf-8"[..]).unwrap();
    request.respond(Response::from_string(html).with_header(header))?;
    Ok(())
}

fn serve_index(request: tiny_http::Request) -> Result<()> {
    let html = r#"<!doctype html>
<html><head><meta charset="utf-8"><title>vicash</title>
<style>
  html,body{margin:0;background:#000;height:100%;font-family:system-ui,sans-serif;color:#9ad}
  img{display:block;width:100%;height:100%;object-fit:contain}
  .help{position:fixed;left:12px;bottom:12px;background:rgba(0,0,0,.55);
        padding:8px 12px;border-radius:6px;font-size:13px;line-height:1.5;
        pointer-events:none;backdrop-filter:blur(4px)}
  .help code{color:#cfe;background:rgba(255,255,255,.06);padding:1px 5px;border-radius:3px}
</style></head>
<body>
<img src="/stream" alt="capture">
<div class="help">
  vicash live stream<br>
  Video + audio in sync (phone, OBS): <code>/live</code><br>
  Direct MJPEG: <code>/stream</code><br>
  Audio only, not synced: <code>/audio.wav</code><br>
  Single frame: <code>/snapshot.jpg</code>
</div>
</body></html>"#;
    let header = Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=utf-8"[..]).unwrap();
    request.respond(Response::from_string(html).with_header(header))?;
    Ok(())
}

fn serve_snapshot(request: tiny_http::Request, jpeg: Arc<SharedJpeg>) -> Result<()> {
    let Some(snap) = jpeg.latest() else {
        let _ = request.respond(Response::from_string("no frame yet").with_status_code(503));
        return Ok(());
    };
    let header = Header::from_bytes(&b"Content-Type"[..], &b"image/jpeg"[..]).unwrap();
    request.respond(Response::from_data(snap.bytes.as_ref().clone()).with_header(header))?;
    Ok(())
}

fn serve_mjpeg(
    request: tiny_http::Request,
    jpeg: Arc<SharedJpeg>,
    info: Arc<RelayInfo>,
) -> Result<()> {
    let mut writer = request.into_writer();
    write!(writer, "HTTP/1.1 200 OK\r\n")?;
    write!(writer, "Content-Type: multipart/x-mixed-replace; boundary={BOUNDARY}\r\n")?;
    write!(writer, "Cache-Control: no-store, no-cache, must-revalidate, max-age=0\r\n")?;
    write!(writer, "Pragma: no-cache\r\n")?;
    write!(writer, "Connection: close\r\n")?;
    write!(writer, "\r\n")?;
    writer.flush()?;

    info.active_clients.fetch_add(1, Ordering::Relaxed);
    info.total_clients.fetch_add(1, Ordering::Relaxed);
    let _guard = ClientGuard(info.clone());

    let mut last_seq: u64 = 0;
    while !info.shutdown.load(Ordering::Relaxed) {
        let Some(snap) = jpeg.wait_for_new(last_seq, &info.shutdown) else {
            return Ok(());
        };
        last_seq = snap.seq;
        write!(
            writer,
            "--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
            snap.bytes.len()
        )?;
        writer.write_all(snap.bytes.as_ref())?;
        writer.write_all(b"\r\n")?;
        writer.flush()?;
    }
    Ok(())
}

/// Decrements active_clients when a streaming connection drops, regardless of
/// how it exited (clean close, broken pipe, our own loop returning).
struct ClientGuard(Arc<RelayInfo>);

impl Drop for ClientGuard {
    fn drop(&mut self) {
        self.0.active_clients.fetch_sub(1, Ordering::Relaxed);
    }
}

pub(crate) fn encode_jpeg(rgb: &[u8], w: u32, h: u32, quality: u8) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(w as usize * h as usize / 4);
    let encoder = JpegEncoder::new_with_quality(&mut out, quality);
    encoder.write_image(rgb, w, h, ColorType::Rgb8.into())?;
    Ok(out)
}

fn frame_to_rgb(data: &FrameData, w: u32, h: u32) -> Vec<u8> {
    match data {
        FrameData::Rgb(b) => b.as_ref().clone(),
        FrameData::Nv12(b) => nv12_to_rgb(b.as_ref(), w, h),
    }
}

fn nv12_to_rgb(nv12: &[u8], width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    if nv12.len() < w * h * 3 / 2 {
        return vec![0u8; w * h * 3];
    }
    let y_plane = &nv12[..w * h];
    let uv_plane = &nv12[w * h..];
    let mut rgb = vec![0u8; w * h * 3];
    for row in 0..h {
        let uv_row = row / 2;
        for col in 0..w {
            let uv_col = col & !1;
            let y = y_plane[row * w + col] as f32;
            let u = uv_plane[uv_row * w + uv_col] as f32;
            let v = uv_plane[uv_row * w + uv_col + 1] as f32;
            let yt = (y - 16.0) * (255.0 / 219.0);
            let ut = (u - 128.0) * (255.0 / 224.0);
            let vt = (v - 128.0) * (255.0 / 224.0);
            let r = yt + 1.5748 * vt;
            let g = yt - 0.1873 * ut - 0.4681 * vt;
            let b = yt + 1.8556 * ut;
            let idx = (row * w + col) * 3;
            rgb[idx] = r.clamp(0.0, 255.0) as u8;
            rgb[idx + 1] = g.clamp(0.0, 255.0) as u8;
            rgb[idx + 2] = b.clamp(0.0, 255.0) as u8;
        }
    }
    rgb
}
