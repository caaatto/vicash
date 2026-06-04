use crate::fmp4_relay::Fmp4Relay;
use crate::frame::{FrameData, SharedFrame};
use crate::settings::Settings;
#[cfg(windows)]
use crate::video_stream::{self, TsBroadcaster};
use anyhow::{Context, Result};
use image::codecs::jpeg::JpegEncoder;
use image::{ColorType, ImageEncoder};
use parking_lot::{Condvar, Mutex};
use std::io::Write;
use std::net::{IpAddr, SocketAddr, UdpSocket};
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
struct LatestJpeg {
    bytes: Arc<Vec<u8>>,
    seq: u64,
}

struct SharedJpeg {
    state: Mutex<Option<LatestJpeg>>,
    notify: Condvar,
}

impl SharedJpeg {
    fn new() -> Self {
        Self { state: Mutex::new(None), notify: Condvar::new() }
    }

    fn publish(&self, snapshot: LatestJpeg) {
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
        })
    }
}

pub fn spawn(
    addr: SocketAddr,
    shared: SharedFrame,
    settings: Arc<Mutex<Settings>>,
) -> Result<Arc<RelayInfo>> {
    let server = Server::http(addr)
        .map_err(|e| anyhow::anyhow!("failed to bind {addr}: {e}"))?;
    let actual = server.server_addr().to_ip().unwrap_or(addr);
    let lan_url = build_lan_url(actual);
    let local_url = format!("http://127.0.0.1:{}", actual.port());
    #[cfg(windows)]
    let ts = TsBroadcaster::new();
    let shutdown = Arc::new(AtomicBool::new(false));
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
    });
    let jpeg = Arc::new(SharedJpeg::new());

    // Bring the H.264 + MPEG-TS pipeline up alongside the MJPEG path. On an
    // NVENC/QSV/AMF capable machine this is basically free; the software
    // fallback uses real CPU, so users on low-end machines may want to
    // ignore the /stream.ts endpoint.
    #[cfg(windows)]
    {
        let bitrate = 6_000_000;
        video_stream::spawn(shared.clone(), ts.clone(), shutdown.clone(), bitrate);
    }

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
        jpeg.publish(LatestJpeg { bytes: Arc::new(bytes), seq: frame.seq });
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
    match url {
        "/" | "/index.html" => serve_index(request),
        "/stream" | "/stream.mjpg" => serve_mjpeg(request, jpeg, info),
        "/snapshot.jpg" => serve_snapshot(request, jpeg),
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
  Direct MJPEG: <code>/stream</code><br>
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

fn encode_jpeg(rgb: &[u8], w: u32, h: u32, quality: u8) -> Result<Vec<u8>> {
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
