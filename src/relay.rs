use crate::frame::{FrameData, SharedFrame};
use crate::settings::Settings;
use anyhow::{Context, Result};
use image::codecs::jpeg::JpegEncoder;
use image::{ColorType, ImageEncoder};
use parking_lot::Mutex;
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
    /// Best-effort LAN IP URL. The bind address may be 0.0.0.0; this is what
    /// a second PC on the same network would actually dial.
    pub lan_url: String,
    /// Localhost URL. Same machine only, handy for a quick browser check.
    pub local_url: String,
    pub active_clients: AtomicUsize,
    pub total_clients: AtomicUsize,
    /// Flipped to true to ask the accept loop to stop after its next
    /// recv_timeout wakeup. Holding the Arc keeps the loop alive; dropping
    /// the last clone of the info is what guarantees shutdown propagation.
    pub shutdown: AtomicBool,
}

impl RelayInfo {
    pub fn stop(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
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
    let info = Arc::new(RelayInfo {
        bind_addr: actual,
        lan_url,
        local_url,
        active_clients: AtomicUsize::new(0),
        total_clients: AtomicUsize::new(0),
        shutdown: AtomicBool::new(false),
    });
    let info_for_thread = info.clone();
    std::thread::Builder::new()
        .name("relay-accept".into())
        .spawn(move || accept_loop(server, shared, settings, info_for_thread))
        .context("failed to spawn relay accept thread")?;
    Ok(info)
}

/// Guess the LAN-routable URL of the host. Done by opening a UDP socket and
/// asking the OS which local interface it would use to reach an external
/// address; that interface's IP is the one a peer on the same network sees.
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
    // No packets actually go out; connect() just picks the route.
    sock.connect("8.8.8.8:80").ok()?;
    sock.local_addr().ok().map(|a| a.ip())
}

fn accept_loop(
    server: Server,
    shared: SharedFrame,
    settings: Arc<Mutex<Settings>>,
    info: Arc<RelayInfo>,
) {
    // Poll-based accept so we can honour the shutdown flag without forcing a
    // request to come through.
    while !info.shutdown.load(Ordering::Relaxed) {
        let request = match server.recv_timeout(Duration::from_millis(200)) {
            Ok(Some(r)) => r,
            Ok(None) => continue,
            Err(e) => {
                log::warn!("relay accept error: {e}");
                continue;
            }
        };
        let shared = shared.clone();
        let settings = settings.clone();
        let info = info.clone();
        let url = request.url().to_string();
        let method = request.method().clone();
        std::thread::Builder::new()
            .name(format!("relay-{}", request.remote_addr().map(|a| a.to_string()).unwrap_or_default()))
            .spawn(move || {
                if let Err(e) = handle(request, &url, &method, shared, settings, info) {
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
    shared: SharedFrame,
    settings: Arc<Mutex<Settings>>,
    info: Arc<RelayInfo>,
) -> Result<()> {
    if method != &Method::Get {
        let _ = request.respond(Response::from_string("method not allowed").with_status_code(405));
        return Ok(());
    }
    match url {
        "/" | "/index.html" => serve_index(request),
        "/stream" | "/stream.mjpg" => serve_mjpeg(request, shared, settings, info),
        "/snapshot.jpg" => serve_snapshot(request, shared, settings),
        _ => {
            let _ = request.respond(Response::from_string("not found").with_status_code(404));
            Ok(())
        }
    }
}

fn serve_index(request: tiny_http::Request) -> Result<()> {
    // Full-bleed viewer page with the live MJPEG stream + tiny help overlay.
    // Suitable for both a browser-on-second-PC and OBS Browser Source. OBS
    // ignores the help overlay because of pointer-events:none.
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

fn serve_snapshot(
    request: tiny_http::Request,
    shared: SharedFrame,
    settings: Arc<Mutex<Settings>>,
) -> Result<()> {
    let Some(frame) = shared.get() else {
        let _ = request.respond(Response::from_string("no frame yet").with_status_code(503));
        return Ok(());
    };
    let quality = settings.lock().jpeg_quality.clamp(1, 100);
    let rgb = frame_to_rgb(&frame.data, frame.width, frame.height);
    let jpeg = encode_jpeg(&rgb, frame.width, frame.height, quality)?;
    let header = Header::from_bytes(&b"Content-Type"[..], &b"image/jpeg"[..]).unwrap();
    request.respond(Response::from_data(jpeg).with_header(header))?;
    Ok(())
}

fn serve_mjpeg(
    request: tiny_http::Request,
    shared: SharedFrame,
    settings: Arc<Mutex<Settings>>,
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
    loop {
        if info.shutdown.load(Ordering::Relaxed) {
            return Ok(());
        }
        let frame = match shared.get() {
            Some(f) if f.seq != last_seq => f,
            _ => {
                std::thread::sleep(Duration::from_millis(5));
                continue;
            }
        };
        last_seq = frame.seq;
        let quality = settings.lock().jpeg_quality.clamp(1, 100);
        let rgb = frame_to_rgb(&frame.data, frame.width, frame.height);
        let jpeg = encode_jpeg(&rgb, frame.width, frame.height, quality)?;
        write!(
            writer,
            "--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
            jpeg.len()
        )?;
        writer.write_all(&jpeg)?;
        writer.write_all(b"\r\n")?;
        writer.flush()?;
    }
}

/// Decrements active_clients when a streaming connection drops, regardless
/// of how it exited (clean close, broken pipe, our own loop returning).
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

/// Get an RGB byte slice from either a Rgb or Nv12 frame. For Rgb we just
/// clone the Arc's bytes; for Nv12 we synthesise RGB via BT.709 limited-range
/// YUV conversion. The relay path is opt-in and cold for most users, so this
/// per-pixel CPU cost is acceptable here.
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
            let uv_col = col & !1; // even-aligned pair
            let y = y_plane[row * w + col] as f32;
            let u = uv_plane[uv_row * w + uv_col] as f32;
            let v = uv_plane[uv_row * w + uv_col + 1] as f32;
            // BT.709 limited range
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
