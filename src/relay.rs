use crate::frame::{FrameData, SharedFrame};
use crate::settings::Settings;
use anyhow::{Context, Result};
use image::codecs::jpeg::JpegEncoder;
use image::{ColorType, ImageEncoder};
use parking_lot::Mutex;
use std::io::Write;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tiny_http::{Header, Method, Response, Server};

const BOUNDARY: &str = "vcshareframe";

pub fn spawn(
    addr: SocketAddr,
    shared: SharedFrame,
    settings: Arc<Mutex<Settings>>,
) -> Result<()> {
    let server = Server::http(addr)
        .map_err(|e| anyhow::anyhow!("failed to bind {addr}: {e}"))?;
    std::thread::Builder::new()
        .name("relay-accept".into())
        .spawn(move || accept_loop(server, shared, settings))
        .context("failed to spawn relay accept thread")?;
    Ok(())
}

fn accept_loop(server: Server, shared: SharedFrame, settings: Arc<Mutex<Settings>>) {
    for request in server.incoming_requests() {
        let shared = shared.clone();
        let settings = settings.clone();
        let url = request.url().to_string();
        let method = request.method().clone();
        std::thread::Builder::new()
            .name(format!("relay-{}", request.remote_addr().map(|a| a.to_string()).unwrap_or_default()))
            .spawn(move || {
                if let Err(e) = handle(request, &url, &method, shared, settings) {
                    log::debug!("client gone: {e}");
                }
            })
            .ok();
    }
}

fn handle(
    request: tiny_http::Request,
    url: &str,
    method: &Method,
    shared: SharedFrame,
    settings: Arc<Mutex<Settings>>,
) -> Result<()> {
    if method != &Method::Get {
        let _ = request.respond(Response::from_string("method not allowed").with_status_code(405));
        return Ok(());
    }
    match url {
        "/" | "/index.html" => serve_index(request),
        "/stream" | "/stream.mjpg" => serve_mjpeg(request, shared, settings),
        "/snapshot.jpg" => serve_snapshot(request, shared, settings),
        _ => {
            let _ = request.respond(Response::from_string("not found").with_status_code(404));
            Ok(())
        }
    }
}

fn serve_index(request: tiny_http::Request) -> Result<()> {
    let html = r#"<!doctype html>
<html><head><meta charset="utf-8"><title>video capture share</title>
<style>
  html,body{margin:0;background:#000;height:100%}
  img{display:block;width:100%;height:100%;object-fit:contain}
</style></head>
<body><img src="/stream" alt="capture"></body></html>"#;
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
) -> Result<()> {
    let mut writer = request.into_writer();
    write!(writer, "HTTP/1.1 200 OK\r\n")?;
    write!(writer, "Content-Type: multipart/x-mixed-replace; boundary={BOUNDARY}\r\n")?;
    write!(writer, "Cache-Control: no-store, no-cache, must-revalidate, max-age=0\r\n")?;
    write!(writer, "Pragma: no-cache\r\n")?;
    write!(writer, "Connection: close\r\n")?;
    write!(writer, "\r\n")?;
    writer.flush()?;

    let mut last_seq: u64 = 0;
    loop {
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
