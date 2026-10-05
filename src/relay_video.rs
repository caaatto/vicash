//! Scaled video variants for the /live player's quality menu (1440p, 1080p,
//! 720p, ...). Each resolution somebody is watching gets its own encoder
//! thread that downscales straight from the capture frame and JPEG encodes
//! the result; viewers of the same resolution share it. A variant whose
//! last viewer left shuts itself down after a short grace period, so an idle
//! relay costs nothing beyond the full-resolution encoder it always had.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use crate::audio_relay;
use crate::frame::{FrameData, SharedFrame};
use crate::relay::{LatestJpeg, SharedJpeg, encode_jpeg};
use crate::settings::Settings;

/// How long a variant without viewers keeps running. Covers a quality
/// switch back and forth or a page reload without restarting the thread.
const IDLE_GRACE: Duration = Duration::from_secs(5);

pub struct VideoVariants {
    shared: SharedFrame,
    settings: Arc<Mutex<Settings>>,
    shutdown: Arc<AtomicBool>,
    variants: Mutex<HashMap<u32, Arc<Variant>>>,
    /// Capture rate as seen by the full-resolution encoder, for /live/info.
    pub source_fps: AtomicU32,
}

pub struct Variant {
    pub jpeg: Arc<SharedJpeg>,
    /// Viewers wanting every frame vs. viewers capped at 30 fps. While only
    /// the latter exist the encoder halves its own rate as well.
    full_rate: AtomicUsize,
    half_rate: AtomicUsize,
}

impl Variant {
    fn viewers(&self) -> usize {
        self.full_rate.load(Ordering::Relaxed) + self.half_rate.load(Ordering::Relaxed)
    }
}

/// Keeps a variant alive while a client streams from it.
pub struct VariantLease {
    pub variant: Arc<Variant>,
    full_rate: bool,
}

impl Drop for VariantLease {
    fn drop(&mut self) {
        let counter = if self.full_rate { &self.variant.full_rate } else { &self.variant.half_rate };
        counter.fetch_sub(1, Ordering::Relaxed);
    }
}

impl VideoVariants {
    pub fn new(shared: SharedFrame, settings: Arc<Mutex<Settings>>, shutdown: Arc<AtomicBool>) -> Arc<Self> {
        Arc::new(Self {
            shared,
            settings,
            shutdown,
            variants: Mutex::new(HashMap::new()),
            source_fps: AtomicU32::new(0),
        })
    }

    /// Current capture size, if a frame has been seen yet.
    pub fn source_size(&self) -> Option<(u32, u32)> {
        self.shared.get().map(|f| (f.width, f.height))
    }

    /// Lease the variant for `height` lines, starting its encoder if needed.
    /// `fps` at or below 30 counts as a half-rate viewer.
    pub fn lease(self: &Arc<Self>, height: u32, fps: Option<f64>) -> VariantLease {
        let full_rate = fps.map_or(true, |f| f > 30.5);
        let mut map = self.variants.lock();
        let variant = map
            .entry(height)
            .or_insert_with(|| {
                let v = Arc::new(Variant {
                    jpeg: Arc::new(SharedJpeg::new()),
                    full_rate: AtomicUsize::new(0),
                    half_rate: AtomicUsize::new(0),
                });
                let this = self.clone();
                let worker = v.clone();
                let spawned = std::thread::Builder::new()
                    .name(format!("relay-encoder-{height}p"))
                    .spawn(move || this.variant_loop(height, worker));
                if let Err(e) = spawned {
                    log::warn!("relay: could not start {height}p encoder: {e}");
                }
                v
            })
            .clone();
        let counter = if full_rate { &variant.full_rate } else { &variant.half_rate };
        counter.fetch_add(1, Ordering::Relaxed);
        VariantLease { variant, full_rate }
    }

    fn variant_loop(self: Arc<Self>, height: u32, variant: Arc<Variant>) {
        log::info!("relay: {height}p encoder started");
        let mut last_seq = 0u64;
        let mut half_rate = Pacer::new(30.0);
        let mut idle_since: Option<Instant> = None;
        while !self.shutdown.load(Ordering::Relaxed) {
            if variant.viewers() == 0 {
                let since = *idle_since.get_or_insert_with(Instant::now);
                if since.elapsed() >= IDLE_GRACE {
                    // Re-check under the map lock so a viewer leasing right
                    // now either sees us gone or keeps us alive.
                    let mut map = self.variants.lock();
                    if variant.viewers() == 0 {
                        map.remove(&height);
                        break;
                    }
                }
                std::thread::sleep(Duration::from_millis(50));
                continue;
            }
            idle_since = None;

            let Some(frame) = self.shared.get() else {
                std::thread::sleep(Duration::from_millis(5));
                continue;
            };
            if frame.seq == last_seq {
                std::thread::sleep(Duration::from_millis(2));
                continue;
            }
            last_seq = frame.seq;
            if variant.full_rate.load(Ordering::Relaxed) == 0 && !half_rate.admit() {
                continue;
            }

            let (out_w, out_h) = scaled_size(frame.width, frame.height, height);
            let rgb = scale_to_rgb(&frame.data, frame.width, frame.height, out_w, out_h);
            let quality = self.settings.lock().jpeg_quality.clamp(1, 100);
            match encode_jpeg(&rgb, out_w, out_h, quality) {
                Ok(bytes) => variant.jpeg.publish(LatestJpeg {
                    bytes: Arc::new(bytes),
                    seq: frame.seq,
                    pts_us: audio_relay::pts_us(frame.captured_at),
                }),
                Err(e) => {
                    log::warn!("relay {height}p encoder: {e}");
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
        }
        log::info!("relay: {height}p encoder stopped");
    }
}

/// Thins a frame stream down to a target rate by keeping every n-th frame,
/// with n derived from the measured input rate (60 -> 30 keeps every
/// second, 50 -> 25 likewise). Timing grids would add judder on top of the
/// capture jitter; whole-frame decimation passes the source cadence through
/// untouched. An input already at or below the target is not thinned.
pub(crate) struct Pacer {
    period: f64,
    last_arrival: Option<Instant>,
    interval: f64,
    count: u32,
}

impl Pacer {
    pub fn new(fps: f64) -> Self {
        Self { period: 1.0 / fps, last_arrival: None, interval: 0.0, count: 0 }
    }

    /// Whether the frame arriving now should be taken.
    pub fn admit(&mut self) -> bool {
        let now = Instant::now();
        if let Some(prev) = self.last_arrival.replace(now) {
            let dt = (now - prev).as_secs_f64();
            if dt < 0.5 {
                self.interval = if self.interval == 0.0 { dt } else { self.interval * 0.9 + dt * 0.1 };
            }
        }
        if self.interval == 0.0 {
            return true;
        }
        let every = (self.period / self.interval).round().max(1.0) as u32;
        self.count += 1;
        if self.count >= every {
            self.count = 0;
            true
        } else {
            false
        }
    }
}

/// Output size for `height` lines keeping the source aspect, both even.
fn scaled_size(src_w: u32, src_h: u32, height: u32) -> (u32, u32) {
    let h = height.min(src_h).max(2) & !1;
    let w = ((src_w as u64 * h as u64 / src_h.max(1) as u64) as u32).max(2) & !1;
    (w, h)
}

/// Downscale and convert to RGB8 in one pass. Each output pixel averages an
/// n x n grid of source samples across its footprint (n grows with the
/// scale factor, capped at 4), which keeps 1080p -> 360p from shimmering
/// the way plain nearest or bilinear sampling would.
fn scale_to_rgb(data: &FrameData, src_w: u32, src_h: u32, out_w: u32, out_h: u32) -> Vec<u8> {
    let (sw, sh, ow, oh) = (src_w as usize, src_h as usize, out_w as usize, out_h as usize);
    let mut rgb = vec![0u8; ow * oh * 3];
    let n = (sw / ow.max(1)).clamp(1, 4);
    // Source offsets of the n sample columns/rows inside each footprint,
    // in 16.16 fixed point relative to the footprint origin.
    let step_x = (sw << 16) / ow.max(1);
    let step_y = (sh << 16) / oh.max(1);
    let sub_x: Vec<usize> = (0..n).map(|i| (step_x * (2 * i + 1)) / (2 * n)).collect();
    let sub_y: Vec<usize> = (0..n).map(|i| (step_y * (2 * i + 1)) / (2 * n)).collect();
    let samples = (n * n) as i32;

    match data {
        FrameData::Nv12(buf) => {
            if buf.len() < sw * sh * 3 / 2 {
                return rgb;
            }
            let (y_plane, uv_plane) = buf.split_at(sw * sh);
            for oy in 0..oh {
                let base_y = oy * step_y;
                for ox in 0..ow {
                    let base_x = ox * step_x;
                    let mut y_sum = 0i32;
                    for dy in &sub_y {
                        let sy = ((base_y + dy) >> 16).min(sh - 1);
                        let row = &y_plane[sy * sw..sy * sw + sw];
                        for dx in &sub_x {
                            y_sum += row[((base_x + dx) >> 16).min(sw - 1)] as i32;
                        }
                    }
                    // Chroma is already quarter resolution; one centre tap.
                    let cx = ((base_x + step_x / 2) >> 16).min(sw - 1) & !1;
                    let cy = ((base_y + step_y / 2) >> 16).min(sh - 1) / 2;
                    let u = uv_plane[cy * sw + cx] as i32 - 128;
                    let v = uv_plane[cy * sw + cx + 1] as i32 - 128;
                    let (r, g, b) = yuv_to_rgb(y_sum / samples, u, v);
                    let o = (oy * ow + ox) * 3;
                    rgb[o] = r;
                    rgb[o + 1] = g;
                    rgb[o + 2] = b;
                }
            }
        }
        FrameData::Rgb(buf) => {
            if buf.len() < sw * sh * 3 {
                return rgb;
            }
            for oy in 0..oh {
                let base_y = oy * step_y;
                for ox in 0..ow {
                    let base_x = ox * step_x;
                    let mut sum = [0i32; 3];
                    for dy in &sub_y {
                        let sy = ((base_y + dy) >> 16).min(sh - 1);
                        for dx in &sub_x {
                            let sx = ((base_x + dx) >> 16).min(sw - 1);
                            let i = (sy * sw + sx) * 3;
                            sum[0] += buf[i] as i32;
                            sum[1] += buf[i + 1] as i32;
                            sum[2] += buf[i + 2] as i32;
                        }
                    }
                    let o = (oy * ow + ox) * 3;
                    rgb[o] = (sum[0] / samples) as u8;
                    rgb[o + 1] = (sum[1] / samples) as u8;
                    rgb[o + 2] = (sum[2] / samples) as u8;
                }
            }
        }
    }
    rgb
}

/// BT.709 limited range to RGB, integer version of relay::nv12_to_rgb.
fn yuv_to_rgb(y: i32, u: i32, v: i32) -> (u8, u8, u8) {
    let c = (y - 16) * 298;
    let r = (c + 459 * v + 128) >> 8;
    let g = (c - 55 * u - 136 * v + 128) >> 8;
    let b = (c + 541 * u + 128) >> 8;
    (r.clamp(0, 255) as u8, g.clamp(0, 255) as u8, b.clamp(0, 255) as u8)
}
