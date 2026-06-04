// Screenshot (PNG) and recording (MP4 via ffmpeg) for the current capture
// feed. The screenshot path is always available because we encode the PNG
// in-process; recording shells out to ffmpeg.exe so it depends on ffmpeg
// being on PATH (vicash does not bundle a codec).

use crate::frame::{Frame, FrameData};
use anyhow::{Context, Result, anyhow};
use directories::UserDirs;
use image::{ColorType, ImageEncoder};
use parking_lot::Mutex;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Arc;
use std::time::SystemTime;

/// Where vicash writes screenshots and recordings. Defaults to the user's
/// Pictures folder under a `vicash` subdir.
pub fn default_output_dir() -> PathBuf {
    UserDirs::new()
        .and_then(|d| d.picture_dir().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("vicash")
}

fn ensure_dir(dir: &Path) -> Result<()> {
    if !dir.exists() {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("could not create {}", dir.display()))?;
    }
    Ok(())
}

fn timestamp() -> String {
    use std::time::{Duration, UNIX_EPOCH};
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO);
    let secs = now.as_secs();
    // Local-time formatting without a heavy dep: use chrono's already-pulled
    // time math by computing yyyymmdd_hhmmss assuming UTC. Good enough for
    // unique filenames.
    let mut t = secs;
    let s = (t % 60) as u32;
    t /= 60;
    let m = (t % 60) as u32;
    t /= 60;
    let h = (t % 24) as u32;
    let days = t / 24;
    let (year, month, day) = days_to_ymd(days as i64);
    format!("{year:04}{month:02}{day:02}_{h:02}{m:02}{s:02}")
}

/// Civil-date conversion: days since 1970-01-01 -> (year, month, day).
fn days_to_ymd(days: i64) -> (i32, u32, u32) {
    // Howard Hinnant's algorithm.
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = if m <= 2 { y + 1 } else { y };
    (year as i32, m, d)
}

/// Save a single PNG of the supplied frame. Returns the full path written.
pub fn save_screenshot(frame: &Frame, dir: &Path) -> Result<PathBuf> {
    ensure_dir(dir)?;
    let path = dir.join(format!("vicash_{}.png", timestamp()));
    let rgb = frame_to_rgb(frame);
    let file = std::fs::File::create(&path)
        .with_context(|| format!("create {}", path.display()))?;
    let writer = std::io::BufWriter::new(file);
    let encoder = image::codecs::png::PngEncoder::new(writer);
    encoder
        .write_image(&rgb, frame.width, frame.height, ColorType::Rgb8.into())
        .context("PNG encode")?;
    Ok(path)
}

fn frame_to_rgb(frame: &Frame) -> Vec<u8> {
    match &frame.data {
        FrameData::Rgb(b) => b.as_ref().clone(),
        FrameData::Nv12(b) => nv12_to_rgb(b.as_ref(), frame.width, frame.height),
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

/// Status of the active recording session, exposed to the UI so the panel
/// can show file path, elapsed time and frame count without holding any
/// thread-bound resources.
pub struct RecorderStatus {
    pub path: PathBuf,
    pub started_at: std::time::Instant,
    pub frames_written: std::sync::atomic::AtomicU64,
}

/// Active recording: holds the ffmpeg subprocess and its stdin pipe. Dropping
/// the recorder closes stdin which lets ffmpeg flush and exit cleanly.
pub struct Recorder {
    child: Child,
    stdin: Option<ChildStdin>,
    pub status: Arc<RecorderStatus>,
    width: u32,
    height: u32,
}

impl Recorder {
    pub fn start(width: u32, height: u32, fps: u32, dir: &Path) -> Result<Self> {
        ensure_dir(dir)?;
        let path = dir.join(format!("vicash_{}.mp4", timestamp()));

        let mut cmd = Command::new("ffmpeg");
        cmd.args([
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "nv12",
            "-video_size", &format!("{width}x{height}"),
            "-framerate", &format!("{fps}"),
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            path.to_str().ok_or_else(|| anyhow!("output path not utf8"))?,
        ]);
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null());

        let mut child = cmd.spawn().with_context(|| {
            "could not spawn ffmpeg.exe (install it and put it on PATH to enable recording)"
        })?;
        let stdin = child.stdin.take();

        log::info!(
            "recording -> {} ({}x{} @ {} fps via ffmpeg libx264)",
            path.display(),
            width,
            height,
            fps
        );

        Ok(Self {
            child,
            stdin,
            status: Arc::new(RecorderStatus {
                path,
                started_at: std::time::Instant::now(),
                frames_written: std::sync::atomic::AtomicU64::new(0),
            }),
            width,
            height,
        })
    }

    /// Push one NV12 frame to ffmpeg. Returns false if the pipe broke,
    /// meaning the recorder should be torn down.
    pub fn push(&mut self, nv12: &[u8]) -> bool {
        let Some(stdin) = self.stdin.as_mut() else {
            return false;
        };
        let expected = (self.width as usize) * (self.height as usize) * 3 / 2;
        if nv12.len() < expected {
            return true;
        }
        if stdin.write_all(&nv12[..expected]).is_err() {
            log::warn!("recording pipe broken, ffmpeg may have crashed");
            self.stdin = None;
            return false;
        }
        self.status
            .frames_written
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        true
    }

    pub fn stop(mut self) -> PathBuf {
        // Drop stdin so ffmpeg sees EOF and writes the moov box, then wait.
        self.stdin = None;
        let _ = self.child.wait();
        let path = self.status.path.clone();
        log::info!("recording stopped: {}", path.display());
        path
    }
}

/// Convenience wrapper used by the preview's F10 handler to atomically
/// swap between recording / not recording without holding the lock across
/// long operations.
pub struct RecordingHandle {
    pub inner: Mutex<Option<Recorder>>,
    pub last_screenshot: Mutex<Option<PathBuf>>,
    pub last_recording: Mutex<Option<PathBuf>>,
}

impl RecordingHandle {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(None),
            last_screenshot: Mutex::new(None),
            last_recording: Mutex::new(None),
        })
    }

    pub fn is_recording(&self) -> bool {
        self.inner.lock().is_some()
    }

    pub fn elapsed_secs(&self) -> Option<u64> {
        self.inner
            .lock()
            .as_ref()
            .map(|r| r.status.started_at.elapsed().as_secs())
    }

    pub fn frames_written(&self) -> u64 {
        self.inner
            .lock()
            .as_ref()
            .map(|r| {
                r.status
                    .frames_written
                    .load(std::sync::atomic::Ordering::Relaxed)
            })
            .unwrap_or(0)
    }

    pub fn current_path(&self) -> Option<PathBuf> {
        self.inner.lock().as_ref().map(|r| r.status.path.clone())
    }

    pub fn push_frame(&self, nv12: &[u8]) {
        let mut guard = self.inner.lock();
        if let Some(rec) = guard.as_mut() {
            if !rec.push(nv12) {
                let stopped = guard.take().unwrap();
                let path = stopped.stop();
                *self.last_recording.lock() = Some(path);
            }
        }
    }
}
