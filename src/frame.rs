use parking_lot::Mutex;
use std::sync::Arc;

/// One frame from the capture device. The data payload is one of:
///   - Rgb: tightly packed RGB8, length = width * height * 3
///   - Nv12: Y plane followed by interleaved UV plane, length = width * height * 3 / 2
#[derive(Clone)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub data: FrameData,
    /// Monotonic counter, increments each time the capture thread publishes.
    pub seq: u64,
}

#[derive(Clone)]
pub enum FrameData {
    Rgb(Arc<Vec<u8>>),
    Nv12(Arc<Vec<u8>>),
}

/// Event sent from background threads to wake the winit event loop. Currently
/// just a "new frame ready" signal; keep it an enum so we can add more (settings
/// reload, audio device change, etc.) without changing every type signature.
#[derive(Debug, Clone, Copy)]
pub enum UiEvent {
    FrameReady,
}

/// Latest-frame slot, single producer many consumer. The capture thread overwrites;
/// readers always see the newest frame, never block the producer, and never queue up.
#[derive(Clone)]
pub struct SharedFrame {
    inner: Arc<Mutex<Option<Frame>>>,
}

impl SharedFrame {
    pub fn new() -> Self {
        Self { inner: Arc::new(Mutex::new(None)) }
    }

    pub fn publish(&self, frame: Frame) {
        *self.inner.lock() = Some(frame);
    }

    pub fn get(&self) -> Option<Frame> {
        self.inner.lock().clone()
    }
}
