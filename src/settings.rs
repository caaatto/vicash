/// Runtime knobs shown in the F1 settings overlay.
#[derive(Debug, Clone)]
pub struct Settings {
    pub show_panel: bool,
    pub show_stats: bool,
    pub fit_mode: FitMode,
    pub background_color: [f32; 3],
    pub jpeg_quality: u8,
    /// Monitor-mode toggles. Together they turn the preview window into a
    /// proper second-screen replacement for the console.
    pub fullscreen: bool,
    pub borderless: bool,
    pub always_on_top: bool,
    pub hide_cursor: bool,
    pub language: crate::i18n::Language,
    pub present_mode: PresentMode,
}

/// Mirror of wgpu::PresentMode so the settings layer does not depend on wgpu
/// types. The preview translates this into the wgpu enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PresentMode {
    /// Lowest latency, no vsync, may tear.
    Immediate,
    /// Low latency, no tearing, requires HW support.
    Mailbox,
    /// Standard vsync, no tearing, slightly higher latency.
    Fifo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitMode {
    /// Stretch to fill the window, ignoring aspect ratio.
    Stretch,
    /// Preserve aspect ratio, letterbox the rest with the background color.
    Fit,
    /// Preserve aspect ratio, fill the window, crop the overflow.
    Fill,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            show_panel: false,
            show_stats: true,
            fit_mode: FitMode::Fit,
            background_color: [0.0, 0.0, 0.0],
            jpeg_quality: 75,
            fullscreen: false,
            borderless: false,
            always_on_top: false,
            hide_cursor: true,
            language: crate::i18n::Language::default(),
            present_mode: PresentMode::Mailbox,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CaptureInfo {
    pub fps_target: u32,
    pub format_label: String,
}
