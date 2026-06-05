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
    pub relay_port: u16,
    pub relay_autostart: bool,
    /// If true, the relay binds to 127.0.0.1 only - no other device on the
    /// LAN can reach it. Useful when on a public network where you do not
    /// want anyone sharing your WiFi to be able to pull your stream.
    pub relay_localhost_only: bool,
    /// Mix all audio input channels down to one and broadcast equally on
    /// every output channel. Fixes the MS2109/MS2130 left-only-channel quirk.
    pub audio_mix_to_mono: bool,
    /// Image colour adjustments applied in the shader before output. All
    /// stored as natural-scale floats so the same values can be re-saved as
    /// presets in toml. Brightness +/- 1.0 maps to a full luminance shift,
    /// contrast 1.0 is identity, saturation 1.0 is identity, hue degrees.
    pub color_brightness: f32,
    pub color_contrast: f32,
    pub color_saturation: f32,
    pub color_hue_deg: f32,
    /// Optional aspect ratio override in the form (width, height). When
    /// `None`, the capture's native aspect ratio is used (current behaviour).
    /// Common values: (4, 3) for retro consoles, (16, 9), (16, 10).
    pub custom_aspect: Option<(u32, u32)>,
    /// User zoom factor, applied around `pan_x/pan_y`. 1.0 = no zoom.
    /// Ctrl+wheel adjusts this in steps of 0.1; right-click resets to 1.0.
    pub zoom: f32,
    pub pan_x: f32,
    pub pan_y: f32,
    /// CRT/scanline shader strength. 0.0 = off, 1.0 = strong dark scanlines
    /// every other row. Useful for retro console (PS1, N64) capture.
    pub crt_strength: f32,
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
            // Immediate is the lowest-latency present mode: the finished frame
            // goes out without waiting for the display refresh cadence, which
            // is what a capture-card preview wants. May tear; switchable to
            // Mailbox/Fifo in the F1 panel. Falls back automatically if the
            // GPU/surface does not support it (see pick_present_mode).
            present_mode: PresentMode::Immediate,
            // 7777 is rarely on the Windows excluded port range; 8080 often
            // is because Hyper-V / WSL / Docker reserve dynamic ranges that
            // collide with it. Users with a free 8080 can still pick it.
            relay_port: 7777,
            relay_autostart: false,
            relay_localhost_only: false,
            audio_mix_to_mono: false,
            color_brightness: 0.0,
            color_contrast: 1.0,
            color_saturation: 1.0,
            color_hue_deg: 0.0,
            custom_aspect: None,
            zoom: 1.0,
            pan_x: 0.0,
            pan_y: 0.0,
            crt_strength: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CaptureInfo {
    pub fps_target: u32,
    pub format_label: String,
}
