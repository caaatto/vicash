use crate::i18n::Language;
use crate::settings::{FitMode, PresentMode, Settings};
use anyhow::Result;
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Persisted form of everything the F1 panel can change. Lives on disk as
/// `%APPDATA%/caaatto/vicash/config.toml`. The runtime form (`Settings`) is
/// derived from this plus the live audio runtime state on every load and is
/// snapshot-saved on changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub language: Language,
    pub display: DisplayConfig,
    pub monitor: MonitorConfig,
    pub relay: RelayConfig,
    pub capture: CaptureConfig,
    pub audio: AudioConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DisplayConfig {
    pub fit_mode: FitModeIo,
    pub show_stats: bool,
    pub background_color: [f32; 3],
    pub present_mode: PresentMode,
    pub color_brightness: f32,
    pub color_contrast: f32,
    pub color_saturation: f32,
    pub color_hue_deg: f32,
    pub crt_strength: f32,
    pub custom_aspect_w: u32,
    pub custom_aspect_h: u32,
    pub use_custom_aspect: bool,
    pub zoom: f32,
    pub pan_x: f32,
    pub pan_y: f32,
    /// Named colour presets. Each preset stores brightness/contrast/saturation/hue.
    /// Keys are user-chosen labels such as "Switch", "PS2 (warmer)", "PS1 CRT".
    #[serde(default)]
    pub color_presets: Vec<ColorPreset>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ColorPreset {
    pub name: String,
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub hue_deg: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MonitorConfig {
    pub fullscreen: bool,
    pub borderless: bool,
    pub always_on_top: bool,
    pub hide_cursor: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RelayConfig {
    pub jpeg_quality: u8,
    pub port: u16,
    pub autostart: bool,
    pub localhost_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CaptureConfig {
    /// The device's human-readable name. Preferred over `device_index` on
    /// reload because Windows can shuffle device indices after a reboot
    /// or driver update.
    pub device_name: Option<String>,
    pub device_index: Option<u32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub fps: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    pub enabled: bool,
    pub input_device: Option<String>,
    pub output_device: Option<String>,
    pub volume_percent: u32,
    pub muted: bool,
    pub delay_ms: u32,
    pub mix_to_mono: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FitModeIo {
    Stretch,
    Fit,
    Fill,
}

impl From<FitMode> for FitModeIo {
    fn from(m: FitMode) -> Self {
        match m {
            FitMode::Stretch => FitModeIo::Stretch,
            FitMode::Fit => FitModeIo::Fit,
            FitMode::Fill => FitModeIo::Fill,
        }
    }
}

impl From<FitModeIo> for FitMode {
    fn from(m: FitModeIo) -> Self {
        match m {
            FitModeIo::Stretch => FitMode::Stretch,
            FitModeIo::Fit => FitMode::Fit,
            FitModeIo::Fill => FitMode::Fill,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        let s = Settings::default();
        Self {
            language: s.language,
            display: DisplayConfig::default(),
            monitor: MonitorConfig {
                fullscreen: s.fullscreen,
                borderless: s.borderless,
                always_on_top: s.always_on_top,
                hide_cursor: s.hide_cursor,
            },
            relay: RelayConfig {
                jpeg_quality: s.jpeg_quality,
                port: s.relay_port,
                autostart: s.relay_autostart,
                localhost_only: s.relay_localhost_only,
            },
            capture: CaptureConfig::default(),
            audio: AudioConfig::default(),
        }
    }
}

impl Default for DisplayConfig {
    fn default() -> Self {
        let s = Settings::default();
        Self {
            fit_mode: s.fit_mode.into(),
            show_stats: s.show_stats,
            background_color: s.background_color,
            present_mode: s.present_mode,
            color_brightness: s.color_brightness,
            color_contrast: s.color_contrast,
            color_saturation: s.color_saturation,
            color_hue_deg: s.color_hue_deg,
            crt_strength: s.crt_strength,
            custom_aspect_w: 4,
            custom_aspect_h: 3,
            use_custom_aspect: false,
            zoom: s.zoom,
            pan_x: s.pan_x,
            pan_y: s.pan_y,
            color_presets: builtin_color_presets(),
        }
    }
}

/// Ship a couple of named presets out of the box so the dropdown is not
/// empty on first run. Users can rename, delete and add their own.
fn builtin_color_presets() -> Vec<ColorPreset> {
    vec![
        ColorPreset { name: "Neutral".into(), brightness: 0.0, contrast: 1.0, saturation: 1.0, hue_deg: 0.0 },
        ColorPreset { name: "Switch (punchier)".into(), brightness: 0.02, contrast: 1.08, saturation: 1.15, hue_deg: 0.0 },
        ColorPreset { name: "Retro warm".into(), brightness: 0.0, contrast: 1.05, saturation: 0.95, hue_deg: 8.0 },
        ColorPreset { name: "Cool".into(), brightness: 0.0, contrast: 1.0, saturation: 1.0, hue_deg: -8.0 },
    ]
}

impl Default for MonitorConfig {
    fn default() -> Self {
        let s = Settings::default();
        Self {
            fullscreen: s.fullscreen,
            borderless: s.borderless,
            always_on_top: s.always_on_top,
            hide_cursor: s.hide_cursor,
        }
    }
}

impl Default for RelayConfig {
    fn default() -> Self {
        Self { jpeg_quality: 75, port: 7777, autostart: false, localhost_only: false }
    }
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            device_name: None,
            device_index: None,
            width: None,
            height: None,
            fps: None,
        }
    }
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            // Audio passthrough on by default so a fresh install hears the
            // capture card without having to pass --audio. Overridable via
            // config (enabled = false) for users who only want video.
            enabled: true,
            input_device: None,
            output_device: None,
            volume_percent: 100,
            muted: false,
            delay_ms: 100,
            mix_to_mono: false,
        }
    }
}

pub fn config_path() -> Option<PathBuf> {
    ProjectDirs::from("com", "caaatto", "vicash").map(|d| d.config_dir().join("config.toml"))
}

pub fn load() -> Config {
    let Some(path) = config_path() else {
        log::warn!("could not resolve config dir, using defaults");
        return Config::default();
    };
    if !path.exists() {
        log::info!("no config at {}, using defaults", path.display());
        return Config::default();
    }
    match std::fs::read_to_string(&path) {
        Ok(text) => match toml::from_str::<Config>(&text) {
            Ok(c) => {
                log::info!("loaded config from {}", path.display());
                c
            }
            Err(e) => {
                log::error!("config parse failed ({e}), using defaults");
                Config::default()
            }
        },
        Err(e) => {
            log::error!("config read failed ({e}), using defaults");
            Config::default()
        }
    }
}

pub fn save(cfg: &Config) -> Result<()> {
    let Some(path) = config_path() else {
        anyhow::bail!("no config dir resolvable");
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let text = toml::to_string_pretty(cfg)?;
    std::fs::write(&path, text)?;
    Ok(())
}

/// Build a runtime `Settings` from a persisted `Config`. The `show_panel`
/// field is always reset to false so the user sees video on launch.
pub fn settings_from_config(cfg: &Config) -> Settings {
    Settings {
        show_panel: false,
        show_stats: cfg.display.show_stats,
        fit_mode: cfg.display.fit_mode.into(),
        background_color: cfg.display.background_color,
        jpeg_quality: cfg.relay.jpeg_quality,
        fullscreen: cfg.monitor.fullscreen,
        borderless: cfg.monitor.borderless,
        always_on_top: cfg.monitor.always_on_top,
        hide_cursor: cfg.monitor.hide_cursor,
        language: cfg.language,
        present_mode: cfg.display.present_mode,
        relay_port: cfg.relay.port,
        relay_autostart: cfg.relay.autostart,
        relay_localhost_only: cfg.relay.localhost_only,
        audio_mix_to_mono: cfg.audio.mix_to_mono,
        color_brightness: cfg.display.color_brightness,
        color_contrast: cfg.display.color_contrast,
        color_saturation: cfg.display.color_saturation,
        color_hue_deg: cfg.display.color_hue_deg,
        custom_aspect: if cfg.display.use_custom_aspect {
            Some((cfg.display.custom_aspect_w, cfg.display.custom_aspect_h))
        } else {
            None
        },
        zoom: cfg.display.zoom,
        pan_x: cfg.display.pan_x,
        pan_y: cfg.display.pan_y,
        crt_strength: cfg.display.crt_strength,
    }
}

/// Snapshot the runtime state back into a `Config` shape for saving.
/// Existing colour presets are preserved through `prior_presets` because
/// they live only in the persisted config, not in the live `Settings`.
pub fn config_from_runtime(
    s: &Settings,
    capture: &CaptureConfig,
    audio: &AudioConfig,
    prior_presets: Vec<ColorPreset>,
) -> Config {
    let (use_aspect, aw, ah) = match s.custom_aspect {
        Some((w, h)) => (true, w, h),
        None => (false, 4, 3),
    };
    Config {
        language: s.language,
        display: DisplayConfig {
            fit_mode: s.fit_mode.into(),
            show_stats: s.show_stats,
            background_color: s.background_color,
            present_mode: s.present_mode,
            color_brightness: s.color_brightness,
            color_contrast: s.color_contrast,
            color_saturation: s.color_saturation,
            color_hue_deg: s.color_hue_deg,
            crt_strength: s.crt_strength,
            custom_aspect_w: aw,
            custom_aspect_h: ah,
            use_custom_aspect: use_aspect,
            zoom: s.zoom,
            pan_x: s.pan_x,
            pan_y: s.pan_y,
            color_presets: prior_presets,
        },
        monitor: MonitorConfig {
            fullscreen: s.fullscreen,
            borderless: s.borderless,
            always_on_top: s.always_on_top,
            hide_cursor: s.hide_cursor,
        },
        relay: RelayConfig {
            jpeg_quality: s.jpeg_quality,
            port: s.relay_port,
            autostart: s.relay_autostart,
            localhost_only: s.relay_localhost_only,
        },
        capture: capture.clone(),
        audio: audio.clone(),
    }
}
