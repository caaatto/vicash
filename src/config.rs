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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CaptureConfig {
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
            display: DisplayConfig {
                fit_mode: s.fit_mode.into(),
                show_stats: s.show_stats,
                background_color: s.background_color,
                present_mode: s.present_mode,
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
        }
    }
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
        Self { jpeg_quality: 75, port: 7777, autostart: false }
    }
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
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
            enabled: false,
            input_device: None,
            output_device: None,
            volume_percent: 100,
            muted: false,
            delay_ms: 100,
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
    }
}

/// Snapshot the runtime state back into a `Config` shape for saving.
pub fn config_from_runtime(
    s: &Settings,
    capture: &CaptureConfig,
    audio: &AudioConfig,
) -> Config {
    Config {
        language: s.language,
        display: DisplayConfig {
            fit_mode: s.fit_mode.into(),
            show_stats: s.show_stats,
            background_color: s.background_color,
            present_mode: s.present_mode,
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
        },
        capture: capture.clone(),
        audio: audio.clone(),
    }
}
