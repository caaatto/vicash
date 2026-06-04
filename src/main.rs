// Suppress the cmd-style console window on release builds. Debug builds
// keep the console so `cargo run` still prints log output during dev. When
// the release binary is invoked from a terminal (e.g. `vicash --list` in
// PowerShell), we re-attach to the parent console at startup so CLI output
// still ends up where the user expects it.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use anyhow::{Context, Result};
use clap::Parser;
use parking_lot::Mutex;
use std::net::SocketAddr;
use std::sync::Arc;

mod audio;
mod capture;
mod config;
mod fmp4_relay;
mod frame;
#[cfg(windows)]
mod h264_encoder;
mod i18n;
mod mpegts;
mod perf;
mod preview;
mod record;
mod relay;
mod settings;
mod threadprio;
mod updater;
#[cfg(windows)]
mod video_stream;

#[derive(Parser, Debug)]
#[command(
    name = "vicash",
    version,
    about = "vicash: low overhead capture card preview and LAN relay",
    long_about = None,
)]
struct Cli {
    /// Index of the capture device to open. Omit to pick interactively.
    #[arg(short, long)]
    device: Option<u32>,

    /// Print the list of devices and exit. Useful for scripts.
    #[arg(long)]
    list: bool,

    /// Open the device, print every supported mode it reports, then exit.
    #[arg(long)]
    probe: bool,

    /// Accept MJPEG and other low fps modes if nothing better fits. By default
    /// modes below 5 fps are rejected because cheap cards advertise them and
    /// they are useless.
    #[arg(long)]
    allow_mjpeg: bool,

    /// Requested width. The device picks the closest supported mode.
    #[arg(long)]
    width: Option<u32>,

    /// Requested height.
    #[arg(long)]
    height: Option<u32>,

    /// Requested frames per second.
    #[arg(long)]
    fps: Option<u32>,

    /// Bind address for the MJPEG HTTP relay, e.g. 0.0.0.0:8080. Omit to skip.
    #[arg(long)]
    serve: Option<SocketAddr>,

    /// JPEG quality for the relay, 1 to 100. Live-adjustable from the F1 panel.
    #[arg(long, default_value_t = 75)]
    quality: u8,

    /// Run without opening a preview window. Useful when only relaying.
    #[arg(long)]
    headless: bool,

    /// Pass audio from the capture card through to the system default output.
    #[arg(long)]
    audio: bool,

    /// Substring matched against audio input device names (case insensitive).
    /// If unset, the audio input that matches the chosen video device name is
    /// preferred; failing that, the system default input is used.
    #[arg(long)]
    audio_device: Option<String>,

    /// Initial audio sync delay in milliseconds. The capture card adds latency
    /// to the video; audio typically arrives faster. Increase this until the
    /// audio matches the picture. Live-adjustable from the F1 panel.
    #[arg(long, default_value_t = 100)]
    audio_delay_ms: u32,

    /// Print audio input devices and exit.
    #[arg(long)]
    list_audio: bool,
}

fn main() -> Result<()> {
    // With windows_subsystem = "windows" there is no console by default.
    // If the user actually launched us from a terminal we re-attach to its
    // console so `vicash --list` and similar CLI flags still show output.
    // Silently no-ops when there is no parent console (e.g. double-click).
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::Console::{ATTACH_PARENT_PROCESS, AttachConsole};
        AttachConsole(ATTACH_PARENT_PROCESS);
    }

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Main thread runs the render / event loop, so raise its priority before
    // anything else schedules.
    threadprio::bump_render_thread();

    // Delete any vicash.exe.old left behind by the previous self-update.
    // Safe no-op on first run.
    updater::cleanup_previous();

    let cli = Cli::parse();

    if cli.list {
        let devices = capture::enumerate()?;
        capture::print_devices(&devices);
        return Ok(());
    }

    if cli.list_audio {
        let names = audio::list_input_devices();
        if names.is_empty() {
            println!("No audio input devices found.");
        } else {
            println!("Audio input devices:");
            for n in names {
                println!("  - {n}");
            }
        }
        return Ok(());
    }

    // Load persisted config first; CLI flags win over it on conflicts.
    let cfg = config::load();

    // Resolve which video device to open. CLI --device wins; otherwise the
    // persisted device name (more stable than the index across reboots);
    // otherwise the persisted index as a last fallback; otherwise prompt.
    let device_index = if let Some(idx) = cli.device {
        idx
    } else if let Some(name) = cfg.capture.device_name.as_deref() {
        match resolve_device_by_name(name) {
            Some(idx) => {
                log::info!("matched persisted device name '{name}' to index {idx}");
                idx
            }
            None => {
                log::warn!(
                    "persisted device '{name}' not found, falling back to index {}",
                    cfg.capture.device_index.unwrap_or(0)
                );
                cfg.capture
                    .device_index
                    .unwrap_or_else(|| capture::pick_device_interactive().unwrap_or(0))
            }
        }
    } else if let Some(idx) = cfg.capture.device_index {
        idx
    } else {
        capture::pick_device_interactive()?
    };

    if cli.probe {
        return capture::probe(device_index);
    }

    let video_device_name = capture::enumerate()
        .ok()
        .and_then(|devs| {
            devs.into_iter()
                .find(|d| capture::index_of(d) == device_index)
                .map(|d| d.human_name())
        });

    let request = capture::CaptureRequest {
        device_index,
        width: cli.width.or(cfg.capture.width),
        height: cli.height.or(cfg.capture.height),
        fps: cli.fps.or(cfg.capture.fps),
        force_mjpeg: cli.allow_mjpeg,
    };

    let mut initial_settings = config::settings_from_config(&cfg);
    // CLI quality overrides config; if user didn't pass --quality the clap
    // default of 75 wins which matches the config default anyway.
    initial_settings.jpeg_quality = cli.quality;
    let shared_settings = Arc::new(Mutex::new(initial_settings));

    let target_fps = cli.fps.or(cfg.capture.fps).unwrap_or(60);
    let capture_info = settings::CaptureInfo {
        fps_target: target_fps,
        format_label: if cli.allow_mjpeg { "any".into() } else { "raw preferred".into() },
    };

    let shared = frame::SharedFrame::new();

    // Build the event loop up front so we can pass a proxy to the capture
    // thread. The capture thread wakes the loop on each new frame, which is
    // what keeps the GPU and CPU idle when nothing is changing.
    let (event_loop, notifier) = if cli.headless {
        (None, None)
    } else {
        let el = preview::build_event_loop()?;
        let proxy = el.create_proxy();
        (Some(el), Some(proxy))
    };

    let metrics = perf::PerfMetrics::new();
    perf::spawn_sampler(metrics.clone());

    let capture_ctrl = Arc::new(
        capture::spawn(request, shared.clone(), notifier, metrics.clone())
            .context("failed to start capture thread")?
    );

    let audio_enabled = cli.audio || cfg.audio.enabled;
    let audio_delay = if cli.audio_delay_ms != 100 {
        cli.audio_delay_ms
    } else {
        cfg.audio.delay_ms
    };
    let audio_runtime = if audio_enabled {
        let hint = cli
            .audio_device
            .as_deref()
            .or(cfg.audio.input_device.as_deref())
            .or(video_device_name.as_deref());
        match audio::start(hint, audio_delay) {
            Ok(rt) => {
                // Apply persisted audio prefs.
                rt.state.set_volume(cfg.audio.volume_percent);
                rt.state.set_muted(cfg.audio.muted);
                rt.state.set_mix_to_mono(cfg.audio.mix_to_mono);
                if let Some(out) = cfg.audio.output_device.as_deref() {
                    if !out.is_empty() && out != rt.state.output_name() {
                        if let Err(e) = rt.set_output(out) {
                            log::warn!("could not restore output device '{out}': {e:#}");
                        }
                    }
                }
                Some(Arc::new(rt))
            }
            Err(e) => {
                log::error!("audio passthrough disabled: {e:#}");
                None
            }
        }
    } else {
        None
    };

    // Live-toggleable relay handle. Either CLI --serve or the persisted
    // autostart flag spins it up at launch; the F1 panel can start and stop
    // it later without touching the rest of the process.
    let relay_slot: Arc<Mutex<Option<Arc<relay::RelayInfo>>>> = Arc::new(Mutex::new(None));
    let autostart_port = if let Some(addr) = cli.serve {
        Some(addr)
    } else if cfg.relay.autostart {
        let host = if cfg.relay.localhost_only {
            [127, 0, 0, 1]
        } else {
            [0, 0, 0, 0]
        };
        Some(SocketAddr::from((host, cfg.relay.port)))
    } else {
        None
    };
    if let Some(addr) = autostart_port {
        match relay::spawn(addr, shared.clone(), shared_settings.clone()) {
            Ok(info) => {
                log::info!("MJPEG relay live at {}/", info.lan_url);
                *relay_slot.lock() = Some(info);
            }
            Err(e) => log::error!("relay autostart failed: {e:#}"),
        }
    }

    // Background saver: snapshots the live state into a Config every second
    // and writes to disk when something changed. Keeps the UI thread free of
    // any TOML serialization or file IO.
    // If --device was passed explicitly on the command line, treat that as
    // a session-only override: do NOT let the config saver overwrite the
    // persisted device_name with whatever this CLI session opened. That
    // turns "vicash --device 1" into a one-time test, not a sticky setting.
    let persist_device_name = cli.device.is_none();
    // Hand the saver the existing presets so a save round-trip does not
    // wipe them. The F1 panel mutates them via shared state when we ship
    // preset editing in a future version; today they are read-only built-ins.
    let initial_presets = cfg.display.color_presets.clone();
    spawn_config_saver(
        shared_settings.clone(),
        capture_ctrl.clone(),
        audio_runtime.as_ref().map(|rt| rt.state.clone()),
        persist_device_name,
        initial_presets,
    );

    // Background update check. Surfaced in the F1 panel; never blocks UI.
    let updater_state = updater::UpdaterState::new();
    updater_state.clone().spawn_background_check();

    match event_loop {
        None => {
            log::info!("headless mode, no preview window. Press Ctrl C to exit.");
            loop {
                std::thread::park();
            }
        }
        Some(el) => preview::run(
            el,
            shared,
            shared_settings,
            capture_info,
            audio_runtime.clone(),
            capture_ctrl,
            metrics,
            relay_slot,
            updater_state,
        )?,
    }

    // Keep audio alive until the preview exits.
    drop(audio_runtime);

    Ok(())
}

/// Look up the current MediaFoundation index of a device by its display
/// name. Returns None if the device is not present right now (unplugged or
/// renamed by a driver update).
fn resolve_device_by_name(name: &str) -> Option<u32> {
    let devices = capture::enumerate().ok()?;
    devices.iter().find_map(|d| {
        if d.human_name() == name {
            Some(capture::index_of(d))
        } else {
            None
        }
    })
}

fn spawn_config_saver(
    settings: Arc<Mutex<settings::Settings>>,
    capture: Arc<capture::CaptureController>,
    audio: Option<Arc<audio::AudioState>>,
    persist_device_name: bool,
    presets: Vec<config::ColorPreset>,
) {
    // Capture the existing name once so we can preserve it when CLI override
    // forbids us from learning a new one. Loaded fresh from disk so we do
    // not depend on the in-memory Settings layout for this niche field.
    let preserved_name = if !persist_device_name {
        config::load().capture.device_name
    } else {
        None
    };
    let _ = std::thread::Builder::new()
        .name("config-saver".into())
        .spawn(move || {
            let mut last_saved: Option<config::Config> = None;
            loop {
                std::thread::sleep(std::time::Duration::from_secs(1));
                let cap_state = capture.state.current.lock().clone();
                let device_name = if persist_device_name {
                    capture.state.current_device_name.lock().clone()
                } else {
                    preserved_name.clone()
                };
                let capture_cfg = config::CaptureConfig {
                    device_name,
                    device_index: Some(capture.last_device_index()),
                    width: cap_state.as_ref().map(|c| c.resolution().width()),
                    height: cap_state.as_ref().map(|c| c.resolution().height()),
                    fps: cap_state.as_ref().map(|c| c.frame_rate()),
                };
                let audio_cfg = match audio.as_ref() {
                    Some(s) => config::AudioConfig {
                        enabled: true,
                        input_device: Some(s.input_name()),
                        output_device: Some(s.output_name()),
                        volume_percent: s.volume(),
                        muted: s.is_muted(),
                        delay_ms: s.delay_ms(),
                        mix_to_mono: s.is_mix_to_mono(),
                    },
                    None => config::AudioConfig {
                        enabled: false,
                        ..config::AudioConfig::default()
                    },
                };
                let snapshot = {
                    let s = settings.lock();
                    config::config_from_runtime(&s, &capture_cfg, &audio_cfg, presets.clone())
                };
                let changed = match &last_saved {
                    Some(prev) => !configs_equivalent(prev, &snapshot),
                    None => true,
                };
                if changed {
                    if let Err(e) = config::save(&snapshot) {
                        log::warn!("config save failed: {e:#}");
                    } else {
                        log::debug!("config saved");
                    }
                    last_saved = Some(snapshot);
                }
            }
        });
}

/// Equality check that ignores fields we never want to trigger a save just on
/// their own (none currently, but a hook for the future).
fn configs_equivalent(a: &config::Config, b: &config::Config) -> bool {
    let serialise = |c: &config::Config| toml::to_string(c).unwrap_or_default();
    serialise(a) == serialise(b)
}
