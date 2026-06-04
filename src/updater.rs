//! GitHub-Releases-backed auto-updater. On startup we ask the API for the
//! latest release, compare its semver against `CARGO_PKG_VERSION`, and
//! offer the user a one-click swap in the F1 panel. The swap uses the
//! Windows file-rename-self trick: a running .exe cannot be overwritten,
//! but it CAN be renamed, so we move the live binary out of the way and
//! drop the freshly downloaded one into its slot before respawning.

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use std::io::Read;
use std::path::PathBuf;
use std::time::Duration;

const REPO: &str = "caaatto/vicash";
pub const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Clone)]
pub struct UpdateInfo {
    pub latest_version: String,
    pub asset_url: String,
    pub notes: String,
}

#[derive(Deserialize)]
struct ReleaseJson {
    tag_name: String,
    #[serde(default)]
    body: String,
    #[serde(default)]
    assets: Vec<AssetJson>,
}

#[derive(Deserialize)]
struct AssetJson {
    name: String,
    browser_download_url: String,
}

/// Query GitHub for the latest release. Returns `Some(info)` if it is
/// strictly newer than `CURRENT_VERSION`, `None` if we are up to date.
pub fn check_for_update() -> Result<Option<UpdateInfo>> {
    let url = format!("https://api.github.com/repos/{REPO}/releases/latest");
    let resp = ureq::get(&url)
        .set("User-Agent", &format!("vicash/{CURRENT_VERSION}"))
        .set("Accept", "application/vnd.github+json")
        .timeout(Duration::from_secs(5))
        .call()
        .context("GitHub releases API call failed")?;
    let body = resp.into_string().context("could not read release body")?;
    let release: ReleaseJson = serde_json::from_str(&body).context("malformed release JSON")?;
    let latest = strip_v_prefix(&release.tag_name);
    if !is_newer(latest, CURRENT_VERSION) {
        log::info!("updater: up to date (running {CURRENT_VERSION}, latest {latest})");
        return Ok(None);
    }
    let exe = release
        .assets
        .iter()
        .find(|a| a.name.eq_ignore_ascii_case("vicash.exe"))
        .ok_or_else(|| anyhow!("release {} has no vicash.exe asset", release.tag_name))?;
    log::info!("updater: {latest} available (running {CURRENT_VERSION})");
    Ok(Some(UpdateInfo {
        latest_version: latest.to_string(),
        asset_url: exe.browser_download_url.clone(),
        notes: release.body,
    }))
}

/// Download the new binary and swap it into place using the rename-self
/// pattern. The caller is responsible for restarting the process; this
/// function only prepares the filesystem.
pub fn apply_update(asset_url: &str) -> Result<()> {
    let exe = std::env::current_exe().context("current_exe() failed")?;
    let dir = exe
        .parent()
        .ok_or_else(|| anyhow!("exe has no parent dir"))?
        .to_path_buf();
    let new_path = dir.join("vicash.exe.new");
    let old_path = dir.join("vicash.exe.old");

    // Clean any stale leftover from a previous run before downloading.
    let _ = std::fs::remove_file(&new_path);

    let resp = ureq::get(asset_url)
        .set("User-Agent", &format!("vicash/{CURRENT_VERSION}"))
        .timeout(Duration::from_secs(120))
        .call()
        .context("download of new binary failed")?;
    let mut reader = resp.into_reader();
    let mut file = std::fs::File::create(&new_path)
        .with_context(|| format!("could not create {}", new_path.display()))?;
    std::io::copy(&mut reader, &mut file).context("write of new binary failed")?;
    file.sync_all().ok();
    drop(file);

    // Get rid of any prior .old before we shuffle this run's binaries.
    let _ = std::fs::remove_file(&old_path);

    std::fs::rename(&exe, &old_path)
        .with_context(|| format!("could not rename running exe out of the way: {}", exe.display()))?;
    std::fs::rename(&new_path, &exe)
        .with_context(|| format!("could not move new exe into place: {}", exe.display()))?;
    log::info!("updater: binary swapped, ready to relaunch");
    Ok(())
}

/// Spawn the (just-replaced) current_exe with the same args and terminate
/// this process. Called immediately after `apply_update`.
pub fn relaunch_and_exit() -> ! {
    let exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("vicash.exe"));
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = std::process::Command::new(&exe).args(&args).spawn() {
        log::error!("updater: relaunch failed: {e:#}");
    }
    std::process::exit(0);
}

/// Best-effort cleanup of the previous run's `vicash.exe.old`. Safe to call
/// at any startup; silently ignores if there is no leftover.
pub fn cleanup_previous() {
    let Ok(exe) = std::env::current_exe() else { return };
    let Some(dir) = exe.parent() else { return };
    let old = dir.join("vicash.exe.old");
    if old.exists() {
        match std::fs::remove_file(&old) {
            Ok(_) => log::info!("updater: removed previous vicash.exe.old"),
            // Still locked by the OS occasionally on very fast relaunches;
            // not fatal, we will retry on the next startup.
            Err(e) => log::debug!("updater: could not yet remove old binary: {e}"),
        }
    }
}

fn strip_v_prefix(s: &str) -> &str {
    s.strip_prefix('v').unwrap_or(s)
}

fn is_newer(latest: &str, current: &str) -> bool {
    parse_triple(latest) > parse_triple(current)
}

fn parse_triple(v: &str) -> (u32, u32, u32) {
    let mut iter = v.split(|c: char| c == '.' || c == '-').take(3);
    let a = iter.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let b = iter.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let c = iter.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    (a, b, c)
}

/// Shared state surfaced into the F1 panel: outcome of the last check,
/// progress of an in-flight apply.
#[derive(Default)]
pub struct UpdaterState {
    pub last_check_result: parking_lot::Mutex<UpdateCheck>,
}

#[derive(Default, Clone)]
pub enum UpdateCheck {
    #[default]
    Idle,
    Checking,
    UpToDate,
    Available(UpdateInfo),
    /// Apply-update in flight - show a busy indicator in the panel.
    Applying,
    Failed(String),
}

impl UpdaterState {
    pub fn new() -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self::default())
    }

    pub fn spawn_background_check(self: std::sync::Arc<Self>) {
        // One-shot startup check on a background thread so we do not stall
        // the window while waiting on GitHub.
        std::thread::Builder::new()
            .name("updater-check".into())
            .spawn(move || {
                *self.last_check_result.lock() = UpdateCheck::Checking;
                let outcome = match check_for_update() {
                    Ok(Some(info)) => UpdateCheck::Available(info),
                    Ok(None) => UpdateCheck::UpToDate,
                    Err(e) => UpdateCheck::Failed(format!("{e:#}")),
                };
                *self.last_check_result.lock() = outcome;
            })
            .ok();
    }

    /// Trigger an apply on a background thread so the UI keeps responding.
    /// On success, the process re-launches itself and exits.
    pub fn spawn_apply(self: std::sync::Arc<Self>, info: UpdateInfo) {
        std::thread::Builder::new()
            .name("updater-apply".into())
            .spawn(move || {
                *self.last_check_result.lock() = UpdateCheck::Applying;
                match apply_update(&info.asset_url) {
                    Ok(_) => relaunch_and_exit(),
                    Err(e) => {
                        *self.last_check_result.lock() = UpdateCheck::Failed(format!("{e:#}"));
                    }
                }
            })
            .ok();
    }
}
