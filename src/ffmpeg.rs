//! Locate ffmpeg.exe. A plain `Command::new("ffmpeg")` only sees the PATH
//! vicash inherited at launch, so an ffmpeg installed while vicash (or the
//! Explorer that started it) was already running stays invisible until the
//! next login. Check next to vicash.exe, the live PATH and the usual install
//! spots (winget, scoop, chocolatey) ourselves before falling back to the
//! plain lookup.

use std::path::PathBuf;
use std::process::Command;

pub fn command() -> Command {
    Command::new(path())
}

fn path() -> PathBuf {
    candidates()
        .into_iter()
        .find(|p| p.is_file())
        .unwrap_or_else(|| PathBuf::from("ffmpeg"))
}

fn candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(PathBuf::from)) {
        out.push(dir.join("ffmpeg.exe"));
    }
    if let Some(path) = std::env::var_os("PATH") {
        out.extend(std::env::split_paths(&path).map(|d| d.join("ffmpeg.exe")));
    }
    if let Some(local) = std::env::var_os("LOCALAPPDATA").map(PathBuf::from) {
        let winget = local.join("Microsoft").join("WinGet");
        out.push(winget.join("Links").join("ffmpeg.exe"));
        // winget unpacks zip builds to Packages/<id>/<build>/bin/ffmpeg.exe
        // and only adds that bin dir to the user PATH in the registry.
        for pkg in read_dir(winget.join("Packages")) {
            let name = pkg.file_name().unwrap_or_default().to_string_lossy().to_ascii_lowercase();
            if name.contains("ffmpeg") {
                out.extend(read_dir(pkg).into_iter().map(|b| b.join("bin").join("ffmpeg.exe")));
            }
        }
    }
    if let Some(home) = std::env::var_os("USERPROFILE").map(PathBuf::from) {
        out.push(home.join("scoop").join("shims").join("ffmpeg.exe"));
    }
    if let Some(data) = std::env::var_os("ProgramData").map(PathBuf::from) {
        out.push(data.join("chocolatey").join("bin").join("ffmpeg.exe"));
    }
    out.push(PathBuf::from(r"C:\ffmpeg\bin\ffmpeg.exe"));
    out
}

fn read_dir(dir: PathBuf) -> Vec<PathBuf> {
    std::fs::read_dir(dir)
        .map(|it| it.flatten().map(|e| e.path()).collect())
        .unwrap_or_default()
}
