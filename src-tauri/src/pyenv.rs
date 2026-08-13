//! Locating a Python interpreter that has the packages we need.
//!
//! Optional features (face tracking via OpenCV, offline transcription via
//! Whisper) are installed into a virtualenv rather than the system Python,
//! which is externally managed on macOS/Homebrew and refuses `pip install`.
//! A bare `python3` lookup would miss that venv entirely, so every candidate
//! path is probed for the module actually required.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Interpreters to try, most specific first.
///
/// `AUTOSHORTS_PYTHON` wins so a user can point at any environment. Then any
/// `.venv` beside the binary or the source tree, then whatever is on PATH.
pub fn candidates() -> Vec<String> {
    let mut out: Vec<String> = Vec::new();

    if let Ok(explicit) = std::env::var("AUTOSHORTS_PYTHON") {
        if !explicit.trim().is_empty() {
            out.push(explicit.trim().to_string());
        }
    }

    let venv_rel: &[&str] = if cfg!(windows) {
        &[".venv/Scripts/python.exe", "venv/Scripts/python.exe"]
    } else {
        &[".venv/bin/python", "venv/bin/python"]
    };

    // Walk up from the executable so both `cargo run` and an installed bundle
    // find a venv checked out alongside the project.
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent().map(Path::to_path_buf);
        for _ in 0..6 {
            let Some(d) = dir else { break };
            roots.push(d.clone());
            dir = d.parent().map(Path::to_path_buf);
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd);
    }
    if let Some(home) = dirs::home_dir() {
        roots.push(home.join("autoshorts"));
    }

    for root in roots {
        for rel in venv_rel {
            let candidate = root.join(rel);
            if candidate.exists() {
                out.push(candidate.to_string_lossy().into_owned());
            }
        }
    }

    if cfg!(windows) {
        out.extend(["python".into(), "py".into(), "python3".into()]);
    } else {
        out.extend(["python3".into(), "python".into()]);
    }

    out.dedup();
    out
}

/// First interpreter that can import `module`, if any.
pub fn find_with_module(module: &str) -> Option<String> {
    let probe = format!("import {module}");
    candidates().into_iter().find(|cmd| {
        Command::new(cmd)
            .args(["-c", &probe])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    })
}

/// First interpreter that runs at all, regardless of installed packages.
pub fn find_any() -> Option<String> {
    candidates()
        .into_iter()
        .find(|cmd| Command::new(cmd).arg("--version").output().is_ok())
}

/// Locate an executable installed into a venv's bin directory (e.g. `whisper`),
/// falling back to PATH.
pub fn find_venv_script(name: &str) -> Option<String> {
    for cand in candidates() {
        let path = Path::new(&cand);
        if let Some(bin_dir) = path.parent() {
            let script = if cfg!(windows) {
                bin_dir.join(format!("{name}.exe"))
            } else {
                bin_dir.join(name)
            };
            if script.exists() {
                return Some(script.to_string_lossy().into_owned());
            }
        }
    }

    let on_path = Command::new(name)
        .arg("--help")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    on_path.then(|| name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_list_prefers_venv_over_path() {
        let all = candidates();
        let venv_idx = all.iter().position(|c| c.contains(".venv"));
        let path_idx = all.iter().position(|c| c == "python3" || c == "python");
        if let (Some(v), Some(p)) = (venv_idx, path_idx) {
            assert!(v < p, "venv interpreter must be probed before bare PATH python");
        }
    }

    /// The app calls `dotenvy::dotenv()`, which walks up from the working
    /// directory. Tauri launches the binary from `src-tauri/`, so this confirms
    /// the repo-root `.env` is still discovered from there.
    #[test]
    #[ignore]
    fn dotenv_is_discoverable_from_src_tauri() {
        dotenvy::dotenv().expect("no .env found walking up from src-tauri");
        let key = std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY missing");
        assert!(key.starts_with("sk-or-"), "unexpected key format");
        assert_eq!(std::env::var("LLM_PROVIDER").as_deref(), Ok("openrouter"));
    }

    // Environment-dependent: run explicitly with `cargo test -- --ignored`.
    #[test]
    #[ignore]
    fn resolves_local_optional_deps() {
        assert!(find_with_module("cv2").is_some(), "cv2 not resolvable");
        assert!(find_with_module("whisper").is_some(), "whisper not resolvable");
        assert!(find_venv_script("whisper").is_some(), "whisper CLI not resolvable");
    }
}
