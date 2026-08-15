//! Branded end card appended to a rendered clip.
//!
//! Shows the app logo and name while a download line is spoken in a voice taken
//! from the clip itself. Speaker choice prefers a female voice when the clip
//! contains one, falling back to whoever is speaking otherwise.
//!
//! Entirely optional: a project without branding, a missing logo, or an
//! unavailable TTS environment all leave the clip untouched rather than
//! failing the render.

use std::path::{Path, PathBuf};
use std::process::Command;

const OUTRO_PY: &str = include_str!("../assets/outro.py");
const VOICE_PICK_PY: &str = include_str!("../assets/voice_pick.py");
const TTS_CLONE_PY: &str = include_str!("../assets/tts_clone.py");

/// Interpreter for the voice-cloning sidecar.
///
/// Separate from the app's own interpreter: the TTS stack has no wheels for
/// Python 3.14, so it lives in its own 3.11 environment.
fn tts_python() -> PathBuf {
    std::env::var("AUTOSHORTS_TTS_PYTHON")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_else(std::env::temp_dir)
                .join("tts-venv/bin/python")
        })
}

/// Materialise the sidecars next to each other so `outro.py` can find its
/// siblings by directory.
fn assets_dir() -> Option<PathBuf> {
    let base = dirs::data_local_dir()
        .or_else(dirs::data_dir)
        .unwrap_or_else(std::env::temp_dir)
        .join("autoshorts")
        .join("outro");
    std::fs::create_dir_all(&base).ok()?;

    for (name, body) in [
        ("outro.py", OUTRO_PY),
        ("voice_pick.py", VOICE_PICK_PY),
        ("tts_clone.py", TTS_CLONE_PY),
    ] {
        let path = base.join(name);
        let stale = std::fs::read_to_string(&path)
            .map(|existing| existing != body)
            .unwrap_or(true);
        if stale {
            std::fs::write(&path, body).ok()?;
        }
    }
    Some(base)
}

/// True when an end card can be drawn. Voice cloning is checked separately so a
/// silent card is still produced when only the TTS environment is missing.
pub fn available() -> bool {
    crate::pyenv::find_with_module("PIL").is_some()
}

/// True when the outro can also speak, rather than being silent.
pub fn can_clone_voice() -> bool {
    tts_python().exists()
}

/// Append the end card to `clip`, writing to `output`.
///
/// `transcript` supplies clip-relative word timings so speaker windows can be
/// separated; without it the middle of the clip is sampled instead.
pub fn append(
    clip: &Path,
    app_name: &str,
    logo: Option<&Path>,
    transcript: Option<&Path>,
    output: &Path,
) -> Result<PathBuf, String> {
    if app_name.trim().is_empty() {
        return Err("no app name configured".to_string());
    }
    let python = crate::pyenv::find_with_module("PIL")
        .ok_or_else(|| "no Python with Pillow available".to_string())?;
    let assets = assets_dir().ok_or_else(|| "could not materialise outro sidecars".to_string())?;

    let mut cmd = Command::new(python);
    cmd.arg(assets.join("outro.py"))
        .arg("--clip")
        .arg(clip)
        .arg("--app-name")
        .arg(app_name)
        .arg("--output")
        .arg(output)
        .arg("--assets")
        .arg(&assets)
        .arg("--tts-python")
        .arg(tts_python());
    if let Some(l) = logo {
        cmd.arg("--logo").arg(l);
    }
    if let Some(t) = transcript {
        cmd.arg("--transcript").arg(t);
    }

    let out = cmd
        .output()
        .map_err(|e| format!("failed to run outro builder: {e}"))?;

    let stderr = String::from_utf8_lossy(&out.stderr);
    for line in stderr
        .lines()
        .filter(|l| l.starts_with("[outro]") || l.starts_with("[voice]") || l.starts_with("[tts]"))
    {
        eprintln!("{line}");
    }

    if !out.status.success() {
        let tail: Vec<&str> = stderr.lines().rev().take(4).collect();
        return Err(tail.into_iter().rev().collect::<Vec<_>>().join(" | "));
    }
    if !output.exists() {
        return Err("outro builder reported success but produced no file".to_string());
    }
    Ok(output.to_path_buf())
}
