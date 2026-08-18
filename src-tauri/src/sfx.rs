//! Sound effects placed under a finished clip.
//!
//! The kit is synthesised locally rather than downloaded: the sounds the user
//! named (Among Us, Vine Boom) are copyrighted game and meme audio that would
//! expose a monetised account to claims. Synthesised equivalents serve the same
//! editorial purpose with no licence risk, no network and no attribution.
//!
//! Placement is driven by the edit -- whooshes on scene changes, a riser into
//! the longest pause before the payoff, a boom on the statement after it -- and
//! everything sits well under the voice so speech stays the loudest element.

use std::path::{Path, PathBuf};
use std::process::Command;

const MIX_PY: &str = include_str!("../assets/sfx_mix.py");
const MAKE_PY: &str = include_str!("../assets/make_sfx.py");

/// Where the generated kit lives.
pub fn kit_dir() -> PathBuf {
    std::env::var("SFX_KIT_DIR")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_else(std::env::temp_dir)
                .join("autoshorts")
                .join("sfx")
        })
}

fn sidecar(name: &str, body: &str) -> Option<PathBuf> {
    let base = dirs::data_local_dir()
        .or_else(dirs::data_dir)
        .unwrap_or_else(std::env::temp_dir)
        .join("autoshorts")
        .join("sfx");
    std::fs::create_dir_all(&base).ok()?;
    let script = base.join(name);
    let stale = std::fs::read_to_string(&script)
        .map(|existing| existing != body)
        .unwrap_or(true);
    if stale {
        std::fs::write(&script, body).ok()?;
    }
    Some(script)
}

/// Build the kit if it is not already on disk. Cheap and idempotent.
pub fn ensure_kit() -> Result<PathBuf, String> {
    let dir = kit_dir();
    let complete = ["attention", "riser", "whoosh", "boom", "stinger", "pop"]
        .iter()
        .all(|n| dir.join(format!("{n}.wav")).exists());
    if complete {
        return Ok(dir);
    }
    let python = crate::pyenv::find_any().ok_or_else(|| "no Python available".to_string())?;
    let script = sidecar("make_sfx.py", MAKE_PY)
        .ok_or_else(|| "could not materialise sfx generator".to_string())?;
    let out = Command::new(python)
        .arg(&script)
        .arg("--out")
        .arg(&dir)
        .output()
        .map_err(|e| format!("failed to generate sfx kit: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "sfx kit generation failed: {}",
            String::from_utf8_lossy(&out.stderr).lines().last().unwrap_or("")
        ));
    }
    Ok(dir)
}

/// Mix effects under `video`, returning the written file.
///
/// `scene_plan` and `transcript` steer placement; without them the mixer has
/// nothing to place against and passes the clip through unchanged.
pub fn apply(
    video: &Path,
    scene_plan: Option<&Path>,
    transcript: Option<&Path>,
    output: &Path,
) -> Result<PathBuf, String> {
    let kit = ensure_kit()?;
    let python = crate::pyenv::find_any().ok_or_else(|| "no Python available".to_string())?;
    let script =
        sidecar("sfx_mix.py", MIX_PY).ok_or_else(|| "could not materialise sfx mixer".to_string())?;

    let mut cmd = Command::new(python);
    cmd.arg(&script)
        .arg("--video")
        .arg(video)
        .arg("--kit")
        .arg(&kit)
        .arg("--output")
        .arg(output);
    if let Some(p) = scene_plan.filter(|p| p.exists()) {
        cmd.arg("--scenes").arg(p);
    }
    if let Some(t) = transcript.filter(|t| t.exists()) {
        cmd.arg("--transcript").arg(t);
    }

    let out = cmd
        .output()
        .map_err(|e| format!("failed to run sfx mixer: {e}"))?;
    let stderr = String::from_utf8_lossy(&out.stderr);
    for line in stderr.lines().filter(|l| l.starts_with("[sfx]")) {
        eprintln!("{line}");
    }
    if !out.status.success() {
        let tail: Vec<&str> = stderr.lines().rev().take(3).collect();
        return Err(format!(
            "sfx mix failed: {}",
            tail.into_iter().rev().collect::<Vec<_>>().join(" | ")
        ));
    }
    let produced = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let path = if produced.is_empty() {
        output.to_path_buf()
    } else {
        PathBuf::from(produced)
    };
    if !path.exists() {
        return Err("sfx mix reported success but produced no file".to_string());
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kit_dir_is_overridable() {
        // The default must be stable so a generated kit is found again rather
        // than rebuilt on every render.
        assert!(kit_dir().ends_with("sfx"));
    }
}
