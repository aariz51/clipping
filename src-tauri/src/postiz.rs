//! Publishing finished clips to Postiz channels.
//!
//! Thin wrapper over the `postiz_post.py` sidecar, which speaks the Postiz
//! public API (upload, then create a post referencing the uploaded media).

use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

const POSTIZ_PY: &str = include_str!("../assets/postiz_post.py");

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Channel {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    /// Provider slug, e.g. `instagram-standalone`, `tiktok`, `x`.
    #[serde(default)]
    pub identifier: Option<String>,
    #[serde(default)]
    pub profile: Option<String>,
    #[serde(default)]
    pub disabled: bool,
}

pub fn configured() -> bool {
    std::env::var("POSTIZ_API_KEY")
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false)
}

fn sidecar() -> Option<PathBuf> {
    let base = dirs::data_local_dir()
        .or_else(dirs::data_dir)
        .unwrap_or_else(std::env::temp_dir)
        .join("autoshorts")
        .join("postiz");
    std::fs::create_dir_all(&base).ok()?;
    let script = base.join("postiz_post.py");
    let stale = std::fs::read_to_string(&script)
        .map(|existing| existing != POSTIZ_PY)
        .unwrap_or(true);
    if stale {
        std::fs::write(&script, POSTIZ_PY).ok()?;
    }
    Some(script)
}

fn run(args: &[&std::ffi::OsStr]) -> Result<String, String> {
    if !configured() {
        return Err("POSTIZ_API_KEY is not set (Postiz: Settings > Developers)".to_string());
    }
    let python = crate::pyenv::find_any().ok_or_else(|| "no Python available".to_string())?;
    let script = sidecar().ok_or_else(|| "could not materialise postiz sidecar".to_string())?;

    let out = Command::new(python)
        .arg(&script)
        .args(args)
        .output()
        .map_err(|e| format!("failed to run postiz sidecar: {e}"))?;

    let stderr = String::from_utf8_lossy(&out.stderr);
    for line in stderr.lines().filter(|l| l.starts_with("[postiz]")) {
        eprintln!("{line}");
    }
    if !out.status.success() {
        let tail: Vec<&str> = stderr.lines().rev().take(4).collect();
        return Err(tail.into_iter().rev().collect::<Vec<_>>().join(" | "));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

/// Channels connected in Postiz, minus any the user has disabled.
pub fn channels() -> Result<Vec<Channel>, String> {
    let stdout = run(&[std::ffi::OsStr::new("integrations")])?;
    let parsed: Vec<Channel> = serde_json::from_str(stdout.trim())
        .map_err(|e| format!("could not parse Postiz channels: {e}"))?;
    Ok(parsed.into_iter().filter(|c| !c.disabled).collect())
}

/// Publish `video` to the given channels. Empty `channel_ids` posts to all.
///
/// `when` schedules for an ISO timestamp; `None` publishes immediately.
pub fn publish(
    video: &Path,
    caption: &str,
    channel_ids: &[String],
    when: Option<&str>,
    dry_run: bool,
) -> Result<String, String> {
    use std::ffi::OsStr;

    let mut args: Vec<&OsStr> = vec![
        OsStr::new("post"),
        OsStr::new("--video"),
        video.as_os_str(),
        OsStr::new("--content"),
        OsStr::new(caption),
    ];
    for id in channel_ids {
        args.push(OsStr::new("--integration"));
        args.push(OsStr::new(id));
    }
    if let Some(w) = when {
        args.push(OsStr::new("--when"));
        args.push(OsStr::new(w));
    }
    if dry_run {
        args.push(OsStr::new("--dry-run"));
    }
    run(&args)
}
