//! Speaker-aware vertical cropping.
//!
//! The stock render centre-crops every clip, which decapitates any speaker who
//! is not sitting dead centre — common in interviews and podcast footage. This
//! module runs a small OpenCV sidecar that follows the primary face and returns
//! either a fixed offset or a time-varying ffmpeg `crop` expression.
//!
//! Every failure path degrades to `None`, in which case the caller keeps the
//! original centre crop. Face tracking is an enhancement, never a hard
//! dependency.

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use serde::Deserialize;

const FACETRACK_PY: &str = include_str!("../assets/facetrack.py");
const YUNET_MODEL: &[u8] = include_bytes!("../assets/face_detection_yunet_2023mar.onnx");

#[derive(Debug, Deserialize)]
struct TrackResult {
    mode: String,
    #[serde(default)]
    x: Option<f64>,
    #[serde(default)]
    y: Option<f64>,
    #[serde(default)]
    x_expr: Option<String>,
    #[serde(default)]
    y_expr: Option<String>,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    coverage: Option<f64>,
    #[serde(default)]
    cuts: Option<i64>,
}

/// Resolved crop offsets, as ffmpeg expression strings.
#[derive(Debug, Clone)]
pub struct CropPlan {
    pub x: String,
    pub y: String,
    /// Human-readable note surfaced in logs / UI so a bad track is diagnosable.
    pub summary: String,
}

/// Locate a Python interpreter that can `import cv2`.
///
/// Checked once per process: an explicit override, then a virtualenv beside the
/// app, then whatever is on PATH.
fn python_with_cv2() -> Option<&'static str> {
    static PYTHON: OnceLock<Option<String>> = OnceLock::new();
    PYTHON
        .get_or_init(|| crate::pyenv::find_with_module("cv2"))
        .as_deref()
}

/// Write the sidecar script and detector weights next to the app data, so the
/// packaged binary works without the source tree present.
fn materialize_assets() -> Option<(PathBuf, PathBuf)> {
    static ASSETS: OnceLock<Option<(PathBuf, PathBuf)>> = OnceLock::new();
    ASSETS
        .get_or_init(|| {
            let base = dirs::data_local_dir()
                .or_else(dirs::data_dir)
                .unwrap_or_else(std::env::temp_dir)
                .join("autoshorts")
                .join("facetrack");
            std::fs::create_dir_all(&base).ok()?;

            let script = base.join("facetrack.py");
            let model = base.join("face_detection_yunet_2023mar.onnx");

            // Rewrite when absent or stale so app updates ship new logic.
            let script_stale = std::fs::read_to_string(&script)
                .map(|existing| existing != FACETRACK_PY)
                .unwrap_or(true);
            if script_stale {
                std::fs::write(&script, FACETRACK_PY).ok()?;
            }

            let model_stale = std::fs::metadata(&model)
                .map(|m| m.len() != YUNET_MODEL.len() as u64)
                .unwrap_or(true);
            if model_stale {
                std::fs::write(&model, YUNET_MODEL).ok()?;
            }

            Some((script, model))
        })
        .clone()
}

/// True when a speaker-tracked crop is possible on this machine.
pub fn available() -> bool {
    python_with_cv2().is_some()
}

/// Compute a speaker-following crop for `[start_sec, end_sec)` of `source_path`.
///
/// Returns `None` whenever tracking is unavailable or untrustworthy, which
/// leaves the caller on its existing centre crop.
pub fn plan_crop(source_path: &str, start_sec: f64, end_sec: f64) -> Option<CropPlan> {
    if end_sec - start_sec <= 0.0 {
        return None;
    }

    let python = python_with_cv2()?;
    let (script, model) = materialize_assets()?;

    let output = Command::new(python)
        .arg(&script)
        .arg("--video")
        .arg(source_path)
        .arg("--model")
        .arg(&model)
        .arg("--start")
        .arg(format!("{start_sec:.3}"))
        .arg("--end")
        .arg(format!("{end_sec:.3}"))
        .output()
        .ok()?;

    if !output.status.success() {
        eprintln!(
            "facetrack: sidecar exited non-zero: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    // The sidecar prints exactly one JSON object; OpenCV may add stderr noise.
    let line = stdout.lines().rev().find(|l| l.trim_start().starts_with('{'))?;

    let parsed: TrackResult = match serde_json::from_str(line.trim()) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("facetrack: could not parse sidecar output: {e}");
            return None;
        }
    };

    if parsed.mode == "none" {
        eprintln!(
            "facetrack: falling back to centre crop ({})",
            parsed.reason.as_deref().unwrap_or("unspecified")
        );
        return None;
    }

    // Default to ffmpeg's own centred expression for any axis the tracker did
    // not constrain, so a horizontal-only track still centres vertically.
    let default_x = "(in_w-out_w)/2".to_string();
    let default_y = "(in_h-out_h)/2".to_string();

    let x = parsed
        .x_expr
        .clone()
        .or_else(|| parsed.x.map(|v| format!("{}", v.round() as i64)))
        .unwrap_or(default_x);
    let y = parsed
        .y_expr
        .clone()
        .or_else(|| parsed.y.map(|v| format!("{}", v.round() as i64)))
        .unwrap_or(default_y);

    let summary = format!(
        "{} track, coverage {:.0}%, {} cut(s)",
        parsed.mode,
        parsed.coverage.unwrap_or(0.0) * 100.0,
        parsed.cuts.unwrap_or(0)
    );

    Some(CropPlan { x, y, summary })
}
