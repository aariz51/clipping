//! B-roll enrichment via the pinned `b-rolls` skill.
//!
//! Runs the skill's own workflow (`prepare_project.py` -> scene plan ->
//! `render_project.py` -> `verify_output.py`) through a sidecar that automates
//! the steps the skill leaves to an agent. Nothing here reimplements the
//! rendering: the pinned `video-use` pipeline still does that work, which is
//! what keeps the output identical to the skill's own results.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Serialize;

use crate::models::TranscriptWord;

const BROLL_PY: &str = include_str!("../assets/broll_pipeline.py");

/// Where the b-rolls skill checkout lives. Overridable for a different clone.
pub fn skill_dir() -> PathBuf {
    std::env::var("BROLLS_SKILL_DIR")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_else(std::env::temp_dir)
                .join("b-rolls-ref")
        })
}

/// True when the skill checkout and a Pillow-capable interpreter are present.
pub fn available() -> bool {
    skill_dir().join("scripts/render_project.py").exists()
        && crate::pyenv::find_with_module("PIL").is_some()
}

fn sidecar() -> Option<PathBuf> {
    let base = dirs::data_local_dir()
        .or_else(dirs::data_dir)
        .unwrap_or_else(std::env::temp_dir)
        .join("autoshorts")
        .join("broll");
    std::fs::create_dir_all(&base).ok()?;
    let script = base.join("broll_pipeline.py");
    let stale = std::fs::read_to_string(&script)
        .map(|existing| existing != BROLL_PY)
        .unwrap_or(true);
    if stale {
        std::fs::write(&script, BROLL_PY).ok()?;
    }
    Some(script)
}

#[derive(Serialize)]
struct ClipTranscript<'a> {
    words: Vec<ClipWord<'a>>,
}

#[derive(Serialize)]
struct ClipWord<'a> {
    text: &'a str,
    start: f64,
    end: f64,
}

/// Write the clip-relative transcript the planner reads.
///
/// Timings are rebased to the clip because the sidecar plans against scene
/// slots that start at zero.
fn write_transcript(
    words: &[TranscriptWord],
    start_sec: f64,
    end_sec: f64,
    dir: &Path,
) -> Option<PathBuf> {
    let payload = ClipTranscript {
        words: words
            .iter()
            .filter(|w| w.end > start_sec && w.start < end_sec)
            .map(|w| ClipWord {
                text: &w.text,
                start: w.start - start_sec,
                end: w.end - start_sec,
            })
            .collect(),
    };
    if payload.words.is_empty() {
        return None;
    }
    let path = dir.join("broll_transcript.json");
    std::fs::write(&path, serde_json::to_vec(&payload).ok()?).ok()?;
    Some(path)
}

/// Add B-roll to `clip_path`, returning the enriched file.
///
/// `topic` steers scene selection; pass the candidate's hook so the visuals
/// track what the clip is actually about.
pub fn enrich(
    clip_path: &Path,
    words: &[TranscriptWord],
    start_sec: f64,
    end_sec: f64,
    topic: &str,
    output: &Path,
) -> Result<PathBuf, String> {
    if !available() {
        return Err(format!(
            "b-rolls skill not found at {} (set BROLLS_SKILL_DIR), or Pillow is missing",
            skill_dir().display()
        ));
    }
    let python = crate::pyenv::find_with_module("PIL")
        .ok_or_else(|| "no Python with Pillow available".to_string())?;
    let script = sidecar().ok_or_else(|| "could not materialise broll sidecar".to_string())?;

    let work = clip_path.parent().unwrap_or(Path::new("."));
    let transcript = write_transcript(words, start_sec, end_sec, work);

    let mut cmd = Command::new(python);
    cmd.arg(&script)
        .arg("--clip")
        .arg(clip_path)
        .arg("--topic")
        .arg(topic)
        .arg("--skill-dir")
        .arg(skill_dir())
        .arg("--output")
        .arg(output);
    if let Some(t) = &transcript {
        cmd.arg("--transcript").arg(t);
    }

    let out = cmd
        .output()
        .map_err(|e| format!("failed to run b-roll pipeline: {e}"))?;

    // The sidecar narrates progress on stderr; surface it for diagnosis.
    let stderr = String::from_utf8_lossy(&out.stderr);
    for line in stderr.lines().filter(|l| l.starts_with("[broll]")) {
        eprintln!("{line}");
    }

    if !out.status.success() {
        let tail: Vec<&str> = stderr.lines().rev().take(6).collect();
        return Err(format!(
            "b-roll pipeline failed: {}",
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
        return Err("b-roll pipeline reported success but produced no file".to_string());
    }
    Ok(path)
}
