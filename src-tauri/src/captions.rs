//! Burned-in captions that do not depend on ffmpeg's `drawtext` filter.
//!
//! `drawtext` needs a build linked against libfreetype, which current Homebrew
//! ffmpeg omits — asking for it aborts the render outright. `overlay` is a core
//! filter present in every build, so captions are rasterised by a Pillow
//! sidecar and composited as a transparent image sequence.
//!
//! As a side benefit the text keeps its punctuation; the drawtext path strips
//! every non-alphanumeric character, turning `don't` into `DONT`.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::Serialize;

use crate::models::TranscriptWord;

const CAPTIONS_PY: &str = include_str!("../assets/captions.py");

#[derive(Serialize)]
struct Chunk {
    text: String,
    start: f64,
    end: f64,
}

#[derive(Serialize)]
struct Spec {
    width: i64,
    height: i64,
    duration: f64,
    style: String,
    out_dir: String,
    chunks: Vec<Chunk>,
}

/// A rendered caption track plus the directory holding its frames.
///
/// The directory is deleted on drop, so it must outlive the ffmpeg call.
pub struct CaptionTrack {
    pub concat_list: PathBuf,
    dir: PathBuf,
}

impl Drop for CaptionTrack {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn sidecar_path() -> Option<PathBuf> {
    let base = dirs::data_local_dir()
        .or_else(dirs::data_dir)
        .unwrap_or_else(std::env::temp_dir)
        .join("autoshorts")
        .join("captions");
    std::fs::create_dir_all(&base).ok()?;

    let script = base.join("captions.py");
    let stale = std::fs::read_to_string(&script)
        .map(|existing| existing != CAPTIONS_PY)
        .unwrap_or(true);
    if stale {
        std::fs::write(&script, CAPTIONS_PY).ok()?;
    }
    Some(script)
}

/// True when captions can be rendered without `drawtext`.
pub fn available() -> bool {
    crate::pyenv::find_with_module("PIL").is_some()
}

/// Group words into short on-screen phrases.
///
/// Two words at a time matches the fast-paced style used by the drawtext
/// renderer and keeps each caption readable on a phone.
fn chunk_words(words: &[TranscriptWord], start_sec: f64, end_sec: f64) -> Vec<Chunk> {
    let in_range: Vec<&TranscriptWord> = words
        .iter()
        .filter(|w| w.end > start_sec && w.start < end_sec)
        .collect();

    in_range
        .chunks(2)
        .filter_map(|pair| {
            let first = pair.first()?;
            let last = pair.last()?;
            // Times are relative to the clip: fast input seeking resets PTS.
            let start = (first.start - start_sec).max(0.0);
            let end = (last.end - start_sec).min(end_sec - start_sec);
            if end <= start {
                return None;
            }
            let text = pair
                .iter()
                .map(|w| w.text.trim())
                .filter(|t| !t.is_empty())
                .collect::<Vec<_>>()
                .join(" ")
                .to_uppercase();
            if text.is_empty() {
                return None;
            }
            Some(Chunk { text, start, end })
        })
        .collect()
}

/// Render a caption track for one clip.
///
/// Returns `None` whenever captions cannot be produced, leaving the clip to
/// render without them rather than failing.
pub fn render_track(
    words: &[TranscriptWord],
    start_sec: f64,
    end_sec: f64,
    width: i64,
    height: i64,
    style: &str,
    work_root: &Path,
) -> Option<CaptionTrack> {
    let duration = end_sec - start_sec;
    if duration <= 0.0 || width <= 0 || height <= 0 {
        return None;
    }

    let chunks = chunk_words(words, start_sec, end_sec);
    if chunks.is_empty() {
        return None;
    }

    let python = crate::pyenv::find_with_module("PIL")?;
    let script = sidecar_path()?;

    let dir = work_root.join(format!("captions-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).ok()?;

    let spec = Spec {
        width,
        height,
        duration,
        style: style.to_string(),
        out_dir: dir.to_string_lossy().into_owned(),
        chunks,
    };
    let payload = serde_json::to_vec(&spec).ok()?;

    let mut child = Command::new(python)
        .arg(&script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;

    if let Some(stdin) = child.stdin.as_mut() {
        if stdin.write_all(&payload).is_err() {
            let _ = std::fs::remove_dir_all(&dir);
            return None;
        }
    }

    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        eprintln!(
            "captions: renderer failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
        let _ = std::fs::remove_dir_all(&dir);
        return None;
    }

    let list = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if list.is_empty() || !Path::new(&list).exists() {
        let _ = std::fs::remove_dir_all(&dir);
        return None;
    }

    Some(CaptionTrack {
        concat_list: PathBuf::from(list),
        dir,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(text: &str, start: f64, end: f64) -> TranscriptWord {
        TranscriptWord {
            text: text.to_string(),
            start,
            end,
            speaker: None,
        }
    }

    #[test]
    fn chunks_pairs_of_words_relative_to_clip_start() {
        let words = vec![
            word("so", 10.0, 10.4),
            word("I", 10.4, 10.6),
            word("was", 10.6, 11.0),
            word("told", 11.0, 11.4),
        ];
        let out = chunk_words(&words, 10.0, 12.0);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].text, "SO I");
        assert!((out[0].start - 0.0).abs() < 1e-9);
        assert_eq!(out[1].text, "WAS TOLD");
        assert!((out[1].start - 0.6).abs() < 1e-9);
    }

    #[test]
    fn keeps_punctuation_that_drawtext_would_strip() {
        let words = vec![word("don't", 0.0, 0.5), word("panic!", 0.5, 1.0)];
        let out = chunk_words(&words, 0.0, 2.0);
        assert_eq!(out[0].text, "DON'T PANIC!");
    }

    #[test]
    fn excludes_words_outside_the_clip() {
        let words = vec![
            word("before", 0.0, 1.0),
            word("inside", 5.0, 5.5),
            word("after", 20.0, 21.0),
        ];
        let out = chunk_words(&words, 4.0, 6.0);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "INSIDE");
    }
}
