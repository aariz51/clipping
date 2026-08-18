//! Live view of the background batch, for the app to display.
//!
//! Reads the batch's own log rather than requiring the batch to report in.
//! That means progress shows up for a run that is already in flight, and a
//! crashed batch cannot leave a stale "still working" flag behind: liveness is
//! decided by whether the process is actually there.

use serde::Serialize;
use std::path::PathBuf;
use std::process::Command;

#[derive(Debug, Clone, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct BatchProgress {
    pub running: bool,
    pub done: usize,
    pub skipped: usize,
    pub failed: usize,
    /// Source video currently being worked through.
    pub current_project: Option<String>,
    /// Candidate count for that project.
    pub current_total: Option<usize>,
    /// Human-readable description of the step in flight.
    pub current_step: Option<String>,
    /// Most recent finished clip.
    pub last_done: Option<String>,
    pub pass: Option<String>,
    pub log_path: String,
}

fn log_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("broll-work")
        .join("finish_all.log")
}

fn batch_is_running() -> bool {
    // Match either runner. The serial batch was replaced by a parallel one and
    // a detector that knew only the old name reported "idle" while ten renders
    // were in flight -- worse than showing nothing, because it reads as a
    // stalled job.
    ["render_parallel", "batch_render_all", "retitle_all"]
        .iter()
        .any(|name| {
            Command::new("pgrep")
                .arg("-f")
                .arg(name)
                .output()
                .map(|o| !o.stdout.is_empty())
                .unwrap_or(false)
        })
}

/// Parse the batch log into something the UI can render.
pub fn read() -> BatchProgress {
    let path = log_path();
    let mut p = BatchProgress {
        running: batch_is_running(),
        log_path: path.to_string_lossy().to_string(),
        ..Default::default()
    };
    let Ok(text) = std::fs::read_to_string(&path) else {
        return p;
    };

    // Counts come from the whole log; "current" from the last mention, because
    // the batch works strictly in order.
    for line in text.lines() {
        let t = line.trim_start();
        if t.starts_with("DONE ") {
            p.done += 1;
            p.last_done = t
                .rsplit('/')
                .next()
                .map(|s| s.to_string())
                .or_else(|| Some(t.to_string()));
        } else if t.starts_with("SKIP ") {
            p.skipped += 1;
        } else if t.starts_with("FAIL ") {
            p.failed += 1;
        } else if let Some(rest) = t.strip_prefix("PROJECT ") {
            // "PROJECT <id> (<file>): <n> candidate(s)"
            if let Some(open) = rest.find('(') {
                let name = rest[open + 1..].split(')').next().unwrap_or("").to_string();
                if !name.is_empty() {
                    p.current_project = Some(name);
                }
            }
            p.current_total = rest
                .rsplit(':')
                .next()
                .and_then(|s| s.trim().split_whitespace().next().map(str::to_string))
                .and_then(|n| n.parse().ok());
        } else if let Some(rest) = t.strip_prefix("=== batch pass ") {
            p.pass = rest.split_whitespace().next().map(str::to_string);
        } else if t.starts_with("[broll]") || t.starts_with("[title]") || t.starts_with("[outro]")
            || t.starts_with("[sfx]")
        {
            // The most recent stage line is the best description of "now".
            p.current_step = Some(t.to_string());
        } else if let Some(rest) = t.strip_prefix("TITLE ") {
            p.current_step = Some(format!("titling: {rest}"));
        }
    }
    if !p.running {
        p.current_step = None;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_reports_a_log_path_even_with_no_log() {
        // The UI shows the path so a stuck run can be inspected; it must be
        // present whether or not a batch has ever run.
        assert!(read().log_path.ends_with("finish_all.log"));
    }
}
