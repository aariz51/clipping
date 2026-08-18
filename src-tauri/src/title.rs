//! A persistent title banner across the top of a finished clip.
//!
//! Two parts: a short headline written from what the clip actually says, and a
//! Pillow-rendered plate burned in for the clip's whole duration. The sidecar
//! measures where the speaker's face is and fits the banner into clear space
//! above it, so the title never covers the person talking.

use std::path::{Path, PathBuf};
use std::process::Command;

const TITLE_PY: &str = include_str!("../assets/title_bar.py");
const YUNET: &[u8] = include_bytes!("../assets/face_detection_yunet_2023mar.onnx");

/// True when a Pillow-capable interpreter is available.
pub fn available() -> bool {
    crate::pyenv::find_with_module("PIL").is_some()
}

fn sidecar() -> Option<PathBuf> {
    let base = dirs::data_local_dir()
        .or_else(dirs::data_dir)
        .unwrap_or_else(std::env::temp_dir)
        .join("autoshorts")
        .join("title");
    std::fs::create_dir_all(&base).ok()?;
    let script = base.join("title_bar.py");
    let stale = std::fs::read_to_string(&script)
        .map(|existing| existing != TITLE_PY)
        .unwrap_or(true);
    if stale {
        std::fs::write(&script, TITLE_PY).ok()?;
    }
    let model = base.join("face_detection_yunet_2023mar.onnx");
    let model_stale = std::fs::metadata(&model)
        .map(|m| m.len() != YUNET.len() as u64)
        .unwrap_or(true);
    if model_stale {
        std::fs::write(&model, YUNET).ok()?;
    }
    Some(script)
}

/// Shorten a hook into a headline without calling a model.
///
/// Used when no Anthropic credential is available, and as the floor when the
/// model returns something unusable. Keeps the opening words, which is where a
/// hook carries its subject.
pub fn fallback_title(hook: &str, project_name: Option<&str>) -> String {
    let cleaned: String = hook
        .trim()
        .trim_start_matches(|c: char| !c.is_alphanumeric())
        .to_string();
    let words: Vec<&str> = cleaned.split_whitespace().take(6).collect();
    if words.is_empty() {
        return project_name.unwrap_or("Watch This").to_string();
    }
    words.join(" ").trim_end_matches(&[',', '.', ';', ':'][..]).to_string()
}

/// Ask Claude for a 3-5 word headline describing what the clip says.
pub async fn write_title(transcript_excerpt: &str, hook: &str) -> String {
    let credential = ["ANTHROPIC_API_KEY", "ANTHROPIC_OAUTH_TOKEN"]
        .into_iter()
        .find_map(|k| std::env::var(k).ok())
        .filter(|v| !v.trim().is_empty());
    let Some(key) = credential else {
        return fallback_title(hook, None);
    };

    // The earlier prompt used "The Truth About Bottled Water" as its example
    // and the model latched onto the formula: two clips from one video, with
    // clearly different hooks, both came back as "The Truth About Cadbury
    // Chocolate". Identical banners across a whole video defeat the point of a
    // title, so the prompt now asks for the specific claim and rules out the
    // stock phrasing.
    let prompt = format!(
        "Write the on-screen title for a short vertical video.\n\n\
Rules:\n\
- 3 to 6 words. Never more than 6.\n\
- Name the SPECIFIC thing this clip reveals, not the general topic. Two clips \
from the same video must get clearly different titles.\n\
- Do NOT start with \"The Truth About\". Do not use \"Secrets\", \"Exposed\" \
or \"You Won't Believe\".\n\
- Concrete nouns a viewer understands instantly. No quotes, no emoji, no \
trailing punctuation, no hashtags.\n\n\
Good: \"Bagels Beat Muffins For Sugar\", \"Kraft Bought Cadbury In 2010\"\n\
Bad: \"The Truth About Food\", \"Shocking Facts Revealed\"\n\n\
Reply with the title only.\n\n\
This clip's hook: {hook}\n\nWhat is said in this clip:\n{transcript_excerpt}"
    );

    match crate::llm::ask_claude_text(&prompt, &key).await {
        Ok(text) => {
            let title = text
                .lines()
                .map(str::trim)
                .find(|l| !l.is_empty())
                .unwrap_or("")
                .trim_matches(|c: char| c == '"' || c == '\'' || c == '*')
                .trim_end_matches(['.', '!'])
                .to_string();
            let words = title.split_whitespace().count();
            // Guard against a model that ignores the limit and returns a
            // sentence: an over-long title would wrap into a plate deep enough
            // to cover the speaker.
            if title.is_empty() || words > 8 {
                fallback_title(hook, None)
            } else {
                title
            }
        }
        Err(_) => fallback_title(hook, None),
    }
}

/// Burn `text` across the top of `video`, returning the written file.
pub fn apply(video: &Path, text: &str, output: &Path) -> Result<PathBuf, String> {
    let python = crate::pyenv::find_with_module("PIL")
        .ok_or_else(|| "no Python with Pillow available".to_string())?;
    let script = sidecar().ok_or_else(|| "could not materialise title sidecar".to_string())?;

    let out = Command::new(python)
        .arg(&script)
        .arg("--video")
        .arg(video)
        .arg("--text")
        .arg(text)
        .arg("--output")
        .arg(output)
        .arg("--assets")
        .arg(script.parent().unwrap_or(Path::new(".")))
        .output()
        .map_err(|e| format!("failed to run title sidecar: {e}"))?;

    let stderr = String::from_utf8_lossy(&out.stderr);
    for line in stderr.lines().filter(|l| l.starts_with("[title]")) {
        eprintln!("{line}");
    }
    if !out.status.success() {
        let tail: Vec<&str> = stderr.lines().rev().take(4).collect();
        return Err(format!(
            "title overlay failed: {}",
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
        return Err("title overlay reported success but produced no file".to_string());
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_keeps_the_opening_words() {
        let t = fallback_title(
            "Ultra-processed foods trigger an addictive response so powerful that people cannot stop",
            None,
        );
        assert!(t.split_whitespace().count() <= 6, "{t}");
        assert!(t.starts_with("Ultra-processed"), "{t}");
    }

    #[test]
    fn fallback_handles_an_empty_hook() {
        assert_eq!(fallback_title("   ", Some("SafeChoice")), "SafeChoice");
        assert_eq!(fallback_title("", None), "Watch This");
    }

    #[test]
    fn fallback_strips_leading_punctuation_and_trailing_commas() {
        let t = fallback_title("\"So, here is the thing,", None);
        assert!(!t.starts_with('"'), "{t}");
        assert!(!t.ends_with(','), "{t}");
    }
}
