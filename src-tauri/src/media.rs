use std::{
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{anyhow, Context, Result};
use serde_json::Value;

use crate::models::MediaProbe;

/// Resolve the ffmpeg/ffprobe binary to use.
///
/// Falls back to whatever is on PATH, but `AUTOSHORTS_FFMPEG` / `AUTOSHORTS_FFPROBE`
/// can point at a fuller build (one with libfreetype for captions) without
/// disturbing the system install that other tools may depend on.
pub fn ffmpeg_bin() -> String {
    std::env::var("AUTOSHORTS_FFMPEG")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "ffmpeg".to_string())
}

pub fn ffprobe_bin() -> String {
    std::env::var("AUTOSHORTS_FFPROBE")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "ffprobe".to_string())
}

pub fn command_exists(name: &str) -> bool {
    // Callers pass logical names; honour any configured override for those.
    let resolved = match name {
        "ffmpeg" => ffmpeg_bin(),
        "ffprobe" => ffprobe_bin(),
        other => other.to_string(),
    };
    Command::new(resolved).arg("-version").output().is_ok()
}

/// Whether the installed ffmpeg was compiled with a given filter.
///
/// Builds without libfreetype have no `drawtext`, and asking for it aborts the
/// whole render. Probing first lets a clip render without captions instead of
/// failing outright. Cached: shelling out per clip is wasteful.
pub fn ffmpeg_has_filter(name: &str) -> bool {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    static CACHE: OnceLock<Mutex<HashMap<String, bool>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));

    if let Ok(map) = cache.lock() {
        if let Some(hit) = map.get(name) {
            return *hit;
        }
    }

    let found = Command::new(ffmpeg_bin())
        .args(["-hide_banner", "-filters"])
        .output()
        .map(|out| {
            let listing = String::from_utf8_lossy(&out.stdout);
            // Filter rows look like " ... drawtext  V->V  Draw text ...".
            listing
                .lines()
                .any(|line| line.split_whitespace().nth(1) == Some(name))
        })
        .unwrap_or(false);

    if let Ok(mut map) = cache.lock() {
        map.insert(name.to_string(), found);
    }
    found
}

/// True when this ffmpeg can burn in captions.
pub fn supports_captions() -> bool {
    ffmpeg_has_filter("drawtext")
}

pub fn probe_media(path: &str) -> Result<MediaProbe> {
    if !command_exists("ffprobe") {
        return Err(anyhow!("ffprobe is not installed or not available on PATH"));
    }

    let output = Command::new(ffprobe_bin())
        .args([
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            path,
        ])
        .output()
        .context("running ffprobe")?;

    if !output.status.success() {
        return Err(anyhow!(
            "ffprobe failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    let json: Value = serde_json::from_slice(&output.stdout).context("parsing ffprobe JSON")?;
    let streams = json
        .get("streams")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    let video = streams
        .iter()
        .find(|stream| stream.get("codec_type").and_then(Value::as_str) == Some("video"));
    let audio = streams
        .iter()
        .find(|stream| stream.get("codec_type").and_then(Value::as_str) == Some("audio"));

    let duration_sec = json
        .get("format")
        .and_then(|format| format.get("duration"))
        .and_then(Value::as_str)
        .and_then(|duration| duration.parse::<f64>().ok());

    Ok(MediaProbe {
        duration_sec,
        has_video: video.is_some(),
        width: video
            .and_then(|stream| stream.get("width"))
            .and_then(Value::as_i64),
        height: video
            .and_then(|stream| stream.get("height"))
            .and_then(Value::as_i64),
        video_codec: video
            .and_then(|stream| stream.get("codec_name"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        audio_codec: audio
            .and_then(|stream| stream.get("codec_name"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
    })
}

pub fn extract_audio(source_path: &str, project_dir: &Path) -> Result<PathBuf> {
    if !command_exists("ffmpeg") {
        return Err(anyhow!("ffmpeg is not installed or not available on PATH"));
    }

    std::fs::create_dir_all(project_dir)?;
    let output_path = project_dir.join("transcription_audio.wav");

    let output = Command::new(ffmpeg_bin())
        .args(["-y", "-i", source_path, "-vn", "-ac", "1", "-ar", "16000"])
        .arg(&output_path)
        .output()
        .context("running ffmpeg audio extraction")?;

    if !output.status.success() {
        return Err(anyhow!(
            "ffmpeg audio extraction failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    Ok(output_path)
}

pub fn render_flat_clip(
    source_path: &str,
    start_sec: f64,
    end_sec: f64,
    output_path: &Path,
    drawtext_filters: Option<&str>,
    caption_overlay: Option<&Path>,
) -> Result<PathBuf> {
    if !command_exists("ffmpeg") {
        return Err(anyhow!("ffmpeg is not installed or not available on PATH"));
    }

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let probe = probe_media(source_path).ok();
    let has_video = probe.map(|p| p.has_video).unwrap_or(false);

    // Follow the speaker when face tracking is available; otherwise ffmpeg's
    // default centred offsets apply, matching the original behaviour.
    let crop_offsets = if has_video {
        match crate::facetrack::plan_crop(source_path, start_sec, end_sec) {
            Some(plan) => {
                eprintln!("facetrack: {}", plan.summary);
                format!(":x='{}':y='{}'", plan.x, plan.y)
            }
            None => String::new(),
        }
    } else {
        String::new()
    };

    let mut cmd = build_render_command(RenderSpec {
        source_path,
        start_sec,
        end_sec,
        output_path,
        drawtext_filters,
        caption_overlay,
        has_video,
        crop_offsets: &crop_offsets,
    });

    let output = cmd.output().context("running ffmpeg clip render")?;

    if !output.status.success() {
        return Err(anyhow!(
            "ffmpeg clip render failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    Ok(output_path.to_path_buf())
}

pub(crate) struct RenderSpec<'a> {
    pub source_path: &'a str,
    pub start_sec: f64,
    pub end_sec: f64,
    pub output_path: &'a Path,
    pub drawtext_filters: Option<&'a str>,
    pub caption_overlay: Option<&'a Path>,
    pub has_video: bool,
    /// Pre-computed `:x=...:y=...` crop suffix, injected so this stays pure.
    pub crop_offsets: &'a str,
}

/// Assemble the ffmpeg invocation without running it.
///
/// Split out because ffmpeg option *position* carries meaning and is easy to
/// get wrong: any option placed before an `-i` binds to that input. `-t` in
/// particular must come after every input, or the output length limit is
/// silently reinterpreted as an input limit and the whole source is encoded.
pub(crate) fn build_render_command(spec: RenderSpec<'_>) -> Command {
    let start = format!("{:.3}", spec.start_sec);
    let duration = format!("{:.3}", (spec.end_sec - spec.start_sec).max(0.1));

    let mut cmd = Command::new(ffmpeg_bin());

    // --- Inputs. Nothing output-related may appear in this section. ---
    cmd.args(["-y", "-ss", &start, "-i", spec.source_path]);

    let overlay_input = spec.caption_overlay.filter(|_| spec.has_video);
    if let Some(list) = overlay_input {
        cmd.args(["-f", "concat", "-safe", "0", "-i"]);
        cmd.arg(list);
    }

    // --- Output options only, from here down. ---
    cmd.args(["-t", &duration]);

    if spec.has_video {
        let mut filter = format!(
            "crop=w='2*trunc(min(iw,ih*9/16)/2)':h='2*trunc(min(ih,iw*16/9)/2)'{}",
            spec.crop_offsets
        );

        // drawtext is a fallback only: used when no pre-rendered overlay was
        // supplied and this ffmpeg actually has the filter.
        if overlay_input.is_none() {
            if let Some(drawtext) = spec.drawtext_filters {
                if !drawtext.is_empty() {
                    if supports_captions() {
                        filter = format!("{},{}", filter, drawtext);
                    } else {
                        eprintln!(
                            "captions skipped: this ffmpeg build has no 'drawtext' filter \
                             (needs libfreetype) and no overlay renderer is available."
                        );
                    }
                }
            }
        }

        if overlay_input.is_some() {
            // Two inputs require filter_complex. `shortest=0` keeps the clip's
            // own length authoritative rather than the caption track's.
            let graph = format!(
                "[0:v]{}[base];[base][1:v]overlay=0:0:format=auto:shortest=0[v]",
                filter
            );
            cmd.args(["-filter_complex", &graph]);
            cmd.args(["-map", "[v]", "-map", "0:a?"]);
        } else {
            cmd.args(["-vf", &filter]);
        }
        cmd.args(["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p"]);
    } else {
        cmd.arg("-vn");
    }

    cmd.args(["-c:a", "aac", "-b:a", "192k"]);
    cmd.arg(spec.output_path);
    cmd
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args_of(overlay: Option<&Path>) -> Vec<String> {
        let cmd = build_render_command(RenderSpec {
            source_path: "/tmp/source.mp4",
            start_sec: 60.0,
            end_sec: 150.0,
            output_path: Path::new("/tmp/out.mp4"),
            drawtext_filters: None,
            caption_overlay: overlay,
            has_video: true,
            crop_offsets: "",
        });
        cmd.get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect()
    }

    /// Regression: `-t` once sat before the caption input, so ffmpeg treated it
    /// as an input limit and encoded the entire 49-minute source instead of a
    /// 90-second clip.
    #[test]
    fn duration_limit_comes_after_every_input() {
        for overlay in [None, Some(Path::new("/tmp/captions.txt"))] {
            let args = args_of(overlay);
            let t = args.iter().position(|a| a == "-t").expect("-t missing");
            let last_input = args
                .iter()
                .enumerate()
                .filter(|(_, a)| a.as_str() == "-i")
                .map(|(i, _)| i)
                .next_back()
                .expect("-i missing");
            assert!(
                t > last_input,
                "-t at {t} must follow the last -i at {last_input}: {args:?}"
            );
        }
    }

    #[test]
    fn clip_duration_is_the_requested_span() {
        let args = args_of(None);
        let t = args.iter().position(|a| a == "-t").unwrap();
        assert_eq!(args[t + 1], "90.000");
    }

    #[test]
    fn caption_overlay_adds_a_second_input_and_maps_it() {
        let args = args_of(Some(Path::new("/tmp/captions.txt")));
        assert_eq!(args.iter().filter(|a| a.as_str() == "-i").count(), 2);
        assert!(args.iter().any(|a| a == "-filter_complex"));
        assert!(args.iter().any(|a| a.contains("overlay=0:0")));
        // Audio must still come from the source, not the caption track.
        assert!(args.iter().any(|a| a == "0:a?"));
    }

    #[test]
    fn without_captions_there_is_one_input_and_no_filter_complex() {
        let args = args_of(None);
        assert_eq!(args.iter().filter(|a| a.as_str() == "-i").count(), 1);
        assert!(!args.iter().any(|a| a == "-filter_complex"));
        assert!(args.iter().any(|a| a == "-vf"));
    }
}
