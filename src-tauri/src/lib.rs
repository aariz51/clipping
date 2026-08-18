mod broll;
mod captions;
mod db;
mod facetrack;
mod llm;
mod media;
mod models;
mod outro;
mod postiz;
mod progress;
mod sfx;
mod title;
mod pyenv;
mod transcription;

use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use tauri::{Emitter, Manager};

use db::Database;
use models::{
    Candidate, EnvironmentStatus, MediaProbe, NormalizedTranscript, Project, ProjectDetail,
    Transcript, TranscriptWord,
};

#[derive(Clone, serde::Serialize)]
struct PullProgressPayload {
    status: String,
    completed: Option<u64>,
    total: Option<u64>,
    percentage: Option<f64>,
}

#[derive(Clone)]
struct AppState {
    db: Database,
    data_dir: PathBuf,
}

#[tauri::command]
async fn environment_status(state: tauri::State<'_, AppState>) -> Result<EnvironmentStatus, String> {
    let llm_provider = std::env::var("LLM_PROVIDER")
        .unwrap_or_else(|_| "deepseek".to_string())
        .to_lowercase();

    let has_local_whisper_model = transcription::whisper_cli_exists() || transcription::whisper_python_exists();

    let has_ollama = reqwest::Client::new()
        .get("http://localhost:11434")
        .timeout(std::time::Duration::from_millis(1000))
        .send()
        .await
        .is_ok();

    Ok(EnvironmentStatus {
        data_dir: state.data_dir.to_string_lossy().to_string(),
        has_ffmpeg: media::command_exists("ffmpeg"),
        has_ffprobe: media::command_exists("ffprobe"),
        has_deepgram_key: std::env::var("DEEPGRAM_API_KEY").is_ok(),
        has_anthropic_key: anthropic_credential().is_some(),
        has_deepseek_key: std::env::var("DEEPSEEK_API_KEY").is_ok(),
        has_gemini_key: std::env::var("GEMINI_API_KEY").is_ok(),
        has_openai_key: std::env::var("OPENAI_API_KEY").is_ok(),
        has_groq_key: std::env::var("GROQ_API_KEY").is_ok(),
        llm_provider,
        has_local_whisper_model,
        has_ollama,
        has_ytdlp: media::command_exists("yt-dlp"),
        has_face_tracking: facetrack::available(),
        // Either renderer can burn in captions: ffmpeg's drawtext, or the
        // Pillow overlay that works on builds lacking it.
        has_caption_support: media::supports_captions() || captions::available(),
        has_outro: outro::available(),
        has_voice_clone: outro::can_clone_voice(),
    })
}

#[tauri::command]
async fn pull_ollama_model(
    app: tauri::AppHandle,
    model_name: String,
) -> Result<(), String> {
    let client = reqwest::Client::new();
    
    let mut response = client
        .post("http://localhost:11434/api/pull")
        .json(&serde_json::json!({
            "name": model_name,
            "stream": true,
        }))
        .send()
        .await
        .map_err(|e| format!("Failed to connect to Ollama: {e}"))?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        return Err(format!("Ollama pull returned status {status}: {text}"));
    }

    let mut buffer = String::new();
    while let Some(chunk) = response.chunk().await.map_err(|e| e.to_string())? {
        let chunk_str = String::from_utf8_lossy(&chunk);
        buffer.push_str(&chunk_str);

        // Process lines in buffer
        while let Some(pos) = buffer.find('\n') {
            let line = buffer[..pos].trim().to_string();
            buffer = buffer[pos + 1..].to_string();

            if line.is_empty() {
                continue;
            }

            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) {
                if let Some(err_msg) = val.get("error").and_then(|v| v.as_str()) {
                    return Err(err_msg.to_string());
                }

                let completed = val.get("completed").and_then(|v| v.as_u64());
                let total = val.get("total").and_then(|v| v.as_u64());

                let mut status = val.get("status")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Downloading...")
                    .to_string();

                if status.starts_with("downloading ") {
                    if let (Some(c), Some(t)) = (completed, total) {
                        let c_mb = c as f64 / 1024.0 / 1024.0;
                        let t_mb = t as f64 / 1024.0 / 1024.0;
                        if t_mb > 100.0 {
                            status = format!("Downloading weights: {:.1} MB / {:.1} MB", c_mb, t_mb);
                        } else {
                            status = format!("Downloading model components: {:.1} MB / {:.1} MB", c_mb, t_mb);
                        }
                    } else {
                        status = "Downloading model components...".to_string();
                    }
                }
                
                let percentage = if let (Some(c), Some(t)) = (completed, total) {
                    if t > 0 {
                        Some((c as f64 / t as f64) * 100.0)
                    } else {
                        None
                    }
                } else {
                    None
                };

                let payload = PullProgressPayload {
                    status,
                    completed,
                    total,
                    percentage,
                };

                let _ = app.emit("ollama-pull-progress", payload);
            }
        }
    }

    Ok(())
}

#[tauri::command]
async fn install_ollama(app: tauri::AppHandle) -> Result<(), String> {
    let _ = app.emit("ollama-install-status", "Checking if Ollama is already installed...");
    let launch = std::process::Command::new("open")
        .args(["-a", "Ollama"])
        .output();
    
    if let Ok(out) = launch {
        if out.status.success() {
            let _ = app.emit("ollama-install-status", "Ollama is installed. Launching...");
            for _ in 0..12 {
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                if reqwest::Client::new().get("http://localhost:11434").send().await.is_ok() {
                    let _ = app.emit("ollama-install-status", "Ollama started successfully!");
                    return Ok(());
                }
            }
        }
    }

    let brew_path = if std::path::Path::new("/opt/homebrew/bin/brew").exists() {
        Some("/opt/homebrew/bin/brew")
    } else if std::path::Path::new("/usr/local/bin/brew").exists() {
        Some("/usr/local/bin/brew")
    } else {
        None
    };

    if let Some(path) = brew_path {
        let _ = app.emit("ollama-install-status", "Installing Ollama via Homebrew Cask...");
        
        let output = std::process::Command::new(path)
            .args(["install", "--cask", "ollama"])
            .output()
            .map_err(|e| format!("Failed to run brew command: {e}"))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            if !stderr.contains("already installed") {
                return Err(format!("Brew install failed: {}", stderr));
            }
        }

        let _ = app.emit("ollama-install-status", "Starting Ollama.app...");
        let launch = std::process::Command::new("open")
            .args(["-a", "Ollama"])
            .output();
        
        if let Ok(out) = launch {
            if out.status.success() {
                for _ in 0..12 {
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                    if reqwest::Client::new().get("http://localhost:11434").send().await.is_ok() {
                        let _ = app.emit("ollama-install-status", "Ollama started successfully!");
                        return Ok(());
                    }
                }
            }
        }
    }

    let _ = app.emit("ollama-install-status", "Downloading Ollama zip from official source...");
    let temp_dir = std::env::temp_dir();
    let zip_path = temp_dir.join("Ollama-darwin.zip");
    
    let response = reqwest::get("https://ollama.com/download/Ollama-darwin.zip")
        .await
        .map_err(|e| format!("Failed to download Ollama: {e}"))?;

    let bytes = response.bytes().await.map_err(|e| format!("Failed to read Ollama bytes: {e}"))?;
    std::fs::write(&zip_path, bytes).map_err(|e| format!("Failed to save Ollama zip: {e}"))?;

    let _ = app.emit("ollama-install-status", "Unzipping Ollama package...");
    let unzip_output = std::process::Command::new("unzip")
        .args(["-o", &zip_path.to_string_lossy().to_string(), "-d", &temp_dir.to_string_lossy().to_string()])
        .output()
        .map_err(|e| format!("Failed to unzip Ollama: {e}"))?;

    if !unzip_output.status.success() {
        return Err(format!("Failed to unzip: {}", String::from_utf8_lossy(&unzip_output.stderr)));
    }

    let _ = app.emit("ollama-install-status", "Installing to Applications folder...");
    let app_src = temp_dir.join("Ollama.app");
    
    let mv_output = std::process::Command::new("mv")
        .args([&app_src.to_string_lossy().to_string(), "/Applications/"])
        .output()
        .map_err(|e| format!("Failed to move Ollama to Applications: {e}"))?;

    if !mv_output.status.success() {
        let user_apps = dirs::home_dir()
            .ok_or_else(|| "Could not find home directory".to_string())?
            .join("Applications");
        std::fs::create_dir_all(&user_apps).map_err(|e| format!("Failed to create ~/Applications: {e}"))?;
        
        let mv_user_output = std::process::Command::new("mv")
            .args([&app_src.to_string_lossy().to_string(), &user_apps.to_string_lossy().to_string()])
            .output()
            .map_err(|e| format!("Failed to move Ollama to ~/Applications: {e}"))?;

        if !mv_user_output.status.success() {
            return Err(format!("Failed to install Ollama to Applications folder: {}", String::from_utf8_lossy(&mv_user_output.stderr)));
        }
    }

    let _ = app.emit("ollama-install-status", "Starting Ollama...");
    let launch = std::process::Command::new("open")
        .args(["-a", "Ollama"])
        .output();

    if launch.is_ok() {
        for _ in 0..12 {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            if reqwest::Client::new().get("http://localhost:11434").send().await.is_ok() {
                let _ = app.emit("ollama-install-status", "Ollama started successfully!");
                return Ok(());
            }
        }
    }

    Err("Ollama installed but could not be automatically started. Please open Ollama from your Applications folder.".to_string())
}

#[tauri::command]
fn create_project_from_path(
    state: tauri::State<'_, AppState>,
    path: String,
    transcription_mode: String,
    caption_style: String,
    brand_name: Option<String>,
    brand_logo_path: Option<String>,
) -> Result<Project, String> {
    validate_media_extension(&path).map_err(to_command_error)?;
    let probe = media::probe_media(&path).ok();

    state
        .db
        .create_project(
            &path,
            &transcription_mode,
            &caption_style,
            probe.and_then(|probe| probe.duration_sec),
            brand_name.as_deref(),
            brand_logo_path.as_deref(),
        )
        .map_err(to_command_error)
}

#[tauri::command]
fn list_projects(state: tauri::State<'_, AppState>) -> Result<Vec<Project>, String> {
    state.db.list_projects().map_err(to_command_error)
}

#[tauri::command]
fn get_project_detail(
    state: tauri::State<'_, AppState>,
    project_id: String,
) -> Result<ProjectDetail, String> {
    state
        .db
        .project_detail(&project_id)
        .map_err(to_command_error)
}

#[tauri::command]
fn probe_project(
    state: tauri::State<'_, AppState>,
    project_id: String,
) -> Result<MediaProbe, String> {
    let project = state
        .db
        .get_project(&project_id)
        .map_err(to_command_error)?;
    let probe = media::probe_media(&project.source_path).map_err(to_command_error)?;
    state
        .db
        .update_project_status(&project_id, "ingest", probe.duration_sec)
        .map_err(to_command_error)?;
    Ok(probe)
}

#[tauri::command]
fn extract_project_audio(
    state: tauri::State<'_, AppState>,
    project_id: String,
) -> Result<String, String> {
    let project = state
        .db
        .get_project(&project_id)
        .map_err(to_command_error)?;
    let audio_path = media::extract_audio(&project.source_path, &project_dir(&state, &project_id))
        .map_err(to_command_error)?;
    Ok(audio_path.to_string_lossy().to_string())
}

#[tauri::command]
async fn transcribe_project(
    state: tauri::State<'_, AppState>,
    project_id: String,
    provider: String,
    api_key: Option<String>,
) -> Result<Transcript, String> {
    let db = state.db.clone();
    let data_dir = state.data_dir.clone();
    let project = db.get_project(&project_id).map_err(to_command_error)?;
    db.update_project_status(&project_id, "transcribing", None)
        .map_err(to_command_error)?;

    let transcript = match provider.as_str() {
        "deepgram" => {
            let key = api_key
                .or_else(|| std::env::var("DEEPGRAM_API_KEY").ok())
                .ok_or_else(|| {
                    "Set DEEPGRAM_API_KEY or paste an API key to use cloud transcription."
                        .to_string()
                })?;
            let audio_path = media::extract_audio(
                &project.source_path,
                &data_dir.join("projects").join(&project_id),
            )
            .map_err(to_command_error)?;
            transcription::transcribe_deepgram(&audio_path.to_string_lossy(), &key)
                .await
                .map_err(to_command_error)?
        }
        "local" => {
            let has_whisper = transcription::whisper_cli_exists() || transcription::whisper_python_exists();
            if !has_whisper {
                return Err("Whisper is not installed. Please install it (e.g., via Homebrew 'brew install whisper-cli' or via Python 'pip3 install openai-whisper').".to_string());
            }
            let audio_path = media::extract_audio(
                &project.source_path,
                &data_dir.join("projects").join(&project_id),
            )
            .map_err(to_command_error)?;
            transcription::transcribe_local(&audio_path.to_string_lossy(), &data_dir.to_string_lossy())
                .await
                .map_err(to_command_error)?
        }
        other => return Err(format!("Unsupported transcription provider: {other}")),
    };

    let raw_json = serde_json::to_string_pretty(&transcript).map_err(to_command_error)?;
    let saved = db
        .save_transcript(
            &project_id,
            &provider,
            &raw_json,
            Some(&transcript.language),
        )
        .map_err(to_command_error)?;
    db.update_project_status(&project_id, "analyzing", Some(transcript.duration))
        .map_err(to_command_error)?;
    Ok(saved)
}

#[tauri::command]
fn save_demo_transcript(
    state: tauri::State<'_, AppState>,
    project_id: String,
) -> Result<Transcript, String> {
    let transcript = demo_transcript();
    let raw_json = serde_json::to_string_pretty(&transcript).map_err(to_command_error)?;
    let saved = state
        .db
        .save_transcript(&project_id, "demo", &raw_json, Some(&transcript.language))
        .map_err(to_command_error)?;
    state
        .db
        .update_project_status(&project_id, "analyzing", Some(transcript.duration))
        .map_err(to_command_error)?;
    Ok(saved)
}

/// The Anthropic credential, in either shape the API accepts.
///
/// A console key (`sk-ant-api...`) and a Claude Code subscription token
/// (`sk-ant-oat...`) authenticate differently but are both valid credentials,
/// so either one counts as "Claude is configured". Looking only at
/// `ANTHROPIC_API_KEY` made a perfectly good OAuth token look like no
/// credential at all.
fn anthropic_credential() -> Option<String> {
    ["ANTHROPIC_API_KEY", "ANTHROPIC_OAUTH_TOKEN"]
        .into_iter()
        .find_map(|name| std::env::var(name).ok())
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

#[tauri::command]
async fn generate_candidates(
    state: tauri::State<'_, AppState>,
    project_id: String,
    api_key: Option<String>,
    provider: Option<String>,
    model_name: Option<String>,
    _allow_demo: bool,
) -> Result<Vec<Candidate>, String> {
    let db = state.db.clone();
    let transcript = db
        .latest_transcript(&project_id)
        .map_err(to_command_error)?
        .ok_or_else(|| "Transcribe the project before detecting moments.".to_string())?;
    let normalized: NormalizedTranscript =
        serde_json::from_str(&transcript.raw_json).map_err(to_command_error)?;

    let active_provider = provider
        .or_else(|| std::env::var("LLM_PROVIDER").ok())
        .unwrap_or_else(|| "claude".to_string())
        .to_lowercase();

    let drafts = match active_provider.as_str() {
        "claude" => {
            let key = api_key
                .filter(|k| !k.trim().is_empty())
                .or_else(anthropic_credential)
                .ok_or_else(|| "Set ANTHROPIC_API_KEY or ANTHROPIC_OAUTH_TOKEN, or supply a Claude key, to generate candidates.".to_string())?;
            llm::detect_candidates_with_claude(&normalized, &key)
                .await
                .map_err(to_command_error)?
        }
        "local" | "ollama" => {
            let model = model_name
                .or_else(|| std::env::var("OLLAMA_MODEL").ok())
                .unwrap_or_else(|| "llama3.2".to_string());
            llm::detect_candidates_with_local_llm(&normalized, &model)
                .await
                .map_err(to_command_error)?
        }
        "gemini" => {
            let key = api_key
                .or_else(|| std::env::var("GEMINI_API_KEY").ok())
                .ok_or_else(|| "Set GEMINI_API_KEY or supply Gemini API Key to generate candidates.".to_string())?;
            llm::detect_candidates_with_gemini(&normalized, &key)
                .await
                .map_err(to_command_error)?
        }
        "openai" => {
            let key = api_key
                .or_else(|| std::env::var("OPENAI_API_KEY").ok())
                .ok_or_else(|| "Set OPENAI_API_KEY or supply OpenAI API Key to generate candidates.".to_string())?;
            llm::detect_candidates_with_openai(&normalized, &key)
                .await
                .map_err(to_command_error)?
        }
        "groq" => {
            let key = api_key
                .or_else(|| std::env::var("GROQ_API_KEY").ok())
                .ok_or_else(|| "Set GROQ_API_KEY or supply Groq API Key to generate candidates.".to_string())?;
            llm::detect_candidates_with_groq(&normalized, &key)
                .await
                .map_err(to_command_error)?
        }
        _ => {
            let key = api_key
                .or_else(|| std::env::var("DEEPSEEK_API_KEY").ok())
                .ok_or_else(|| "Set DEEPSEEK_API_KEY or supply DeepSeek API Key to generate candidates.".to_string())?;
            llm::detect_candidates_with_deepseek(&normalized, &key, model_name.as_deref())
                .await
                .map_err(to_command_error)?
        }
    };

    if drafts.is_empty() {
        return Err("No viable clip candidates were returned for this transcript.".to_string());
    }

    let candidates = db
        .replace_candidates(&project_id, &drafts)
        .map_err(to_command_error)?;
    db.update_project_status(&project_id, "ready", None)
        .map_err(to_command_error)?;
    Ok(candidates)
}

#[tauri::command]
fn set_selected_clip_count(
    state: tauri::State<'_, AppState>,
    project_id: String,
    count: usize,
) -> Result<Vec<Candidate>, String> {
    state
        .db
        .set_selected_clip_count(&project_id, count.clamp(0, 10))
        .map_err(to_command_error)
}

#[tauri::command]
async fn add_broll_to_clip(
    state: tauri::State<'_, AppState>,
    candidate_id: String,
) -> Result<String, String> {
    let db = state.db.clone();
    let candidate_id_for_task = candidate_id.clone();

    let enriched = tokio::task::spawn_blocking(move || -> Result<(PathBuf, Candidate, Vec<TranscriptWord>), String> {
        let candidate_id = candidate_id_for_task;
        let (candidate, project) = db
            .get_candidate_with_project(&candidate_id)
            .map_err(to_command_error)?;

        // B-roll enriches an already-rendered clip, so require the cut first.
        let detail = db.project_detail(&project.id).map_err(to_command_error)?;
        let clip = detail
            .clips
            .iter()
            .find(|c| c.candidate_id == candidate_id)
            .ok_or_else(|| "cut this clip before adding B-roll".to_string())?;
        let source = clip
            .output_path
            .as_deref()
            .ok_or_else(|| "clip has no rendered file yet".to_string())?;
        let source = PathBuf::from(source);
        if !source.exists() {
            return Err(format!("rendered clip is missing: {}", source.display()));
        }

        let words = db
            .latest_transcript(&project.id)
            .ok()
            .flatten()
            .and_then(|t| serde_json::from_str::<NormalizedTranscript>(&t.raw_json).ok())
            .map(|n| n.words)
            .unwrap_or_default();

        let output = source.with_file_name(format!(
            "{}_broll.mp4",
            source.file_stem().unwrap_or_default().to_string_lossy()
        ));

        // The hook describes what the clip is about, which is what steers scene
        // selection; fall back to the project name when it is blank.
        let topic = if candidate.hook.trim().is_empty() {
            project.name.clone().unwrap_or_else(|| "a short social video".to_string())
        } else {
            candidate.hook.clone()
        };

        let produced = broll::enrich(
            &source,
            &words,
            candidate.start_sec,
            candidate.end_sec,
            &topic,
            &output,
        )?;
        Ok((produced, candidate, words))
    })
    .await
    .map_err(|e| e.to_string())??;

    let (produced, candidate, words) = enriched;

    // A persistent headline across the top, so a scroller knows the subject
    // before hearing a word. Applied after B-roll so it sits over every scene,
    // not just the speaker ones.
    let spoken: String = words
        .iter()
        .filter(|w| w.end > candidate.start_sec && w.start < candidate.end_sec)
        .map(|w| w.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let headline = title::write_title(&spoken, &candidate.hook).await;

    let db2 = state.db.clone();
    let cid = candidate_id.clone();
    tokio::task::spawn_blocking(move || {
        let titled_out = produced.with_file_name(format!(
            "{}_titled.mp4",
            produced.file_stem().unwrap_or_default().to_string_lossy()
        ));
        // A failure here must not lose the B-roll render, so fall back to the
        // untitled file rather than erroring the whole action.
        let final_path = match title::apply(&produced, &headline, &titled_out) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[title] skipped: {e}");
                produced.clone()
            }
        };

        // Sound effects, chosen from the edit. The video stream is copied, so
        // this adds atmosphere without touching picture quality.
        let plan = produced
            .parent()
            .map(|d| d.join(format!(
                "edit_{}",
                produced.file_stem().unwrap_or_default().to_string_lossy()
                    .replace("_broll", ""))).join("scene_plan.json"));
        let tx = produced.parent().map(|d| d.join("broll_transcript.json"));
        let sfx_out = final_path.with_file_name(format!(
            "{}_sfx.mp4",
            final_path.file_stem().unwrap_or_default().to_string_lossy()));
        let final_path = match sfx::apply(
            &final_path, plan.as_deref(), tx.as_deref(), &sfx_out) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[sfx] skipped: {e}");
                final_path
            }
        };

        let final_path = final_path.to_string_lossy().to_string();
        db2.set_broll_path(&cid, &final_path)
            .map_err(to_command_error)?;
        Ok(final_path)
    })
    .await
    .map_err(|e| e.to_string())?
}

/// True when the project carries real branding the user actually supplied.
///
/// An outro is only wanted when there is something to advertise. Checking
/// merely that a name exists let a placeholder through -- a project created by
/// the end-to-end test defaulted to "My App" and every clip ended with "My App
/// - Download on the App Store & Google Play", which is worse than no outro.
/// Both a real name and a logo are required, so an unbranded project ends on
/// its own last frame.
pub(crate) fn has_real_branding(project: &Project) -> bool {
    const PLACEHOLDERS: [&str; 4] = ["my app", "app name", "your app", "test"];
    let named = project
        .brand_name
        .as_deref()
        .map(str::trim)
        .filter(|n| !n.is_empty())
        .map(|n| !PLACEHOLDERS.contains(&n.to_lowercase().as_str()))
        .unwrap_or(false);
    let logo = project
        .brand_logo_path
        .as_deref()
        .map(|p| !p.trim().is_empty() && std::path::Path::new(p).exists())
        .unwrap_or(false);
    named && logo
}

/// Whether the *automated* pipeline should append an end card.
///
/// It should not. An end card is an advert, and whether a clip carries one is
/// an editorial decision the user makes per clip with the "Add Outro" button --
/// not something a batch decides on their behalf. This stays separate from
/// `has_real_branding` so the branding check keeps its plain meaning.
/// `OUTRO_IN_BATCH=1` opts the batch back in.
pub(crate) fn batch_should_add_outro(project: &Project) -> bool {
    std::env::var("OUTRO_IN_BATCH").ok().as_deref() == Some("1") && has_real_branding(project)
}

/// Appending the branded end card, independent of Tauri state so the same code
/// the button calls can be exercised directly in tests.
pub(crate) fn add_outro_blocking(
    db: Database,
    candidate_id: String,
    video_path: String,
) -> Result<String, String> {
        let (candidate, project) = db
            .get_candidate_with_project(&candidate_id)
            .map_err(to_command_error)?;

        let app_name = project
            .brand_name
            .clone()
            .filter(|n| !n.trim().is_empty())
            .ok_or_else(|| "no app name set for this project".to_string())?;

        let source = PathBuf::from(&video_path);
        if !source.exists() {
            return Err(format!("video not found: {video_path}"));
        }

        // The outro is appended last, so whichever artefact the user is about to
        // publish -- the plain cut or the B-roll version -- gets the end card.
        let output = source.with_file_name(format!(
            "{}_final.mp4",
            source.file_stem().unwrap_or_default().to_string_lossy()
        ));

        // Word timings let the voice picker separate speakers; without them the
        // middle of the clip is sampled instead.
        let transcript = db
            .latest_transcript(&project.id)
            .ok()
            .flatten()
            .and_then(|t| serde_json::from_str::<NormalizedTranscript>(&t.raw_json).ok())
            .and_then(|n| {
                let words: Vec<_> = n
                    .words
                    .iter()
                    .filter(|w| w.end > candidate.start_sec && w.start < candidate.end_sec)
                    .map(|w| {
                        serde_json::json!({
                            "text": w.text,
                            "start": w.start - candidate.start_sec,
                            "end": w.end - candidate.start_sec,
                            "speaker": w.speaker,
                        })
                    })
                    .collect();
                if words.is_empty() {
                    return None;
                }
                let path = source.with_extension("outro_words.json");
                std::fs::write(&path, serde_json::json!({ "words": words }).to_string()).ok()?;
                Some(path)
            });

        let logo = project.brand_logo_path.as_ref().map(PathBuf::from);
        let produced = outro::append(
            &source,
            &app_name,
            logo.as_deref(),
            transcript.as_deref(),
            &output,
        )?;
        Ok(produced.to_string_lossy().to_string())
    }

#[tauri::command]
async fn add_outro_to_clip(
    state: tauri::State<'_, AppState>,
    candidate_id: String,
    video_path: String,
) -> Result<String, String> {
    let db = state.db.clone();
    tokio::task::spawn_blocking(move || add_outro_blocking(db, candidate_id, video_path))
        .await
        .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn batch_progress() -> Result<progress::BatchProgress, String> {
    tokio::task::spawn_blocking(progress::read)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn postiz_channels() -> Result<Vec<postiz::Channel>, String> {
    tokio::task::spawn_blocking(postiz::channels)
        .await
        .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn publish_clip_to_postiz(
    state: tauri::State<'_, AppState>,
    video_path: String,
    caption: String,
    channel_ids: Vec<String>,
    schedule_at: Option<String>,
    dry_run: bool,
    publish_now: Option<bool>,
    candidate_id: Option<String>,
) -> Result<String, String> {
    let db = state.db.clone();
    tokio::task::spawn_blocking(move || {
        let path = PathBuf::from(&video_path);
        if !path.exists() {
            return Err(format!("video not found: {video_path}"));
        }
        let published = publish_now.unwrap_or(false);
        let out = postiz::publish(
            &path,
            &caption,
            &channel_ids,
            schedule_at.as_deref(),
            dry_run,
            published,
        )?;

        // Record the outcome so the card can show it. A dry run changes
        // nothing in Postiz, so it must not mark the clip either.
        if !dry_run {
            if let Some(cid) = candidate_id {
                let stamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs().to_string())
                    .unwrap_or_default();
                let label = if published { "posted" } else { "draft" };
                if let Err(e) = db.set_postiz_state(&cid, label, &stamp) {
                    eprintln!("[postiz] could not record state: {e}");
                }
            }
        }
        Ok(out)
    })
    .await
    .map_err(|e| e.to_string())?
}

/// The clip-cutting pipeline, independent of Tauri state so it can be exercised
/// directly in tests as well as from the command.
pub(crate) fn cut_candidate_blocking(
    db: Database,
    data_dir: PathBuf,
    candidate_id: String,
) -> Result<String, String> {
        let (candidate, project) = db
            .get_candidate_with_project(&candidate_id)
            .map_err(to_command_error)?;
        db
            .update_clip_for_candidate(&candidate_id, "cutting", None, None, None)
            .map_err(to_command_error)?;

        let output_path = documents_project_dir(&project)?
            .join("clips")
            .join(format!("clip-{:02}_flat.mp4", candidate.rank));

        let mut srt_path = None;
        let mut drawtext_filters = None;
        let mut caption_track = None;

        let probe = media::probe_media(&project.source_path).ok();
        let (cropped_width, cropped_height) = if let Some(p) = &probe {
            let iw = p.width.unwrap_or(1920) as f64;
            let ih = p.height.unwrap_or(1080) as f64;
            let w = (iw.min(ih * 9.0 / 16.0) / 2.0).floor() * 2.0;
            let h = (ih.min(iw * 16.0 / 9.0) / 2.0).floor() * 2.0;
            (w as i64, h as i64)
        } else {
            (1080, 1920)
        };

        if let Ok(Some(transcript_record)) = db.latest_transcript(&project.id) {
            if let Ok(normalized) = serde_json::from_str::<NormalizedTranscript>(&transcript_record.raw_json) {
                let srt_content = generate_srt(&normalized.words, candidate.start_sec, candidate.end_sec);
                let clip_srt_path = data_dir.join("projects").join(&project.id).join(format!("clip-{}.srt", candidate.id));
                if std::fs::write(&clip_srt_path, srt_content).is_ok() {
                    srt_path = Some(clip_srt_path);
                }
                let style = project.caption_style.as_deref().unwrap_or("modern-box");

                // Preferred path: rasterised overlay. It works on any ffmpeg
                // build and preserves punctuation that drawtext strips.
                caption_track = captions::render_track(
                    &normalized.words,
                    candidate.start_sec,
                    candidate.end_sec,
                    cropped_width,
                    cropped_height,
                    style,
                    &std::env::temp_dir(),
                );

                if caption_track.is_none() {
                    let drawtext = build_drawtext_filters(
                        &normalized.words,
                        candidate.start_sec,
                        candidate.end_sec,
                        cropped_width,
                        style,
                    );
                    if !drawtext.is_empty() {
                        drawtext_filters = Some(drawtext);
                    }
                }
            }
        }

        match media::render_flat_clip(
            &project.source_path,
            candidate.start_sec,
            candidate.end_sec,
            &output_path,
            drawtext_filters.as_deref(),
            caption_track.as_ref().map(|t| t.concat_list.as_path()),
        ) {
            Ok(path) => {
                let path_string = path.to_string_lossy().to_string();
                let srt_string = srt_path.map(|p| p.to_string_lossy().to_string());
                db
                    .update_clip_for_candidate(
                        &candidate_id,
                        "done",
                        Some(&path_string),
                        srt_string.as_deref(),
                        None,
                    )
                    .map_err(to_command_error)?;
                Ok(path_string)
            }
            Err(error) => {
                let err_msg = error.to_string();
                // Fallback retry rendering without captions overlay on any error
                match media::render_flat_clip(
                    &project.source_path,
                    candidate.start_sec,
                    candidate.end_sec,
                    &output_path,
                    None,
                    None,
                ) {
                    Ok(path) => {
                        let path_string = path.to_string_lossy().to_string();
                        let srt_string = srt_path.map(|p| p.to_string_lossy().to_string());
                        let warning_msg = format!(
                            "Clip rendered successfully, but captions were skipped. Error: {}",
                            err_msg
                        );
                        db
                            .update_clip_for_candidate(
                                &candidate_id,
                                "done",
                                Some(&path_string),
                                srt_string.as_deref(),
                                Some(&warning_msg),
                            )
                            .map_err(to_command_error)?;
                        Ok(path_string)
                    }
                    Err(retry_err) => {
                        let message = retry_err.to_string();
                        db
                            .update_clip_for_candidate(&candidate_id, "error", None, None, Some(&message))
                            .map_err(to_command_error)?;
                        Err(message)
                    }
                }
            }
        }
    }

#[tauri::command]
async fn render_flat_clip_for_candidate(
    state: tauri::State<'_, AppState>,
    candidate_id: String,
) -> Result<String, String> {
    let db = state.db.clone();
    let data_dir = state.data_dir.clone();
    tokio::task::spawn_blocking(move || cut_candidate_blocking(db, data_dir, candidate_id))
        .await
        .map_err(|e| e.to_string())?
}

#[tauri::command]
fn delete_project(state: tauri::State<'_, AppState>, project_id: String) -> Result<(), String> {
    state.db.delete_project(&project_id).map_err(to_command_error)
}

#[tauri::command]
fn rename_project(
    state: tauri::State<'_, AppState>,
    project_id: String,
    name: String,
) -> Result<(), String> {
    state.db.rename_project(&project_id, &name).map_err(to_command_error)
}

/// Load `.env` from wherever it actually is.
///
/// `dotenvy::dotenv()` walks up from the *working directory*, which is fine
/// under `tauri dev` (launched from `src-tauri/`) but useless for a bundled
/// `.app`: Finder starts it with the working directory at `/`, so the file is
/// never found and every API credential silently goes missing. These locations
/// cover both, plus an explicit override.
fn load_env() {
    if let Ok(explicit) = std::env::var("AUTOSHORTS_ENV") {
        if dotenvy::from_path(&explicit).is_ok() {
            eprintln!("[env] loaded {explicit}");
            return;
        }
    }
    if let Ok(path) = dotenvy::dotenv() {
        eprintln!("[env] loaded {}", path.display());
        return;
    }

    let mut tried: Vec<PathBuf> = Vec::new();
    // Beside the executable, then upwards: inside a bundle that climbs out of
    // `AutoShorts.app/Contents/MacOS` and on to the checkout it was built from.
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent().map(std::path::Path::to_path_buf);
        while let Some(d) = dir {
            tried.push(d.join(".env"));
            dir = d.parent().map(std::path::Path::to_path_buf);
        }
    }
    if let Some(home) = dirs::home_dir() {
        tried.push(home.join("autoshorts/.env"));
        tried.push(home.join(".autoshorts/.env"));
    }

    for path in tried {
        if path.is_file() && dotenvy::from_path(&path).is_ok() {
            eprintln!("[env] loaded {}", path.display());
            return;
        }
    }
    eprintln!("[env] no .env found; API credentials must come from the environment");
}

pub fn run() {
    load_env();

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let data_dir = app
                .path()
                .app_data_dir()
                .context("resolving app data directory")?;
            std::fs::create_dir_all(&data_dir).context("creating app data directory")?;
            std::fs::create_dir_all(data_dir.join("models")).context("creating models directory")?;
            let db = Database::open(&data_dir.join("autoshorts.sqlite"))?;
            app.manage(AppState { db, data_dir });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            environment_status,
            pull_ollama_model,
            install_ollama,
            create_project_from_path,
            list_projects,
            get_project_detail,
            probe_project,
            extract_project_audio,
            transcribe_project,
            save_demo_transcript,
            generate_candidates,
            set_selected_clip_count,
            render_flat_clip_for_candidate,
            delete_project,
            rename_project,
            check_youtube_copyright,
            download_youtube_video,
            add_broll_to_clip,
            add_outro_to_clip,
            postiz_channels,
            batch_progress,
            publish_clip_to_postiz
        ])
        .run(tauri::generate_context!())
        .expect("error while running AutoShorts");
}

#[derive(serde::Serialize)]
pub struct CopyrightCheckResult {
    is_safe: bool,
    license: Option<String>,
}

#[tauri::command]
async fn check_youtube_copyright(url: String) -> Result<CopyrightCheckResult, String> {
    tokio::task::spawn_blocking(move || {
        let output = std::process::Command::new("yt-dlp")
            .args(&["--dump-json", &url])
            .output()
            .map_err(|e| format!("Failed to run yt-dlp: {}", e))?;

        if !output.status.success() {
            return Err("Failed to fetch video metadata from YouTube.".to_string());
        }

        let json_str = String::from_utf8_lossy(&output.stdout);
        let parsed: serde_json::Value = serde_json::from_str(&json_str).map_err(|_| "Failed to parse yt-dlp output".to_string())?;

        let license = parsed.get("license").and_then(|v| v.as_str()).map(|s| s.to_string());
        
        let is_safe = if let Some(lic) = &license {
            lic.to_lowercase().contains("creative commons") || lic.to_lowercase().contains("reuse allowed")
        } else {
            false
        };

        Ok(CopyrightCheckResult { is_safe, license })
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn download_youtube_video(url: String) -> Result<String, String> {
    tokio::task::spawn_blocking(move || {
        let downloads_dir = dirs::download_dir()
            .ok_or_else(|| "Could not find Downloads folder".to_string())?;
        
        let output_template = downloads_dir.join("AutoShorts_%(id)s.%(ext)s");
        let output_template_str = output_template.to_string_lossy().to_string();

        let output = std::process::Command::new("yt-dlp")
            .args(&[
                "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "--merge-output-format", "mp4",
                "-o", &output_template_str,
                "--print", "after_move:filepath",
                "--no-simulate",
                &url
            ])
            .output()
            .map_err(|e| format!("Failed to run yt-dlp: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("yt-dlp failed: {}", stderr));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let filepath = stdout.lines().last().unwrap_or("").trim();

        if filepath.is_empty() || !std::path::Path::new(filepath).exists() {
            return Err("Could not locate downloaded file".to_string());
        }

        Ok(filepath.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

fn project_dir(state: &AppState, project_id: &str) -> PathBuf {
    state.data_dir.join("projects").join(project_id)
}

fn documents_project_dir(project: &Project) -> Result<PathBuf, String> {
    let documents_dir = dirs::document_dir()
        .ok_or_else(|| "Could not find your Documents folder for clip output.".to_string())?;
    let base = documents_dir.join("AutoShorts").join(project_output_slug(project));

    // The slug comes from the source filename alone, so importing the same
    // video twice would point both projects at one folder and silently
    // overwrite `clip-01_flat.mp4`. A marker records which project owns the
    // directory; later projects get their own suffixed folder.
    let marker = base.join(".project-id");
    match std::fs::read_to_string(&marker) {
        Ok(owner) if owner.trim() != project.id => {
            let short: String = project.id.chars().take(8).collect();
            let mine = base.with_file_name(format!(
                "{}-{}",
                base.file_name().unwrap_or_default().to_string_lossy(),
                short
            ));
            let _ = std::fs::create_dir_all(&mine);
            let _ = std::fs::write(mine.join(".project-id"), &project.id);
            Ok(mine)
        }
        Ok(_) => Ok(base),
        Err(_) => {
            let _ = std::fs::create_dir_all(&base);
            let _ = std::fs::write(&marker, &project.id);
            Ok(base)
        }
    }
}

fn project_output_slug(project: &Project) -> String {
    let stem = std::path::Path::new(&project.source_path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or(&project.id);
    let slug = stem
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string();

    if slug.is_empty() {
        project.id.clone()
    } else {
        slug
    }
}

fn validate_media_extension(path: &str) -> Result<()> {
    let extension = std::path::Path::new(path)
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.to_ascii_lowercase())
        .ok_or_else(|| anyhow!("Selected file does not have an extension"))?;

    let allowed = ["mp4", "mov", "mp3", "wav", "m4a"];
    if allowed.contains(&extension.as_str()) {
        Ok(())
    } else {
        Err(anyhow!(
            "Unsupported file type .{extension}. Use mp4, mov, mp3, wav, or m4a."
        ))
    }
}

fn demo_transcript() -> NormalizedTranscript {
    let lines = [
        "The surprising thing about short-form clips is that the best moment is rarely the loudest moment.",
        "It is usually the point where someone finally says the quiet part plainly and the listener can feel the stakes.",
        "That is why the system needs to understand the transcript as a story, not just search for keywords.",
        "A good clip opens with tension, resolves one idea, and ends before the energy leaks away.",
        "If you can rank those moments consistently, the rendering pipeline becomes much easier to trust.",
        "The creator still decides what represents them, but the machine removes the first exhausting pass through hours of footage.",
        "The goal is not to automate taste completely. The goal is to give taste a faster starting point.",
        "Once the strongest moments are visible, captions and platform copy become finishing work instead of discovery work.",
        "That is the workflow AutoShorts is designed around.",
    ];

    let mut words = Vec::new();
    let mut cursor = 0.0;
    for line in lines {
        for token in line.split_whitespace() {
            let clean = token.to_string();
            let end = cursor + 0.32;
            words.push(TranscriptWord {
                text: clean,
                start: cursor,
                end,
                speaker: Some("A".to_string()),
            });
            cursor = end + 0.08;
        }
        cursor += 0.75;
    }

    let segments = transcription::build_segments(&words);

    NormalizedTranscript {
        language: "en".to_string(),
        duration: cursor,
        speakers: vec!["A".to_string()],
        words,
        segments,
    }
}

fn to_command_error(error: impl std::fmt::Display) -> String {
    error.to_string()
}

fn generate_srt(words: &[TranscriptWord], start_sec: f64, end_sec: f64) -> String {
    let mut srt = String::new();
    let mut index = 1;

    let candidate_words: Vec<&TranscriptWord> = words
        .iter()
        .filter(|w| w.end > start_sec && w.start < end_sec)
        .collect();

    for chunk in candidate_words.chunks(3) {
        if chunk.is_empty() {
            continue;
        }
        let first = chunk[0];
        let last = chunk[chunk.len() - 1];

        let start_rel = (first.start - start_sec).max(0.0);
        let end_rel = (last.end - start_sec).min(end_sec - start_sec).max(0.0);

        let text = chunk
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        srt.push_str(&format!("{index}\n"));
        srt.push_str(&format!(
            "{}\n",
            format_srt_time(start_rel, end_rel)
        ));
        srt.push_str(&format!("{text}\n\n"));
        index += 1;
    }

    srt
}

fn format_srt_time(start: f64, end: f64) -> String {
    let format_time = |secs: f64| {
        let hours = (secs / 3600.0) as u32;
        let mins = ((secs % 3600.0) / 60.0) as u32;
        let secs_only = (secs % 60.0) as u32;
        let ms = ((secs.fract()) * 1000.0) as u32;
        format!("{hours:02}:{mins:02}:{secs_only:02},{ms:03}")
    };
    format!("{} --> {}", format_time(start), format_time(end))
}

fn build_drawtext_filters(
    words: &[TranscriptWord],
    start_sec: f64,
    end_sec: f64,
    cropped_width: i64,
    caption_style: &str,
) -> String {
    let candidate_words: Vec<&TranscriptWord> = words
        .iter()
        .filter(|w| w.end > start_sec && w.start < end_sec)
        .collect();

    if candidate_words.is_empty() {
        return String::new();
    }

    // Build font option once for drawtext filter
    let mut font_paths = vec![
        // macOS
        "/System/Library/Fonts/Supplemental/Futura.ttc".to_string(),
        "/System/Library/Fonts/Avenir Next.ttc".to_string(),
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf".to_string(),
        "/System/Library/Fonts/Helvetica.ttc".to_string(),
        // Windows standard
        "C:/Windows/Fonts/SegoeUIb.ttf".to_string(),
        "C:/Windows/Fonts/segoeuib.ttf".to_string(),
        "C:/Windows/Fonts/SegoeUI.ttf".to_string(),
        "C:/Windows/Fonts/segoeui.ttf".to_string(),
        "C:/Windows/Fonts/arialbd.ttf".to_string(),
        "C:/Windows/Fonts/arial.ttf".to_string(),
        // Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf".to_string(),
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf".to_string(),
    ];
    if let Ok(windir) = std::env::var("WINDIR").or_else(|_| std::env::var("SystemRoot")) {
        let windir_fonts = format!("{}/Fonts", windir.replace('\\', "/"));
        font_paths.push(format!("{}/SegoeUIb.ttf", windir_fonts));
        font_paths.push(format!("{}/segoeuib.ttf", windir_fonts));
        font_paths.push(format!("{}/SegoeUI.ttf", windir_fonts));
        font_paths.push(format!("{}/arialbd.ttf", windir_fonts));
        font_paths.push(format!("{}/arial.ttf", windir_fonts));
    }

    let mut font_option = String::new();
    for path in &font_paths {
        if std::path::Path::new(path).exists() {
            let normalized_path = path.replace('\\', "/");
            let escaped_path = normalized_path.replace('\'', "'\\''");
            font_option = format!("fontfile='{}':", escaped_path);
            break;
        }
    }

    let mut drawtext_filters = Vec::new();

    // Group into chunks of 2 words for fast-paced style captions
    for chunk in candidate_words.chunks(2) {
        if chunk.is_empty() {
            continue;
        }
        let first = chunk[0];
        let last = chunk[chunk.len() - 1];

        // Timestamps relative to clip start (due to fast input seeking resetting stream PTS)
        let start_rel = (first.start - start_sec).max(0.0);
        let end_rel = (last.end - start_sec).min(end_sec - start_sec).max(0.0);
        if end_rel <= start_rel {
            continue;
        }

        let text = chunk
            .iter()
            .map(|w| w.text.to_uppercase())
            .collect::<Vec<_>>()
            .join(" ");

        // Clean text to avoid breaking filter parameters
        let clean_text: String = text.chars()
            .filter(|c| c.is_alphanumeric() || *c == ' ' || *c == '!' || *c == '?')
            .collect();

        // Responsive font size and padding box
        let fontsize = ((cropped_width as f64) * 0.075).clamp(16.0, 80.0).round() as i64;
        let padding = ((fontsize as f64) * 0.3).clamp(4.0, 24.0).round() as i64;


        let drawtext = match caption_style {
            "classic-outline" => {
                // Classic yellow text with a bold outline (CapCut style)
                let borderw = ((fontsize as f64) * 0.1).clamp(2.0, 8.0).round() as i64;
                format!(
                    "drawtext={}text='{}':x=(w-text_w)/2:y=h*0.65:fontsize={}:fontcolor=yellow:borderw={}:bordercolor=black:enable='between(t,{:.3},{:.3})'",
                    font_option, clean_text, fontsize, borderw, start_rel, end_rel
                )
            }
            "minimal-shadow" => {
                // Sleek white text with a soft drop shadow (Minimalist)
                format!(
                    "drawtext={}text='{}':x=(w-text_w)/2:y=h*0.7:fontsize={}:fontcolor=white:shadowcolor=black@0.5:shadowx=2:shadowy=2:enable='between(t,{:.3},{:.3})'",
                    font_option, clean_text, fontsize, start_rel, end_rel
                )
            }
            "vibrant-cyan" => {
                // Modern Avenir Next look with clean cyan color and thin shadow
                format!(
                    "drawtext={}text='{}':x=(w-text_w)/2:y=h*0.7:fontsize={}:fontcolor=0x00FFFF:shadowcolor=black@0.6:shadowx=2:shadowy=2:enable='between(t,{:.3},{:.3})'",
                    font_option, clean_text, fontsize, start_rel, end_rel
                )
            }
            "vibrant-yellow-box" => {
                // Vibrant black text inside a solid yellow background box (Motivational/TikTok style)
                format!(
                    "drawtext={}text='{}':x=(w-text_w)/2:y=h*0.72:fontsize={}:fontcolor=black:box=1:boxcolor=0xffff00e0:boxborderw={}:enable='between(t,{:.3},{:.3})'",
                    font_option, clean_text, fontsize, padding, start_rel, end_rel
                )
            }
            "vibrant-green" => {
                // High-energy neon green text with outline & drop shadow (Hormozi style)
                let borderw = ((fontsize as f64) * 0.08).clamp(1.5, 6.0).round() as i64;
                format!(
                    "drawtext={}text='{}':x=(w-text_w)/2:y=h*0.7:fontsize={}:fontcolor=0x39FF14:borderw={}:bordercolor=black:shadowcolor=black@0.6:shadowx=2:shadowy=2:enable='between(t,{:.3},{:.3})'",
                    font_option, clean_text, fontsize, borderw, start_rel, end_rel
                )
            }
            "vibrant-red" => {
                // Dramatic red text with outline & drop shadow (Gaming/Drama style)
                let borderw = ((fontsize as f64) * 0.08).clamp(1.5, 6.0).round() as i64;
                format!(
                    "drawtext={}text='{}':x=(w-text_w)/2:y=h*0.7:fontsize={}:fontcolor=0xFF3B30:borderw={}:bordercolor=black:shadowcolor=black@0.6:shadowx=2:shadowy=2:enable='between(t,{:.3},{:.3})'",
                    font_option, clean_text, fontsize, borderw, start_rel, end_rel
                )
            }
            _ => {
                // modern-box (Default): white text with clean box background
                format!(
                    "drawtext={}text='{}':x=(w-text_w)/2:y=h*0.72:fontsize={}:fontcolor=white:box=1:boxcolor=0x000000b0:boxborderw={}:enable='between(t,{:.3},{:.3})'",
                    font_option, clean_text, fontsize, padding, start_rel, end_rel
                )
            }
        };
        drawtext_filters.push(drawtext);
    }

    drawtext_filters.join(",")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::TranscriptWord;

    #[test]
    fn test_build_drawtext_filters_formatting() {
        let words = vec![
            TranscriptWord {
                text: "Hello".to_string(),
                start: 0.0,
                end: 1.0,
                speaker: None,
            },
            TranscriptWord {
                text: "world".to_string(),
                start: 1.0,
                end: 2.0,
                speaker: None,
            },
        ];

        let result = build_drawtext_filters(&words, 0.0, 5.0, 1080, "classic-outline");
        assert!(!result.is_empty());
        assert!(result.contains("drawtext="));
        assert!(result.contains("text='HELLO WORLD'"));

        if result.contains("fontfile=") {
            assert!(result.contains("fontfile='"));
            assert!(!result.contains("\\:"));
        }
    }
}

#[cfg(test)]
mod pipeline_tests {
    use super::*;

    /// Runs the real cut pipeline (face tracking + captions + crop) for one
    /// candidate, exactly as the Tauri command does.
    /// cargo test --lib -- --ignored --nocapture cut_one_candidate
    #[test]
    #[ignore]
    fn cut_one_candidate() {
        let _ = dotenvy::from_path("../.env");
        let candidate_id = std::env::var("CUT_CANDIDATE_ID").expect("CUT_CANDIDATE_ID not set");
        let data_dir = dirs::data_dir().unwrap().join("com.autoshorts.desktop");
        let db = Database::open(&data_dir.join("autoshorts.sqlite")).expect("open db");
        let out = cut_candidate_blocking(db, data_dir, candidate_id).expect("cut failed");
        println!("CUT_OUTPUT={out}");
    }

    /// Runs the app's B-roll enrichment for one candidate, as the command does.
    /// cargo test --lib -- --ignored --nocapture broll_one_candidate
    #[test]
    #[ignore]
    fn broll_one_candidate() {
        let _ = dotenvy::from_path("../.env");
        let candidate_id = std::env::var("CUT_CANDIDATE_ID").expect("CUT_CANDIDATE_ID not set");
        let data_dir = dirs::data_dir().unwrap().join("com.autoshorts.desktop");
        let db = Database::open(&data_dir.join("autoshorts.sqlite")).expect("open db");

        let (candidate, project) = db.get_candidate_with_project(&candidate_id).expect("candidate");
        let detail = db.project_detail(&project.id).expect("detail");
        let clip = detail.clips.iter().find(|c| c.candidate_id == candidate_id).expect("clip");
        let source = PathBuf::from(clip.output_path.as_deref().expect("cut first"));

        let words = db.latest_transcript(&project.id).ok().flatten()
            .and_then(|t| serde_json::from_str::<NormalizedTranscript>(&t.raw_json).ok())
            .map(|n| n.words).unwrap_or_default();

        let output = source.with_file_name(format!(
            "{}_broll.mp4", source.file_stem().unwrap().to_string_lossy()));
        let produced = broll::enrich(&source, &words, candidate.start_sec, candidate.end_sec,
                                     &candidate.hook, &output).expect("broll failed");
        println!("BROLL_OUTPUT={}", produced.display());
    }

    /// Drives the three buttons' code paths in order on an existing project:
    /// Cut -> Add B-roll -> Add Outro.
    /// BTN_CANDIDATE=<id> cargo test --lib -- --ignored --nocapture button_flow
    #[test]
    #[ignore]
    fn button_flow() {
        let _ = dotenvy::from_path("../.env");
        let candidate_id = std::env::var("BTN_CANDIDATE").expect("BTN_CANDIDATE not set");
        let data_dir = dirs::data_dir().unwrap().join("com.autoshorts.desktop");
        let db = Database::open(&data_dir.join("autoshorts.sqlite")).expect("open db");

        // 1. Cut button
        let cut = cut_candidate_blocking(db.clone(), data_dir.clone(), candidate_id.clone())
            .expect("cut failed");
        println!("STEP1_CUT={cut}");

        // 2. Add B-roll button
        let (candidate, project) = db.get_candidate_with_project(&candidate_id).expect("cand");
        let words = db.latest_transcript(&project.id).ok().flatten()
            .and_then(|t| serde_json::from_str::<NormalizedTranscript>(&t.raw_json).ok())
            .map(|n| n.words).unwrap_or_default();
        let cut_path = PathBuf::from(&cut);
        let broll_out = cut_path.with_file_name(format!(
            "{}_broll.mp4", cut_path.file_stem().unwrap().to_string_lossy()));
        let brolled = broll::enrich(&cut_path, &words, candidate.start_sec, candidate.end_sec,
                                    &candidate.hook, &broll_out).expect("broll failed");
        println!("STEP2_BROLL={}", brolled.display());

        // STEP 2b: the persistent title banner, exactly as the button applies it.
        let spoken: String = words.iter()
            .filter(|w| w.end > candidate.start_sec && w.start < candidate.end_sec)
            .map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");
        let headline = tokio::runtime::Runtime::new().unwrap()
            .block_on(title::write_title(&spoken, &candidate.hook));
        println!("STEP2b_TITLE={headline}");
        let titled_out = brolled.with_file_name(format!(
            "{}_titled.mp4", brolled.file_stem().unwrap().to_string_lossy()));
        let brolled = match title::apply(&brolled, &headline, &titled_out) {
            Ok(p) => { println!("STEP2b_TITLED={}", p.display()); p }
            Err(e) => { println!("title skipped: {e}"); brolled }
        };

        // 3. Add Outro button - the exact command the button invokes. Branding
        // is optional, so a project without an app name must refuse cleanly and
        // leave the B-roll cut as the finished video.
        match add_outro_blocking(db, candidate_id, brolled.to_string_lossy().to_string()) {
            Ok(final_path) => println!("STEP3_FINAL={final_path}"),
            Err(reason) if project.brand_name.is_none() => {
                println!("STEP3_SKIPPED (no branding, as expected): {reason}")
            }
            Err(reason) => panic!("outro failed on a branded project: {reason}"),
        }
    }
}

#[cfg(test)]
mod e2e_tests {
    use super::*;

    /// Whole pipeline on one source video, through the app's own code paths:
    /// import -> transcribe -> rank -> cut (captions) -> B-roll -> branded outro.
    ///
    /// E2E_SOURCE=/path/to.mp4 E2E_APP="Name" E2E_LOGO=/path/to.png \
    ///   cargo test --lib -- --ignored --nocapture full_pipeline
    #[tokio::test]
    #[ignore]
    async fn full_pipeline() {
        let _ = dotenvy::from_path("../.env");
        let source = std::env::var("E2E_SOURCE").expect("E2E_SOURCE not set");
        let app_name = std::env::var("E2E_APP").unwrap_or_else(|_| "My App".into());
        let logo = std::env::var("E2E_LOGO").ok();

        let data_dir = dirs::data_dir().unwrap().join("com.autoshorts.desktop");
        let db = Database::open(&data_dir.join("autoshorts.sqlite")).expect("open db");

        let probe = media::probe_media(&source).ok();
        let project = db
            .create_project(&source, "local", "classic-outline",
                            probe.and_then(|p| p.duration_sec),
                            Some(&app_name), logo.as_deref())
            .expect("create project");
        println!("PROJECT={}", project.id);

        // 1. Transcribe with local Whisper.
        let project_dir = data_dir.join("projects").join(&project.id);
        let audio = media::extract_audio(&source, &project_dir).expect("extract audio");
        println!("transcribing (this is the slow step)...");
        let normalized = transcription::transcribe_local(
            &audio.to_string_lossy(), &data_dir.to_string_lossy())
            .await.expect("transcribe");
        println!("words={} duration={:.1}s", normalized.words.len(), normalized.duration);
        let raw = serde_json::to_string(&normalized).unwrap();
        db.save_transcript(&project.id, "local", &raw, Some(&normalized.language))
            .expect("save transcript");

        // 2. Rank viral moments with Anthropic.
        let key = std::env::var("ANTHROPIC_API_KEY")
            .or_else(|_| std::env::var("ANTHROPIC_OAUTH_TOKEN"))
            .expect("no Anthropic credential");
        let drafts = llm::detect_candidates_with_claude(&normalized, &key)
            .await.expect("detect candidates");
        let candidates = db.replace_candidates(&project.id, &drafts).expect("save candidates");
        println!("candidates={}", candidates.len());
        for c in candidates.iter().take(3) {
            println!("  [{:.0}s-{:.0}s] {:.2} {}", c.start_sec, c.end_sec, c.score, c.hook);
        }

        // 3. Cut the top candidate, captions burned in.
        let top = candidates.first().expect("at least one candidate").clone();
        let cut = cut_candidate_blocking(db.clone(), data_dir.clone(), top.id.clone())
            .expect("cut");
        println!("CUT={cut}");

        // 4. B-roll.
        let cut_path = PathBuf::from(&cut);
        let broll_out = cut_path.with_file_name(format!(
            "{}_broll.mp4", cut_path.file_stem().unwrap().to_string_lossy()));
        let brolled = broll::enrich(&cut_path, &normalized.words, top.start_sec,
                                    top.end_sec, &top.hook, &broll_out)
            .expect("broll");
        println!("BROLL={}", brolled.display());

        // 4b. Persistent title banner across the top, over every scene.
        let spoken: String = normalized.words.iter()
            .filter(|w| w.end > top.start_sec && w.start < top.end_sec)
            .map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");
        let headline = title::write_title(&spoken, &top.hook).await;
        println!("TITLE={headline}");
        let titled_out = brolled.with_file_name(format!(
            "{}_titled.mp4", brolled.file_stem().unwrap().to_string_lossy()));
        let brolled = match title::apply(&brolled, &headline, &titled_out) {
            Ok(p) => { println!("TITLED={}", p.display()); p }
            Err(e) => { println!("title skipped: {e}"); brolled }
        };

        // 5. Branded outro on the final artefact.
        let words: Vec<_> = normalized.words.iter()
            .filter(|w| w.end > top.start_sec && w.start < top.end_sec)
            .map(|w| serde_json::json!({
                "text": w.text, "start": w.start - top.start_sec,
                "end": w.end - top.start_sec, "speaker": w.speaker }))
            .collect();
        let tpath = brolled.with_extension("outro_words.json");
        std::fs::write(&tpath, serde_json::json!({"words": words}).to_string()).unwrap();

        let final_out = brolled.with_file_name(format!(
            "{}_final.mp4", brolled.file_stem().unwrap().to_string_lossy()));
        let done = outro::append(&brolled, &app_name,
                                 logo.as_ref().map(std::path::Path::new),
                                 Some(&tpath), &final_out).expect("outro");
        println!("FINAL={}", done.display());
    }
}

#[cfg(test)]
mod branding_tests {
    use super::*;

    /// Branding is optional: a project created without an app name must render
    /// as before, and asking for an end card must refuse cleanly rather than
    /// producing a broken clip.
    #[test]
    fn project_without_branding_has_no_end_card() {
        let dir = std::env::temp_dir().join(format!("autoshorts-brand-{}", uuid::Uuid::new_v4()));
        let db = Database::open(&dir.join("t.sqlite")).expect("open db");

        let plain = db
            .create_project("/tmp/x.mp4", "local", "classic-outline", Some(10.0), None, None)
            .expect("create plain project");
        assert!(plain.brand_name.is_none(), "no app name should be stored");
        assert!(plain.brand_logo_path.is_none(), "no logo should be stored");

        let branded = db
            .create_project("/tmp/y.mp4", "local", "classic-outline", Some(10.0),
                            Some("LabelWise: Food Scanner"),
                            Some("/tmp/logo.png"))
            .expect("create branded project");
        assert_eq!(branded.brand_name.as_deref(), Some("LabelWise: Food Scanner"));
        assert_eq!(branded.brand_logo_path.as_deref(), Some("/tmp/logo.png"));

        // Blank strings count as absent, so a user tabbing through the fields
        // does not end up with an empty end card.
        let blank = db
            .create_project("/tmp/z.mp4", "local", "classic-outline", Some(10.0), Some("  "), Some(""))
            .expect("create blank-branding project");
        assert!(blank.brand_name.is_none(), "whitespace app name must be treated as absent");
        assert!(blank.brand_logo_path.is_none(), "empty logo path must be treated as absent");

        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod resume_tests {
    use super::*;

    /// Continue an interrupted import: reuse the audio already extracted for an
    /// existing project, then transcribe, rank, cut, B-roll and close with the
    /// end card. Avoids re-downloading and re-extracting a long source.
    ///
    /// RESUME_PROJECT=<id> cargo test --lib -- --ignored --nocapture resume_pipeline
    #[tokio::test]
    #[ignore]
    async fn resume_pipeline() {
        let _ = dotenvy::from_path("../.env");
        let project_id = std::env::var("RESUME_PROJECT").expect("RESUME_PROJECT not set");
        let data_dir = dirs::data_dir().unwrap().join("com.autoshorts.desktop");
        let db = Database::open(&data_dir.join("autoshorts.sqlite")).expect("open db");
        let project = db.get_project(&project_id).expect("project");

        // Reuse an existing transcript when the run died after that point.
        let normalized = match db.latest_transcript(&project_id).ok().flatten() {
            Some(t) => {
                println!("reusing saved transcript");
                serde_json::from_str::<NormalizedTranscript>(&t.raw_json).expect("parse transcript")
            }
            None => {
                let audio = data_dir.join("projects").join(&project_id).join("transcription_audio.wav");
                let audio = if audio.exists() {
                    println!("reusing extracted audio");
                    audio
                } else {
                    media::extract_audio(&project.source_path,
                                         &data_dir.join("projects").join(&project_id))
                        .expect("extract audio")
                };
                println!("transcribing (slow step)...");
                let n = transcription::transcribe_local(
                    &audio.to_string_lossy(), &data_dir.to_string_lossy())
                    .await.expect("transcribe");
                db.save_transcript(&project_id, "local",
                                   &serde_json::to_string(&n).unwrap(), Some(&n.language))
                    .expect("save transcript");
                n
            }
        };
        println!("words={} duration={:.1}s", normalized.words.len(), normalized.duration);

        let mut candidates = db.list_candidates(&project_id).unwrap_or_default();
        if candidates.is_empty() {
            let key = std::env::var("ANTHROPIC_API_KEY")
                .or_else(|_| std::env::var("ANTHROPIC_OAUTH_TOKEN"))
                .expect("no Anthropic credential");
            let drafts = llm::detect_candidates_with_claude(&normalized, &key)
                .await.expect("detect candidates");
            candidates = db.replace_candidates(&project_id, &drafts).expect("save candidates");
        }
        println!("candidates={}", candidates.len());
        for c in candidates.iter().take(3) {
            println!("  [{:.0}s-{:.0}s] {:.2} {}", c.start_sec, c.end_sec, c.score, c.hook);
        }

        let top = candidates.first().expect("a candidate").clone();
        let cut = cut_candidate_blocking(db.clone(), data_dir.clone(), top.id.clone())
            .expect("cut");
        println!("CUT={cut}");

        let cut_path = PathBuf::from(&cut);
        let broll_out = cut_path.with_file_name(format!(
            "{}_broll.mp4", cut_path.file_stem().unwrap().to_string_lossy()));
        let brolled = broll::enrich(&cut_path, &normalized.words, top.start_sec,
                                    top.end_sec, &top.hook, &broll_out).expect("broll");
        println!("BROLL={}", brolled.display());

        match add_outro_blocking(db, top.id, brolled.to_string_lossy().to_string()) {
            Ok(f) => println!("FINAL={f}"),
            Err(e) => println!("OUTRO SKIPPED: {e}"),
        }
    }
}

#[cfg(test)]
mod batch_tests {
    use super::*;

    /// Render every candidate of every project that has a transcript.
    ///
    /// Resumable: a candidate whose finished file already exists is skipped, so
    /// an interrupted run continues rather than repeating hours of encoding.
    /// Re-ranks first when a project holds fewer candidates than MAX_CANDIDATES,
    /// so raising the cap yields more clips from sources already transcribed.
    ///
    /// cargo test --lib -- --ignored --nocapture batch_render_all
    #[tokio::test]
    #[ignore]
    async fn batch_render_all() {
        let _ = dotenvy::from_path("../.env");
        let data_dir = dirs::data_dir().unwrap().join("com.autoshorts.desktop");
        let db = Database::open(&data_dir.join("autoshorts.sqlite")).expect("open db");
        let want: usize = std::env::var("MAX_CANDIDATES").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(25);

        let mut projects = db.list_projects().expect("projects");
        // Skip the duplicate branding-test projects: same source, same clips.
        let mut seen_sources = std::collections::HashSet::new();
        projects.retain(|p| seen_sources.insert(p.source_path.clone()));
        println!("BATCH: {} unique source(s)", projects.len());

        let mut made = 0usize;
        let mut skipped = 0usize;
        let mut failed = 0usize;

        for project in projects {
            let Some(transcript) = db.latest_transcript(&project.id).ok().flatten() else {
                println!("SKIP {}: no transcript", &project.id[..8]);
                continue;
            };
            let Ok(normalized) = serde_json::from_str::<NormalizedTranscript>(&transcript.raw_json)
            else {
                println!("SKIP {}: unreadable transcript", &project.id[..8]);
                continue;
            };

            let mut candidates = db.list_candidates(&project.id).unwrap_or_default();
            if candidates.len() < want {
                let key = std::env::var("ANTHROPIC_API_KEY")
                    .or_else(|_| std::env::var("ANTHROPIC_OAUTH_TOKEN"))
                    .unwrap_or_default();
                if !key.is_empty() {
                    println!("RANK {}: {} -> asking for up to {}",
                             &project.id[..8], candidates.len(), want);
                    match llm::detect_candidates_with_claude(&normalized, &key).await {
                        Ok(drafts) => {
                            if drafts.len() > candidates.len() {
                                let _ = db.replace_candidates(&project.id, &drafts);
                            }
                        }
                        Err(e) => println!("  ranking failed, keeping existing: {e}"),
                    }
                    // Re-read rather than trusting the returned vector: ranking
                    // replaces rows, and the app may hold the database open at
                    // the same time. The stored rows are the only ones whose ids
                    // the cut step can resolve.
                    candidates = db.list_candidates(&project.id).unwrap_or_default();
                }
            }
            println!("PROJECT {} ({}): {} candidate(s)",
                     &project.id[..8],
                     std::path::Path::new(&project.source_path)
                         .file_name().unwrap_or_default().to_string_lossy(),
                     candidates.len());

            for candidate in candidates {
                let label = format!("{}#{}", &project.id[..8], candidate.rank);

                let cut = match cut_candidate_blocking(
                    db.clone(), data_dir.clone(), candidate.id.clone()) {
                    Ok(c) => c,
                    Err(e) if e.contains("no rows") => {
                        println!("  SKIP {label}: candidate no longer in the database");
                        skipped += 1;
                        continue;
                    }
                    Err(e) => { println!("  FAIL {label} cut: {e}"); failed += 1; continue; }
                };
                let cut_path = PathBuf::from(&cut);
                let broll_out = cut_path.with_file_name(format!(
                    "{}_broll.mp4", cut_path.file_stem().unwrap().to_string_lossy()));
                let final_out = broll_out.with_file_name(format!(
                    "{}_final.mp4", broll_out.file_stem().unwrap().to_string_lossy()));

                // Resume: a finished clip is left alone.
                //
                // The artefact name grew as stages were added (_broll ->
                // _titled -> _titled_sfx -> _titled_sfx_final), and a check that
                // only knew the old names stopped recognising finished work --
                // so the batch re-planned clips it had already rendered and
                // burned quota doing it. Check every finished form there is.
                let stem = broll_out.file_stem().unwrap_or_default().to_string_lossy().to_string();
                let finished_forms = [
                    format!("{stem}_titled_sfx_final.mp4"),
                    format!("{stem}_titled_sfx.mp4"),
                    format!("{stem}_titled_final.mp4"),
                    format!("{stem}_titled.mp4"),
                    format!("{stem}_final.mp4"),
                ];
                if let Some(found) = finished_forms
                    .iter()
                    .map(|n| broll_out.with_file_name(n))
                    .find(|p| p.exists())
                {
                    println!("  SKIP {label}: already rendered ({})",
                             found.file_name().unwrap_or_default().to_string_lossy());
                    // Make sure the app knows about it even if an earlier run
                    // finished before paths were being recorded.
                    let _ = db.set_broll_path(&candidate.id, &found.to_string_lossy());
                    skipped += 1;
                    continue;
                }
                let _ = &final_out;

                let brolled = if broll_out.exists() {
                    broll_out.clone()
                } else {
                    match broll::enrich(&cut_path, &normalized.words, candidate.start_sec,
                                        candidate.end_sec, &candidate.hook, &broll_out) {
                        Ok(b) => b,
                        Err(e) => { println!("  FAIL {label} broll: {e}"); failed += 1; continue; }
                    }
                };

                // Persistent headline across the top, applied after B-roll so
                // it holds over every scene and before the outro so it does not
                // sit on the end card.
                let spoken: String = normalized.words.iter()
                    .filter(|w| w.end > candidate.start_sec && w.start < candidate.end_sec)
                    .map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");
                let headline = title::write_title(&spoken, &candidate.hook).await;
                let titled_out = brolled.with_file_name(format!(
                    "{}_titled.mp4",
                    brolled.file_stem().unwrap_or_default().to_string_lossy()));
                let brolled = match title::apply(&brolled, &headline, &titled_out) {
                    Ok(p) => { println!("  TITLE {label}: {headline}"); p }
                    Err(e) => { println!("  title skipped {label}: {e}"); brolled }
                };

                // Sound effects, placed from the edit itself. Mixed before the
                // outro so the end card keeps its own cloned-voice line clean,
                // and after the title because the video stream is copied here --
                // adding sound costs no picture quality.
                let plan = cut_path.with_file_name(format!(
                    "edit_{}", cut_path.file_stem().unwrap_or_default().to_string_lossy()))
                    .join("scene_plan.json");
                let tx = cut_path.with_file_name("broll_transcript.json");
                let sfx_out = brolled.with_file_name(format!(
                    "{}_sfx.mp4",
                    brolled.file_stem().unwrap_or_default().to_string_lossy()));
                let brolled = match sfx::apply(&brolled, Some(&plan), Some(&tx), &sfx_out) {
                    Ok(p) => { println!("  SFX {label}"); p }
                    Err(e) => { println!("  sfx skipped {label}: {e}"); brolled }
                };

                // No end card unless it is asked for. Appending it in bulk
                // put an advert on clips that were never meant to carry one;
                // the outro is now a deliberate press of the clip's own button.
                let finished = brolled.to_string_lossy().to_string();
                // Record it, so the app shows the finished clip and its path
                // instead of offering to build one that already exists.
                if let Err(e) = db.set_broll_path(&candidate.id, &finished) {
                    println!("  (could not record path: {e})");
                }
                println!("  DONE {label}: {finished}");
                made += 1;
            }
        }
        println!("\nBATCH COMPLETE: {made} rendered, {skipped} already done, {failed} failed");
    }
}

#[cfg(test)]
mod rank_debug {
    use super::*;

    /// Re-rank one project and check the rows actually land in the database.
    /// RANK_PROJECT=<id> cargo test --lib -- --ignored --nocapture rank_persists
    #[tokio::test]
    #[ignore]
    async fn rank_persists() {
        let _ = dotenvy::from_path("../.env");
        let pid = std::env::var("RANK_PROJECT").expect("RANK_PROJECT not set");
        let data_dir = dirs::data_dir().unwrap().join("com.autoshorts.desktop");
        let db = Database::open(&data_dir.join("autoshorts.sqlite")).expect("open db");

        let before = db.list_candidates(&pid).unwrap_or_default().len();
        let t = db.latest_transcript(&pid).unwrap().unwrap();
        let n: NormalizedTranscript = serde_json::from_str(&t.raw_json).unwrap();

        let key = std::env::var("ANTHROPIC_API_KEY")
            .or_else(|_| std::env::var("ANTHROPIC_OAUTH_TOKEN")).unwrap();
        let drafts = llm::detect_candidates_with_claude(&n, &key).await.expect("rank");
        println!("MAX_CANDIDATES={:?}", std::env::var("MAX_CANDIDATES"));
        println!("drafts returned by llm : {}", drafts.len());

        let saved = db.replace_candidates(&pid, &drafts).expect("save");
        println!("replace_candidates gave: {}", saved.len());

        let after = db.list_candidates(&pid).unwrap_or_default();
        println!("rows in db after       : {}", after.len());
        println!("before                 : {before}");

        // The bug to catch: ids handed back that are not actually stored.
        let stored: std::collections::HashSet<_> = after.iter().map(|c| c.id.clone()).collect();
        let missing: Vec<_> = saved.iter().filter(|c| !stored.contains(&c.id)).collect();
        println!("handed back but NOT in db: {}", missing.len());
    }
}

#[cfg(test)]
mod broll_path_tests {
    use super::*;

    /// Confirms the B-roll path survives the round trip the UI depends on:
    /// database -> project_detail -> serialised JSON field `brollPath`.
    /// PROJ=<project_id> cargo test --lib -- --ignored --nocapture broll_path_reaches_ui
    #[test]
    #[ignore]
    fn broll_path_reaches_ui() {
        let data_dir = dirs::data_dir().unwrap().join("com.autoshorts.desktop");
        let db = Database::open(&data_dir.join("autoshorts.sqlite")).expect("open db");
        let pid = std::env::var("PROJ").expect("PROJ not set");
        let detail = db.project_detail(&pid).expect("detail");
        let json = serde_json::to_value(&detail).expect("serialise");
        let clips = json["clips"].as_array().expect("clips array");
        let mut found = 0;
        for c in clips {
            if let Some(b) = c["brollPath"].as_str() {
                found += 1;
                println!("brollPath present: {b}");
            }
        }
        println!("clips: {}, with brollPath: {found}", clips.len());
        assert!(found > 0, "no clip carried brollPath through to the UI payload");
    }
}

#[cfg(test)]
mod retitle_tests {
    use super::*;

    /// Add the title banner to every clip already rendered, without redoing the
    /// expensive work.
    ///
    /// A clip's B-roll plan costs thousands of tokens; its title costs a few
    /// dozen. So finished renders are retitled in place rather than rebuilt --
    /// the whole sweep costs about as much as planning one clip. Where the
    /// project is branded, the outro is re-appended afterwards so the banner
    /// does not sit over the end card. Outro rendering is entirely local, so
    /// that step costs no quota at all.
    ///
    /// cargo test --lib --release -- --ignored --nocapture retitle_all
    #[tokio::test]
    #[ignore]
    async fn retitle_all() {
        let _ = dotenvy::from_path("../.env");
        let data_dir = dirs::data_dir().unwrap().join("com.autoshorts.desktop");
        let db = Database::open(&data_dir.join("autoshorts.sqlite")).expect("open db");

        let projects = db.list_projects().expect("projects");
        let mut done = 0usize;
        let mut skipped = 0usize;
        let mut failed = 0usize;

        for project in projects {
            let Some(transcript) = db.latest_transcript(&project.id).ok().flatten() else {
                continue;
            };
            let Ok(normalized) = serde_json::from_str::<NormalizedTranscript>(&transcript.raw_json)
            else {
                continue;
            };

            for candidate in db.list_candidates(&project.id).unwrap_or_default() {
                let Ok(clips_dir) = documents_project_dir(&project).map(|d| d.join("clips"))
                else {
                    continue;
                };
                let broll = clips_dir.join(format!("clip-{:02}_flat_broll.mp4", candidate.rank));
                if !broll.exists() {
                    continue;
                }
                let titled = clips_dir.join(format!("clip-{:02}_flat_broll_titled.mp4", candidate.rank));
                if titled.exists() {
                    println!("SKIP {} #{}: already titled", &project.id[..8], candidate.rank);
                    skipped += 1;
                    continue;
                }

                let spoken: String = normalized
                    .words
                    .iter()
                    .filter(|w| w.end > candidate.start_sec && w.start < candidate.end_sec)
                    .map(|w| w.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
                let headline = title::write_title(&spoken, &candidate.hook).await;

                match title::apply(&broll, &headline, &titled) {
                    Ok(p) => {
                        println!("TITLED {} #{}: \"{headline}\" -> {}",
                                 &project.id[..8], candidate.rank, p.display());
                        done += 1;
                        // Same sound pass the batch applies, so a retitled clip
                        // is indistinguishable from a freshly rendered one.
                        let plan = clips_dir
                            .join(format!("edit_clip-{:02}_flat", candidate.rank))
                            .join("scene_plan.json");
                        let tx = clips_dir.join("broll_transcript.json");
                        let sfx_out = clips_dir.join(
                            format!("clip-{:02}_flat_broll_titled_sfx.mp4", candidate.rank));
                        let p = match sfx::apply(&p, Some(&plan), Some(&tx), &sfx_out) {
                            Ok(q) => { println!("  sfx mixed"); q }
                            Err(e) => { println!("  sfx skipped: {e}"); p }
                        };
                        // Outro deliberately not applied here either.
                        let _ = &project;
                        {
                            let _ = db.set_broll_path(
                                &candidate.id, &p.to_string_lossy().to_string());
                        }
                    }
                    Err(e) => {
                        println!("FAIL {} #{}: {e}", &project.id[..8], candidate.rank);
                        failed += 1;
                    }
                }
            }
        }
        println!("\nRETITLE COMPLETE: {done} titled, {skipped} already done, {failed} failed");
    }
}

#[cfg(test)]
mod parallel_batch {
    use super::*;

    /// Render every pending clip, several at a time.
    ///
    /// The serial batch spent most of its wall-clock waiting: one Anthropic
    /// planning call, then minutes of ffmpeg, then the next call. Running
    /// several clips at once overlaps one clip's encode with another's planning,
    /// which both finishes sooner and spends the quota while it is available.
    ///
    /// Safe to parallelise because each clip now owns its working directory and
    /// its downloads are named per clip, so two renders cannot touch the same
    /// files. Concurrency is deliberately modest: ffmpeg is itself threaded, so
    /// too many workers thrash rather than help.
    ///
    /// BATCH_CONCURRENCY=3 cargo test --lib --release -- --ignored --nocapture render_parallel
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    #[ignore]
    async fn render_parallel() {
        let _ = dotenvy::from_path("../.env");
        let data_dir = dirs::data_dir().unwrap().join("com.autoshorts.desktop");
        let db = Database::open(&data_dir.join("autoshorts.sqlite")).expect("open db");
        let concurrency: usize = std::env::var("BATCH_CONCURRENCY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);

        // Build the work list first, so the pending set is known up front and
        // the same clip cannot be picked up twice.
        let mut work: Vec<(Project, Candidate, NormalizedTranscript)> = Vec::new();
        let mut projects = db.list_projects().expect("projects");
        let mut seen = std::collections::HashSet::new();
        projects.retain(|p| seen.insert(p.source_path.clone()));

        for project in projects {
            let Some(t) = db.latest_transcript(&project.id).ok().flatten() else { continue };
            let Ok(normalized) = serde_json::from_str::<NormalizedTranscript>(&t.raw_json) else {
                continue;
            };
            let Ok(clips_dir) = documents_project_dir(&project).map(|d| d.join("clips")) else {
                continue;
            };
            for candidate in db.list_candidates(&project.id).unwrap_or_default() {
                let stem = format!("clip-{:02}_flat_broll", candidate.rank);
                let finished = [
                    format!("{stem}_titled_sfx_final.mp4"),
                    format!("{stem}_titled_sfx.mp4"),
                    format!("{stem}_titled.mp4"),
                    format!("{stem}_final.mp4"),
                ]
                .iter()
                .any(|n| clips_dir.join(n).exists());
                if finished {
                    continue;
                }
                work.push((project.clone(), candidate, normalized.clone()));
            }
        }
        println!("PARALLEL: {} clip(s) pending, {concurrency} at a time", work.len());

        let sem = std::sync::Arc::new(tokio::sync::Semaphore::new(concurrency));
        let mut set = tokio::task::JoinSet::new();

        for (project, candidate, normalized) in work {
            let sem = sem.clone();
            let db = db.clone();
            let data_dir = data_dir.clone();
            set.spawn(async move {
                let _permit = sem.acquire_owned().await.ok()?;
                let label = format!("{}#{}", &project.id[..8], candidate.rank);

                let cut = tokio::task::spawn_blocking({
                    let db = db.clone();
                    let data_dir = data_dir.clone();
                    let cid = candidate.id.clone();
                    move || cut_candidate_blocking(db, data_dir, cid)
                })
                .await
                .ok()?;
                let cut = match cut {
                    Ok(c) => PathBuf::from(c),
                    Err(e) => { println!("  FAIL {label} cut: {e}"); return None; }
                };

                let broll_out = cut.with_file_name(format!(
                    "{}_broll.mp4", cut.file_stem().unwrap_or_default().to_string_lossy()));
                let brolled = tokio::task::spawn_blocking({
                    let words = normalized.words.clone();
                    let (s, e, hook) = (candidate.start_sec, candidate.end_sec, candidate.hook.clone());
                    let (cut, out) = (cut.clone(), broll_out.clone());
                    move || broll::enrich(&cut, &words, s, e, &hook, &out)
                })
                .await
                .ok()?;
                let brolled = match brolled {
                    Ok(b) => b,
                    Err(e) => { println!("  FAIL {label} broll: {e}"); return None; }
                };

                let spoken: String = normalized.words.iter()
                    .filter(|w| w.end > candidate.start_sec && w.start < candidate.end_sec)
                    .map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");
                let headline = title::write_title(&spoken, &candidate.hook).await;
                println!("  TITLE {label}: {headline}");

                let finished = tokio::task::spawn_blocking({
                    let db = db.clone();
                    let cid = candidate.id.clone();
                    let branded = batch_should_add_outro(&project);
                    let cut = cut.clone();
                    move || {
                        let titled = brolled.with_file_name(format!(
                            "{}_titled.mp4",
                            brolled.file_stem().unwrap_or_default().to_string_lossy()));
                        let v = title::apply(&brolled, &headline, &titled).unwrap_or(brolled);

                        let plan = cut.with_file_name(format!(
                            "edit_{}", cut.file_stem().unwrap_or_default().to_string_lossy()))
                            .join("scene_plan.json");
                        let tx = cut.with_file_name("broll_transcript.json");
                        let sfx_out = v.with_file_name(format!(
                            "{}_sfx.mp4", v.file_stem().unwrap_or_default().to_string_lossy()));
                        let v = sfx::apply(&v, Some(&plan), Some(&tx), &sfx_out).unwrap_or(v);

                        // No end card unless it is asked for. Aariz adds the
                        // outro himself from the clip's own button when he
                        // wants one, so appending it automatically put an
                        // advert on clips that were never meant to carry one.
                        let _ = branded;
                        let _ = db.set_broll_path(&cid, &v.to_string_lossy());
                        v
                    }
                })
                .await
                .ok()?;
                println!("  DONE {label}: {}", finished.display());
                Some(())
            });
        }

        let mut made = 0usize;
        while let Some(res) = set.join_next().await {
            if matches!(res, Ok(Some(()))) { made += 1; }
        }
        println!("\nPARALLEL COMPLETE: {made} rendered");
    }
}

#[cfg(test)]
mod branding_gate_tests {
    use super::*;

    fn project_with(name: Option<&str>, logo: Option<&str>) -> Project {
        Project {
            id: "t".into(),
            name: None,
            source_path: "/tmp/x.mp4".into(),
            source_duration: None,
            status: "ready".into(),
            transcription_mode: "local".into(),
            caption_style: None,
            brand_name: name.map(str::to_string),
            brand_logo_path: logo.map(str::to_string),
            created_at: String::new(),
            updated_at: String::new(),
        }
    }

    /// The end card advertised "My App - download on the App Store" on a
    /// project that never had an app name: a placeholder left by a test run
    /// satisfied a bare is_some() check. A placeholder is not branding.
    #[test]
    fn placeholder_app_name_is_not_branding() {
        for name in ["My App", "my app", "  App Name  ", "your app", "TEST"] {
            assert!(
                !has_real_branding(&project_with(Some(name), Some("/tmp"))),
                "{name:?} should not count as branding"
            );
        }
    }

    /// A real name with no logo is still not enough to brand the end card.
    #[test]
    fn a_name_without_a_logo_is_not_branding() {
        assert!(!has_real_branding(&project_with(Some("LabelWise"), None)));
        assert!(!has_real_branding(&project_with(Some("LabelWise"), Some(""))));
        assert!(!has_real_branding(&project_with(
            Some("LabelWise"),
            Some("/nope/missing-logo.png")
        )));
    }

    /// A real name plus a logo that exists on disk does brand it.
    ///
    /// The logo is created here rather than borrowed from the system: pointing
    /// at a path like /etc/hosts makes the test depend on the host's layout.
    #[test]
    fn real_name_and_present_logo_is_branding() {
        let logo = std::env::temp_dir().join("autoshorts_test_logo.png");
        std::fs::write(&logo, b"not really a png, but it exists").expect("write logo");
        assert!(has_real_branding(&project_with(
            Some("LabelWise: Food Scanner"),
            Some(&logo.to_string_lossy())
        )));
        let _ = std::fs::remove_file(&logo);
    }
}
