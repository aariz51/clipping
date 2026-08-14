use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::models::{CandidateDraft, NormalizedTranscript, TranscriptSegment};

#[derive(Debug, Deserialize)]
struct AnthropicMessage {
    content: Vec<AnthropicContent>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContent {
    text: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct DeepseekMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct DeepseekChoice {
    message: DeepseekMessage,
}

#[derive(Debug, Deserialize)]
struct DeepseekResponse {
    choices: Vec<DeepseekChoice>,
}

pub async fn detect_candidates_with_deepseek(
    transcript: &NormalizedTranscript,
    api_key: &str,
    model_name: Option<&str>,
) -> Result<Vec<CandidateDraft>> {
    let segments = compact_segments(&transcript.segments);
    let prompt = format!(
        "You are an elite, world-class social media strategist with a track record of generating viral multi-million-view Shorts, TikToks, and Reels. \
Your sole objective is to identify the ABSOLUTE BEST, most highly-engaging, and trend-setting short-form clip candidates from the provided transcript. \
Do NOT pick random or mediocre segments. Be ruthless in your selection, but extract AS MANY highly viral moments as possible. \
Every candidate must have an insanely strong, curiosity-inducing hook in the first 3 seconds to stop the scroll. \
Clips should be 30-90 seconds long, completely self-contained, cut at clean boundaries, and deliver a massive payoff (a mind-blowing fact, hilarious joke, highly controversial opinion, or deep emotional insight). \
Return up to 25 candidates as JSON matching exactly this schema: \
{{\"candidates\":[{{\"start\":0.0,\"end\":0.0,\"score\":0.0,\"hook\":\"...\",\"rationale\":\"...\"}}]}}

Transcript:
{segments}"
    );

    let default_model = "deepseek-chat".to_string();
    let model = model_name
        .filter(|m| !m.trim().is_empty())
        .map(|m| m.trim().to_string())
        .or_else(|| std::env::var("DEEPSEEK_MODEL").ok().filter(|m| !m.trim().is_empty()))
        .unwrap_or(default_model);

    let response = reqwest::Client::new()
        .post("https://api.deepseek.com/chat/completions")
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&json!({
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.2,
            "response_format": {
                "type": "json_object"
            }
        }))
        .send()
        .await
        .context("calling DeepSeek")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!("DeepSeek request failed ({status}): {body}"));
    }

    let res_body: DeepseekResponse = response.json().await.context("parsing DeepSeek response")?;
    let text = res_body
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .ok_or_else(|| anyhow!("DeepSeek response did not include choices content"))?;

    let min_duration = if transcript.duration < 60.0 {
        (transcript.duration * 0.5).max(5.0)
    } else {
        30.0
    };
    parse_candidate_json(&text, min_duration)
}

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: GeminiContent,
}

#[derive(Debug, Deserialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Deserialize)]
struct GeminiPart {
    text: Option<String>,
}

pub async fn detect_candidates_with_gemini(
    transcript: &NormalizedTranscript,
    api_key: &str,
) -> Result<Vec<CandidateDraft>> {
    let segments = compact_segments(&transcript.segments);
    let prompt = format!(
        "You are an elite, world-class social media strategist with a track record of generating viral multi-million-view Shorts, TikToks, and Reels. \
Your sole objective is to identify the ABSOLUTE BEST, most highly-engaging, and trend-setting short-form clip candidates from the provided transcript. \
Do NOT pick random or mediocre segments. Be ruthless in your selection, but extract AS MANY highly viral moments as possible. \
Every candidate must have an insanely strong, curiosity-inducing hook in the first 3 seconds to stop the scroll. \
Clips should be 30-90 seconds long, completely self-contained, cut at clean boundaries, and deliver a massive payoff (a mind-blowing fact, hilarious joke, highly controversial opinion, or deep emotional insight). \
Return up to 25 candidates as JSON matching exactly this schema: \
{{\"candidates\":[{{\"start\":0.0,\"end\":0.0,\"score\":0.0,\"hook\":\"...\",\"rationale\":\"...\"}}]}}

Transcript:
{segments}"
    );

    let model = std::env::var("GEMINI_MODEL").unwrap_or_else(|_| "gemini-2.5-flash".to_string());
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        model, api_key
    );

    let response = reqwest::Client::new()
        .post(&url)
        .json(&json!({
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.2
            }
        }))
        .send()
        .await
        .context("calling Gemini")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!("Gemini request failed ({status}): {body}"));
    }

    let res_body: GeminiResponse = response.json().await.context("parsing Gemini response")?;
    let text = res_body
        .candidates
        .first()
        .and_then(|c| c.content.parts.first())
        .and_then(|p| p.text.clone())
        .ok_or_else(|| anyhow!("Gemini response did not include content text"))?;

    let min_duration = if transcript.duration < 60.0 {
        (transcript.duration * 0.5).max(5.0)
    } else {
        30.0
    };
    parse_candidate_json(&text, min_duration)
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatCompletionChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionChoice {
    message: ChatCompletionMessage,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionMessage {
    content: String,
}

pub async fn detect_candidates_with_openai(
    transcript: &NormalizedTranscript,
    api_key: &str,
) -> Result<Vec<CandidateDraft>> {
    let segments = compact_segments(&transcript.segments);
    let prompt = format!(
        "You are an elite, world-class social media strategist with a track record of generating viral multi-million-view Shorts, TikToks, and Reels. \
Your sole objective is to identify the ABSOLUTE BEST, most highly-engaging, and trend-setting short-form clip candidates from the provided transcript. \
Do NOT pick random or mediocre segments. Be ruthless in your selection, but extract AS MANY highly viral moments as possible. \
Every candidate must have an insanely strong, curiosity-inducing hook in the first 3 seconds to stop the scroll. \
Clips should be 30-90 seconds long, completely self-contained, cut at clean boundaries, and deliver a massive payoff (a mind-blowing fact, hilarious joke, highly controversial opinion, or deep emotional insight). \
Return up to 25 candidates as JSON matching exactly this schema: \
{{\"candidates\":[{{\"start\":0.0,\"end\":0.0,\"score\":0.0,\"hook\":\"...\",\"rationale\":\"...\"}}]}}

Transcript:
{segments}"
    );

    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());

    let response = reqwest::Client::new()
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&json!({
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.2,
            "response_format": {
                "type": "json_object"
            }
        }))
        .send()
        .await
        .context("calling OpenAI")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!("OpenAI request failed ({status}): {body}"));
    }

    let res_body: ChatCompletionResponse =
        response.json().await.context("parsing OpenAI response")?;
    let text = res_body
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .ok_or_else(|| anyhow!("OpenAI response did not include choices content"))?;

    let min_duration = if transcript.duration < 60.0 {
        (transcript.duration * 0.5).max(5.0)
    } else {
        30.0
    };
    parse_candidate_json(&text, min_duration)
}

pub async fn detect_candidates_with_openrouter(
    transcript: &NormalizedTranscript,
    api_key: &str,
    model_name: Option<&str>,
) -> Result<Vec<CandidateDraft>> {
    let segments = compact_segments(&transcript.segments);
    let prompt = format!(
        "You are an elite, world-class social media strategist with a track record of generating viral multi-million-view Shorts, TikToks, and Reels. \
Your sole objective is to identify the ABSOLUTE BEST, most highly-engaging, and trend-setting short-form clip candidates from the provided transcript. \
Do NOT pick random or mediocre segments. Be ruthless in your selection, but extract AS MANY highly viral moments as possible. \
Every candidate must have an insanely strong, curiosity-inducing hook in the first 3 seconds to stop the scroll. \
Clips should be 30-90 seconds long, completely self-contained, cut at clean boundaries, and deliver a massive payoff (a mind-blowing fact, hilarious joke, highly controversial opinion, or deep emotional insight). \
Return up to 25 candidates as JSON matching exactly this schema: \
{{\"candidates\":[{{\"start\":0.0,\"end\":0.0,\"score\":0.0,\"hook\":\"...\",\"rationale\":\"...\"}}]}}

Transcript:
{segments}"
    );

    let default_model = "google/gemini-2.5-flash".to_string();
    let model = model_name
        .filter(|m| !m.trim().is_empty())
        .map(|m| m.trim().to_string())
        .or_else(|| std::env::var("OPENROUTER_MODEL").ok().filter(|m| !m.trim().is_empty()))
        .unwrap_or(default_model);

    // OpenRouter fronts many vendors and not all accept `response_format`.
    // Anthropic models in particular reject JSON mode, so it is requested only
    // where supported, and a rejection is retried without it. The parser
    // tolerates prose-wrapped JSON either way.
    let supports_json_mode = {
        let m = model.to_lowercase();
        !(m.contains("anthropic") || m.contains("claude"))
    };

    let client = reqwest::Client::new();
    let build_body = |json_mode: bool| {
        let mut body = json!({
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.2,
        });
        if json_mode {
            body["response_format"] = json!({ "type": "json_object" });
        }
        body
    };

    let send = |json_mode: bool| {
        client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .header("Authorization", format!("Bearer {api_key}"))
            // OpenRouter attributes traffic via these; harmless but expected.
            .header("HTTP-Referer", "https://github.com/JayWebtech/autoshorts")
            .header("X-Title", "AutoShorts")
            .json(&build_body(json_mode))
            .send()
    };

    let mut json_mode = supports_json_mode;
    let mut response = send(json_mode).await.context("calling OpenRouter")?;

    if !response.status().is_success() && json_mode {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        // Only a complaint about the JSON-mode parameter is worth retrying.
        if body.to_lowercase().contains("response_format") {
            json_mode = false;
            response = send(json_mode)
                .await
                .context("retrying OpenRouter without JSON mode")?;
        } else {
            return Err(anyhow!("OpenRouter request failed ({status}): {body}"));
        }
    }

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!("OpenRouter request failed ({status}): {body}"));
    }

    let res_body: ChatCompletionResponse = response
        .json()
        .await
        .context("parsing OpenRouter response")?;
    let text = res_body
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .ok_or_else(|| anyhow!("OpenRouter response did not include choices content"))?;

    let min_duration = if transcript.duration < 60.0 {
        (transcript.duration * 0.5).max(5.0)
    } else {
        30.0
    };
    parse_candidate_json(&text, min_duration)
}

pub async fn detect_candidates_with_groq(
    transcript: &NormalizedTranscript,
    api_key: &str,
) -> Result<Vec<CandidateDraft>> {
    let segments = compact_segments(&transcript.segments);
    let prompt = format!(
        "You are an elite, world-class social media strategist with a track record of generating viral multi-million-view Shorts, TikToks, and Reels. \
Your sole objective is to identify the ABSOLUTE BEST, most highly-engaging, and trend-setting short-form clip candidates from the provided transcript. \
Do NOT pick random or mediocre segments. Be ruthless in your selection, but extract AS MANY highly viral moments as possible. \
Every candidate must have an insanely strong, curiosity-inducing hook in the first 3 seconds to stop the scroll. \
Clips should be 30-90 seconds long, completely self-contained, cut at clean boundaries, and deliver a massive payoff (a mind-blowing fact, hilarious joke, highly controversial opinion, or deep emotional insight). \
Return up to 25 candidates as JSON matching exactly this schema: \
{{\"candidates\":[{{\"start\":0.0,\"end\":0.0,\"score\":0.0,\"hook\":\"...\",\"rationale\":\"...\"}}]}}

Transcript:
{segments}"
    );

    let model =
        std::env::var("GROQ_MODEL").unwrap_or_else(|_| "llama-3.3-70b-versatile".to_string());

    let response = reqwest::Client::new()
        .post("https://api.groq.com/openai/v1/chat/completions")
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&json!({
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0.2,
            "response_format": {
                "type": "json_object"
            }
        }))
        .send()
        .await
        .context("calling Groq")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!("Groq request failed ({status}): {body}"));
    }

    let res_body: ChatCompletionResponse = response
        .json()
        .await
        .context("parsing Groq response")?;
    let text = res_body
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .ok_or_else(|| anyhow!("Groq response did not include choices content"))?;

    let min_duration = if transcript.duration < 60.0 {
        (transcript.duration * 0.5).max(5.0)
    } else {
        30.0
    };
    parse_candidate_json(&text, min_duration)
}

#[derive(Debug, Serialize)]
struct ClaudeMessage<'a> {
    role: &'a str,
    content: String,
}

pub async fn detect_candidates_with_claude(
    transcript: &NormalizedTranscript,
    api_key: &str,
) -> Result<Vec<CandidateDraft>> {
    let segments = compact_segments(&transcript.segments);
    let prompt = format!(
        "You are an elite, world-class social media strategist with a track record of generating viral multi-million-view Shorts, TikToks, and Reels. \
Your sole objective is to identify the ABSOLUTE BEST, most highly-engaging, and trend-setting short-form clip candidates from the provided transcript. \
Do NOT pick random or mediocre segments. Be ruthless in your selection, but extract AS MANY highly viral moments as possible. \
Every candidate must have an insanely strong, curiosity-inducing hook in the first 3 seconds to stop the scroll. \
Clips should be 30-90 seconds long, completely self-contained, cut at clean boundaries, and deliver a massive payoff (a mind-blowing fact, hilarious joke, highly controversial opinion, or deep emotional insight). \
Return up to 25 candidates as JSON matching exactly this schema: \
{{\"candidates\":[{{\"start\":0.0,\"end\":0.0,\"score\":0.0,\"hook\":\"...\",\"rationale\":\"...\"}}]}}

Transcript:
{segments}"
    );

    let model =
        std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-3-5-sonnet-latest".to_string());

    // Anthropic accepts two credential shapes and they authenticate differently.
    // Console keys (`sk-ant-api...`) go in `x-api-key`. Claude Code subscription
    // tokens (`sk-ant-oat...`) are OAuth access tokens: they authenticate as a
    // bearer token with the OAuth beta header, and are rejected outright by
    // `x-api-key`.
    let is_oauth = api_key.starts_with("sk-ant-oat");
    let mut builder = reqwest::Client::new()
        .post("https://api.anthropic.com/v1/messages")
        .header("anthropic-version", "2023-06-01");
    builder = if is_oauth {
        builder
            .header("authorization", format!("Bearer {api_key}"))
            .header("anthropic-beta", "oauth-2025-04-20")
    } else {
        builder.header("x-api-key", api_key)
    };

    let response = builder
        .json(&json!({
            "model": model,
            "max_tokens": 8000,
            "temperature": 0.2,
            "messages": [
                ClaudeMessage {
                    role: "user",
                    content: prompt,
                }
            ]
        }))
        .send()
        .await
        .context("calling Claude")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!("Claude request failed ({status}): {body}"));
    }

    let message: AnthropicMessage = response.json().await.context("parsing Claude response")?;
    let text = message
        .content
        .into_iter()
        .find_map(|content| content.text)
        .ok_or_else(|| anyhow!("Claude response did not include text content"))?;

    let min_duration = if transcript.duration < 60.0 {
        (transcript.duration * 0.5).max(5.0)
    } else {
        30.0
    };
    parse_candidate_json(&text, min_duration)
}

#[derive(Debug, Deserialize)]
struct OllamaMessage {
    content: String,
}

#[derive(Debug, Deserialize)]
struct OllamaResponse {
    message: OllamaMessage,
}

pub async fn detect_candidates_with_local_llm(
    transcript: &NormalizedTranscript,
    model_name: &str,
) -> Result<Vec<CandidateDraft>> {
    let segments = compact_segments(&transcript.segments);

    let system_instructions = "You are an elite, world-class social media strategist with a track record of generating viral multi-million-view Shorts, TikToks, and Reels. \
Your sole objective is to identify the ABSOLUTE BEST, most highly-engaging, and trend-setting short-form clip candidates from the provided transcript. \
Do NOT pick random or mediocre segments. Be ruthless in your selection. \
Every candidate must have an insanely strong, curiosity-inducing hook in the first 3 seconds to stop the scroll. \
CRITICAL: Each clip candidate MUST have a duration between 30 and 90 seconds (i.e. 'end' minus 'start' must be between 30.0 and 90.0). \
Do NOT return short clips of less than 30 seconds. Combine multiple adjacent sentences to build a meaningful segment of 30-90 seconds. \
Favor highly shareable content: concrete stories, strong opinions, emotional turns, surprising or counter-intuitive claims, clear payoffs, and high-energy/dramatic peaks. \
You MUST identify and return at least 3-10 candidates. Do not return an empty candidates list. \
Ensure the 'start' and 'end' values correspond to actual timestamps in the transcript. Do not output 0.0 for start and end times.";

    let user_content = format!("Transcript:\n{}", segments);

    let response = reqwest::Client::new()
        .post("http://localhost:11434/api/chat")
        .json(&json!({
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": system_instructions,
                },
                {
                    "role": "user",
                    "content": user_content,
                }
            ],
            "stream": false,
            "options": {
                "temperature": 0.2
            },
            "format": {
                "type": "object",
                "properties": {
                    "candidates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "start": { "type": "number" },
                                "end": { "type": "number" },
                                "score": { "type": "number" },
                                "hook": { "type": "string" },
                                "rationale": { "type": "string" }
                            },
                            "required": ["start", "end", "score", "hook", "rationale"]
                        }
                    }
                },
                "required": ["candidates"]
            }
        }))
        .send()
        .await
        .context("calling local Ollama")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!("Local Ollama request failed ({status}): {body}"));
    }

    let res_body: OllamaResponse = response
        .json()
        .await
        .context("parsing local Ollama response")?;
    let min_duration = if transcript.duration < 60.0 {
        (transcript.duration * 0.5).max(5.0)
    } else {
        30.0
    };
    parse_candidate_json(&res_body.message.content, min_duration)
}

fn compact_segments(segments: &[TranscriptSegment]) -> String {
    segments
        .iter()
        .map(|segment| {
            let speaker = segment.speaker.as_deref().unwrap_or("Speaker");
            format!(
                "[{:.2}-{:.2}] {}: {}",
                segment.start, segment.end, speaker, segment.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Find the first balanced JSON object or array inside a model reply.
///
/// Not every model can be pinned to JSON mode — Anthropic via OpenRouter, and
/// local Ollama models, routinely wrap the payload in a sentence or two. Brace
/// matching (skipping over string literals) recovers it without a regex.
fn extract_json_span(text: &str) -> Option<&str> {
    let bytes = text.as_bytes();
    let start = bytes.iter().position(|&b| b == b'{' || b == b'[')?;
    let open = bytes[start];
    let close = if open == b'{' { b'}' } else { b']' };

    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;

    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        if b == b'"' {
            in_string = true;
        } else if b == open {
            depth += 1;
        } else if b == close {
            depth -= 1;
            if depth == 0 {
                return text.get(start..=i);
            }
        }
    }
    None
}

fn parse_candidate_json(text: &str, min_duration: f64) -> Result<Vec<CandidateDraft>> {
    let trimmed = text
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    let val: serde_json::Value = match serde_json::from_str(trimmed) {
        Ok(v) => v,
        Err(direct_err) => {
            // Fall back to carving the JSON out of surrounding prose.
            let span = extract_json_span(trimmed)
                .ok_or_else(|| anyhow!("parsing candidate JSON: {direct_err}"))?;
            serde_json::from_str(span).context("parsing candidate JSON")?
        }
    };

    let candidates_arr = if val.is_array() {
        val.as_array().cloned()
    } else if val.is_object() {
        let mut found_arr = None;
        for key in &[
            "candidates",
            "Candidates",
            "moments",
            "clips",
            "segments",
            "results",
        ] {
            if let Some(arr) = val.get(*key).and_then(|v| v.as_array()) {
                found_arr = Some(arr.clone());
                break;
            }
        }
        if found_arr.is_none() {
            if let Some(obj) = val.as_object() {
                for (_key, value) in obj {
                    if let Some(arr) = value.as_array() {
                        found_arr = Some(arr.clone());
                        break;
                    }
                }
            }
        }
        if found_arr.is_some() {
            found_arr
        } else if val.get("start").is_some() && val.get("end").is_some() {
            Some(vec![val.clone()])
        } else {
            None
        }
    } else {
        None
    };

    let concrete_arr = candidates_arr.ok_or_else(|| {
        anyhow!(
            "Ollama output does not contain a candidates array. Raw output: {}",
            trimmed
        )
    })?;

    let mut drafts = Vec::new();
    for item in &concrete_arr {
        let start = match item.get("start") {
            Some(v) => {
                if let Some(f) = v.as_f64() {
                    f
                } else if let Some(s) = v.as_str() {
                    s.parse::<f64>().unwrap_or(0.0)
                } else if let Some(i) = v.as_i64() {
                    i as f64
                } else {
                    0.0
                }
            }
            None => 0.0,
        };

        let end = match item.get("end") {
            Some(v) => {
                if let Some(f) = v.as_f64() {
                    f
                } else if let Some(s) = v.as_str() {
                    s.parse::<f64>().unwrap_or(0.0)
                } else if let Some(i) = v.as_i64() {
                    i as f64
                } else {
                    0.0
                }
            }
            None => 0.0,
        };

        let mut score = match item.get("score") {
            Some(v) => {
                if let Some(f) = v.as_f64() {
                    f
                } else if let Some(s) = v.as_str() {
                    s.parse::<f64>().unwrap_or(0.8)
                } else if let Some(i) = v.as_i64() {
                    i as f64
                } else {
                    0.8
                }
            }
            None => 0.8,
        };

        if score > 1.0 && score <= 10.0 {
            score /= 10.0;
        } else if score > 10.0 && score <= 100.0 {
            score /= 100.0;
        } else if score > 100.0 {
            score = 1.0;
        } else if score < 0.0 {
            score = 0.0;
        }

        let hook = item
            .get("hook")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let rationale = item
            .get("rationale")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        drafts.push(CandidateDraft {
            start,
            end,
            score,
            hook,
            rationale,
        });
    }

    let mut candidates = drafts
        .clone()
        .into_iter()
        .filter(|candidate| {
            (candidate.end - candidate.start) >= min_duration && !candidate.hook.trim().is_empty()
        })
        .collect::<Vec<_>>();

    if candidates.is_empty() {
        candidates = drafts
            .into_iter()
            .filter(|candidate| {
                (candidate.end - candidate.start) >= 5.0 && !candidate.hook.trim().is_empty()
            })
            .collect::<Vec<_>>();
    }

    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut candidates = suppress_overlaps(candidates, 0.5);
    candidates.truncate(10);
    Ok(candidates)
}

/// Drop candidates that mostly repeat a higher-scoring one.
///
/// Models routinely return the same moment at several different boundaries
/// (65-100, 65-140, 100-140 ...). Rendering each costs a face-tracking pass and
/// an encode to produce near-duplicate clips, so the best-scoring version of an
/// overlapping group wins.
///
/// Overlap is measured against the *shorter* candidate, not the union: a 30s
/// clip wholly inside a 90s one is a duplicate even though it covers only a
/// third of it.
fn suppress_overlaps(sorted_by_score: Vec<CandidateDraft>, max_overlap: f64) -> Vec<CandidateDraft> {
    let mut kept: Vec<CandidateDraft> = Vec::new();

    for candidate in sorted_by_score {
        let duration = candidate.end - candidate.start;
        if duration <= 0.0 {
            continue;
        }

        let duplicates_existing = kept.iter().any(|k| {
            let overlap = candidate.end.min(k.end) - candidate.start.max(k.start);
            if overlap <= 0.0 {
                return false;
            }
            let shorter = duration.min(k.end - k.start);
            shorter > 0.0 && (overlap / shorter) > max_overlap
        });

        if !duplicates_existing {
            kept.push(candidate);
        }
    }

    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_json_wrapped_in_prose() {
        let reply = "Sure! Here are the best moments:\n{\"candidates\":[]}\nHope that helps.";
        assert_eq!(extract_json_span(reply), Some("{\"candidates\":[]}"));
    }

    #[test]
    fn ignores_braces_inside_strings() {
        let reply = "text {\"hook\":\"use } now\",\"n\":1} trailing";
        assert_eq!(
            extract_json_span(reply),
            Some("{\"hook\":\"use } now\",\"n\":1}")
        );
    }

    #[test]
    fn handles_escaped_quotes() {
        let reply = "{\"hook\":\"she said \\\"stop\\\" firmly\"}";
        assert_eq!(extract_json_span(reply), Some(reply));
    }

    #[test]
    fn parses_prose_wrapped_candidates() {
        let reply = "Here you go:\n{\"candidates\":[{\"start\":10.0,\"end\":50.0,\"score\":0.9,\
                     \"hook\":\"h\",\"rationale\":\"r\"}]}";
        let parsed = parse_candidate_json(reply, 30.0).expect("should recover JSON from prose");
        assert_eq!(parsed.len(), 1);
    }

    fn draft(start: f64, end: f64, score: f64) -> CandidateDraft {
        CandidateDraft { start, end, score, hook: "h".into(), rationale: "r".into() }
    }

    #[test]
    fn drops_candidate_contained_in_higher_scoring_one() {
        let out = suppress_overlaps(vec![draft(65.0, 140.0, 0.94), draft(100.0, 140.0, 0.93)], 0.5);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].start, 65.0);
    }

    #[test]
    fn keeps_adjacent_non_overlapping_moments() {
        let out = suppress_overlaps(vec![draft(0.0, 60.0, 0.9), draft(60.0, 120.0, 0.8)], 0.5);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn keeps_slightly_overlapping_distinct_moments() {
        // 10s shared out of a 60s clip is a different moment, not a duplicate.
        let out = suppress_overlaps(vec![draft(0.0, 60.0, 0.9), draft(50.0, 110.0, 0.8)], 0.5);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn highest_score_wins_within_an_overlapping_group() {
        let out = suppress_overlaps(
            vec![draft(140.0, 180.0, 0.97), draft(140.0, 220.0, 0.85), draft(150.0, 175.0, 0.7)],
            0.5,
        );
        assert_eq!(out.len(), 1);
        assert!((out[0].score - 0.97).abs() < 1e-9);
    }

    #[test]
    fn returns_none_when_unbalanced() {
        assert_eq!(extract_json_span("{\"candidates\": ["), None);
    }
}

#[cfg(test)]
mod live_tests {
    use super::*;
    use crate::models::TranscriptSegment;

    fn pregnancy_transcript() -> NormalizedTranscript {
        // Condensed from a real pregnancy-advice interview: the kind of source
        // this pipeline targets.
        let lines: &[(f64, f64, &str)] = &[
            (0.0, 30.0, "Pregnancy is divided into three trimesters. The first trimester is when the major body parts of the baby are forming."),
            (30.0, 65.0, "People say the first trimester is very risky. The answer is maybe yes and maybe no. Uncontrolled medical conditions, wrong dietary choices or unsupervised medications at this time can cause birth defects and even miscarriage."),
            (65.0, 100.0, "Here is the part nobody tells you. Papaya is not dangerous because of the fruit itself. It is the raw, unripe papaya that contains latex, which can trigger contractions. Ripe papaya is completely safe and full of vitamin C."),
            (100.0, 140.0, "The same confusion exists with fish. Everyone says avoid fish in pregnancy. That is wrong. Low mercury fish like salmon and sardines are one of the best things for your baby's brain development. It is the high mercury fish, shark, swordfish, king mackerel, that you must avoid."),
            (140.0, 180.0, "Let me tell you the biggest myth of all. Eating for two. You do not need double the calories. In the first trimester you need zero extra calories. Zero. In the second trimester only three hundred forty extra, and in the third about four hundred fifty."),
            (180.0, 220.0, "Soft cheeses, unpasteurised milk, and deli meats carry listeria risk. Listeria can cross the placenta. This is one of the few things where the warning is genuinely serious and not a myth."),
            (220.0, 260.0, "Caffeine, you can have up to two hundred milligrams a day, which is about one cup of coffee. You do not have to give it up completely, despite what your relatives will tell you."),
        ];
        NormalizedTranscript {
            language: "en".into(),
            duration: 260.0,
            speakers: vec!["Doctor".into()],
            words: vec![],
            segments: lines
                .iter()
                .map(|(s, e, t)| TranscriptSegment {
                    start: *s,
                    end: *e,
                    speaker: Some("Doctor".into()),
                    text: (*t).into(),
                })
                .collect(),
        }
    }

    /// Validates whatever is in ANTHROPIC_API_KEY against the real API through
    /// the app's own provider path. Run with:
    ///   cargo test --lib -- --ignored --nocapture live_anthropic


    /// Hits the live Anthropic API with whichever credential `.env` holds,
    /// proving the app's own moment detection works with a subscription token.
    /// Run: cargo test --lib -- --ignored --nocapture live_anthropic
    #[tokio::test]
    #[ignore]
    async fn live_anthropic_returns_candidates() {
        let _ = dotenvy::from_path("../.env");
        let key = std::env::var("ANTHROPIC_API_KEY")
            .or_else(|_| std::env::var("ANTHROPIC_OAUTH_TOKEN"))
            .expect("no Anthropic credential in .env");
        let kind = if key.starts_with("sk-ant-oat") { "subscription token" } else { "API key" };
        println!("\n=== Anthropic via {} ({}) ===", kind,
                 std::env::var("ANTHROPIC_MODEL").unwrap_or_default());

        let out = detect_candidates_with_claude(&pregnancy_transcript(), &key)
            .await
            .expect("Anthropic call failed");
        assert!(!out.is_empty(), "no candidates returned");
        for c in &out {
            println!("  [{:6.1}s -> {:6.1}s] {:.2}  {}", c.start, c.end, c.score, c.hook);
            assert!(c.end > c.start);
        }
    }

    /// Hits the live OpenRouter API. Costs a few cents.
    /// Run with: cargo test --lib -- --ignored --nocapture live_openrouter
    #[tokio::test]
    #[ignore]
    async fn live_openrouter_claude_returns_candidates() {
        let _ = dotenvy::from_path("../.env");
        let key = std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY not set");
        let model = std::env::var("OPENROUTER_MODEL")
            .unwrap_or_else(|_| "anthropic/claude-sonnet-4.5".to_string());

        let out = detect_candidates_with_openrouter(&pregnancy_transcript(), &key, Some(&model))
            .await
            .expect("OpenRouter call failed");

        assert!(!out.is_empty(), "no candidates returned");
        println!("\n=== {} returned {} candidates ===", model, out.len());
        for c in &out {
            println!(
                "  [{:6.1}s -> {:6.1}s] score {:.2}\n    hook: {}",
                c.start, c.end, c.score, c.hook
            );
            assert!(c.end > c.start, "candidate has non-positive duration");
        }
    }
}
