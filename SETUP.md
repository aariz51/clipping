# AutoShorts — local setup notes

Changes made on top of upstream `JayWebtech/autoshorts`, plus what you need to
supply before the pipeline will run end to end.

---

## 1. What was added

### Speaker-aware cropping (the big one)

Upstream crops every clip to a fixed centre window:

```
crop=w='2*trunc(min(iw,ih*9/16)/2)':h='2*trunc(min(ih,iw*16/9)/2)'
```

On a 640x360 source that window is x=219..421. On the pregnancy interview used
for testing, the doctor's face sat at x≈436 — **entirely outside the crop**. The
stock render produced clips of an empty chair.

New behaviour: before each render, a small OpenCV sidecar samples the clip,
tracks the primary speaker's face, and hands ffmpeg either a fixed offset or a
time-varying `crop` expression that follows them.

- `src-tauri/assets/facetrack.py` — the tracker (YuNet face detection)
- `src-tauri/assets/face_detection_yunet_2023mar.onnx` — detector weights (227KB)
- `src-tauri/src/facetrack.rs` — resolves Python, runs the sidecar, parses the plan

Design notes:

- **Fails soft.** Any problem — no Python, no OpenCV, no faces found, low
  detection coverage — returns `None` and the original centre crop is used. Face
  tracking can never break a render.
- **Static when possible.** If the speaker barely moves, a fixed offset is
  emitted rather than a moving crop, which avoids drift and reads as a
  deliberate framing choice.
- **Cut-aware.** Jumps larger than 25% of the crop width are treated as camera
  cuts. Smoothing never averages across one, and the crop *snaps* at the cut
  instead of sliding into place over the next second.
- **Jitter control.** A centred moving average (~1.5s) plus a deadzone keeps the
  frame from vibrating with every small head movement.
- Assets are embedded in the binary (`include_str!` / `include_bytes!`) and
  written to the app data dir on first run, so the packaged `.app` works without
  the source tree.

Cost: roughly 1 second of tracking per 60 seconds of clip.

### Captions that work without `drawtext`

Your ffmpeg (Homebrew 8.1.2) is a reduced build with **no `drawtext` filter** —
it lacks libfreetype, libass, fontconfig and harfbuzz. Upstream passes drawtext
filters unconditionally, so every captioned render aborted with
`No such filter: 'drawtext'`. Choosing a caption style in the import dialog
could never have helped; the filter simply does not exist in this build.

A full-featured ffmpeg is not available as a Homebrew bottle for macOS 26 on
arm64 (source build only, 30-60 min), and relying on one would leave captions
hostage to whichever ffmpeg happens to be installed.

Instead captions are now rasterised with Pillow and composited using
`overlay`, a **core filter present in every ffmpeg build**:

- `src-tauri/assets/captions.py` — renders one transparent PNG per phrase and
  an ffmpeg concat list with per-frame durations
- `src-tauri/src/captions.rs` — chunks words into pairs, drives the sidecar,
  cleans up the frame directory on drop

Details worth knowing:

- **Punctuation survives.** The drawtext path filtered text to alphanumerics,
  so `don't` rendered as `DONT`. The overlay path keeps it.
- All seven original styles are reproduced, so the look is unchanged.
- Identical phrases reuse one PNG, and silent stretches become explicit
  transparent entries because concat has no concept of an absent frame.
- Text wraps at 86% of frame width and scales with resolution.
- Selection order: overlay renderer, then `drawtext` if this ffmpeg has it,
  then no captions. A clip never fails because captions could not be drawn.

The **Captions** status lamp now reflects either renderer being available.

### ffmpeg argument ordering (regression guard)

ffmpeg option *position* carries meaning: anything placed before an `-i` binds
to that input. When the caption overlay was added as a second input, `-t` ended
up ahead of it and was silently reinterpreted as an *input* limit — the output
lost its duration cap and encoded the entire source. A 96-second candidate
rendered as a 49-minute, 1.9 GB file.

`build_render_command` now assembles inputs first and every output option
after, and is a pure function so the ordering can be asserted. Four tests in
`media.rs` cover it, including one that fails if `-t` ever precedes the last
`-i` again.

### ffmpeg binary override

`AUTOSHORTS_FFMPEG` and `AUTOSHORTS_FFPROBE` override which binary is used.
This lets AutoShorts use a fully-featured ffmpeg without relinking the system
one that your Remotion render pipeline depends on.

### OpenRouter fixes (needed for Claude-via-OpenRouter)

Two upstream bugs made the OpenRouter + Anthropic combination unusable:

1. **`response_format: json_object` was sent unconditionally.** Anthropic models
   on OpenRouter reject that parameter, so the first analysis run would fail.
   It is now sent only to models that support it, and a rejection retries
   automatically without it.
2. **The candidate parser required a pure-JSON reply.** Without JSON mode,
   models often answer `Here are the best moments: {...}`, which failed to
   parse. `extract_json_span` now brace-matches the payload out of surrounding
   prose, correctly skipping braces inside string literals and escaped quotes.

Both changes benefit every provider. Covered by unit tests in `llm.rs`.

### Overlapping-candidate suppression

Models return the same moment at several boundaries. A real Claude run on a
pregnancy transcript produced `65-100`, `65-140`, `100-140`, `100-180`,
`140-180`, `140-220` — six entries for three actual moments. Upstream sorted by
score and kept the top 10, so you would render near-duplicate clips, each
costing a face-tracking pass and an encode.

`suppress_overlaps` now keeps the highest-scoring version of any overlapping
group. Overlap is measured against the **shorter** candidate, so a 30s clip
sitting inside a 90s one counts as a duplicate even though it covers only a
third of it. Threshold is 50%; genuinely adjacent moments survive.

On that live run it cut 10 candidates to 6 distinct, non-overlapping clips.

### Live end-to-end check

Verifies the real OpenRouter path, key included. Costs about a cent:

```bash
cd src-tauri
cargo test --lib -- --ignored --nocapture live_openrouter
```

### Shared Python resolver (`src-tauri/src/pyenv.rs`)

Upstream looked for Whisper only on a bare `python3` from PATH. Optional
dependencies live in `~/autoshorts/.venv` (Homebrew Python is externally
managed and refuses `pip install`), so that lookup would never have found it
and offline transcription would have reported "not installed" forever.

Interpreter resolution is now shared by face tracking and transcription, and
probes in order: `AUTOSHORTS_PYTHON`, any `.venv` beside the binary or source
tree, then PATH. The `whisper` CLI is resolved the same way rather than being
invoked by bare name.

### Status indicators

The status bar gained **Face Track** and **Captions** lamps, so a missing
dependency is visible instead of silently degrading output.

---

## 2. Dependencies already satisfied on this machine

| Tool | Status |
|---|---|
| ffmpeg / ffprobe | installed (see caveat below) |
| yt-dlp | installed |
| Rust / cargo | 1.95.0 |
| Node | v26.3.0 |
| Python venv + OpenCV | created at `~/autoshorts/.venv` |
| Local Whisper | installed in the venv (module + CLI) |

**Whisper speed, measured on this Mac:** 60s of audio took **48s** with the
`base` model — roughly 0.8x real time, CPU-only. A 1-hour source video is
therefore about 45-50 minutes of transcription. Fine to leave running, painful
if you are iterating. If that becomes the bottleneck, either switch to a
Deepgram key or try `faster-whisper`, which is several times quicker on the
same hardware. Word-level timestamps are produced correctly (121 words over the
60s test), which is what caption sync and clip boundaries depend on.

---

## 3. What you still need to supply

### API key — OpenRouter only

Transcription is handled locally (see below), so the single remaining
requirement is one LLM key. Put it in `.env` in the repo root — that file is
gitignored:

```env
OPENROUTER_API_KEY=sk-or-...
LLM_PROVIDER=openrouter
OPENROUTER_MODEL=anthropic/claude-sonnet-4.5
```

`.env` is already populated from `OPENROUTER_KEY` in `~/.zshrc` and chmod 600.
Note the name differs: the shell exports `OPENROUTER_KEY`, the app reads
`OPENROUTER_API_KEY`.

`OPENROUTER_MODEL` accepts any OpenRouter model id. Without it the default is
`google/gemini-2.5-flash`. Other providers (DeepSeek, Gemini, OpenAI, Groq,
Anthropic direct, Ollama) remain wired if you ever want to switch.

Placeholder keys were removed deliberately. The app tests providers with
`env::var(...).is_ok()`, so a leftover `DEEPGRAM_API_KEY=your_key_here` — or
even an empty assignment — makes it report that provider as configured.

Rough cost per source video (transcript in, candidates out):

| Model | ~1-hour video |
|---|---|
| `anthropic/claude-sonnet-4.5` | ~$0.08 |
| `anthropic/claude-opus-5` | ~$0.13 |
| `google/gemini-2.5-flash` | ~$0.01 |

Sensible pattern: Gemini Flash while iterating on the pipeline, Claude for the
runs whose output you will actually post.

### Captions (optional)

Only needed if you want burned-in subtitles. Get an ffmpeg built with
libfreetype, then point AutoShorts at it without touching your system install:

```bash
brew install homebrew-ffmpeg/ffmpeg/ffmpeg   # already tapped
export AUTOSHORTS_FFMPEG=/opt/homebrew/opt/homebrew-ffmpeg/ffmpeg/bin/ffmpeg
```

Verify with `ffmpeg -filters | grep drawtext`. Until then the Captions lamp
stays dark and clips render clean (no subtitles).

---

## 4. Running it

```bash
cd ~/autoshorts
npm run tauri:dev     # development, live reload
npm run tauri:build   # packaged .app + .dmg under src-tauri/target/release/bundle/
```

---

## 5. Verifying face tracking by itself

Useful for checking a specific source video before running the whole pipeline:

```bash
~/autoshorts/.venv/bin/python src-tauri/assets/facetrack.py \
  --video /path/to/video.mp4 \
  --model src-tauri/assets/face_detection_yunet_2023mar.onnx \
  --start 20 --end 80
```

Read the JSON it prints:

- `mode: dynamic | static` — a plan was produced
- `mode: none` — centre crop will be used; `reason` says why
- `coverage` — fraction of sampled frames with a detected face. Below `0.15` the
  track is rejected as untrustworthy. Low coverage usually means the source is
  b-roll, screen recording, or heavily shot-reverse edited.
- `cuts` — camera cuts detected in the range

---

## 6. Known limits

- Tracks **one** speaker — the largest, most confident face, biased toward
  whoever was already being tracked. Two-person interviews follow the dominant
  face rather than cutting between speakers; it will not split-screen them.
- Detection runs at 4 fps on a 480px-wide downscale. Fast head motion between
  samples is smoothed rather than followed exactly.
- Talking-head footage is the good case. Slideshows, screen recordings and
  heavy b-roll will fall back to centre crop, which is the correct outcome.
- Upstream's caption text is stripped to alphanumerics, so `don't` renders as
  `DONT`. Untouched here — it only matters once you have a drawtext-capable
  ffmpeg.

---

## 7. Onboarding wizard

Neither original card fitted a local-Whisper + OpenRouter setup:

- **Cloud APIs** hard-required a Deepgram key and had no OpenRouter field at
  all, despite the backend supporting the provider.
- **Fully Offline** auto-installs Ollama and pulls a multi-GB model, then pins
  the LLM to `local`.

The cloud step now offers an OpenRouter key + model, makes Deepgram optional
whenever local Whisper is detected, and accepts a key already present in `.env`
instead of demanding it be retyped.

**Choose "Cloud APIs"**, then:

1. Leave *Use local Whisper* ticked (pre-ticked when Whisper is installed).
2. Leave Deepgram blank — the field disables itself.
3. Leave OpenRouter blank to use the `.env` key, or paste one to override.
4. Optionally set a model, e.g. `anthropic/claude-sonnet-4.5`.

To redo this later: **API Settings → Reset App Configuration & Onboarding**.
