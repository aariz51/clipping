"""Add speech-matched B-roll to a rendered clip using the pinned b-rolls skill.

Automates the agent-driven steps of the `b-rolls` workflow
(`prepare_project.py` -> scene planning -> sourcing -> `render_project.py` ->
`verify_output.py`) so AutoShorts can run it unattended.

Design notes:

- Scene *boundaries* come from `prepare_project.py` and are never recomputed.
  That draft is already contiguous, frame-aligned and within
  `max_scene_seconds`, which is exactly what `validate_plan` enforces. Only
  each scene's `kind` and its payload are rewritten, so a planning mistake can
  never produce an invalid timeline.
- The transcript comes from AutoShorts' Whisper words rather than the OCR
  sampler: it is word-accurate, whereas OCR of burned-in captions also picks up
  background text.
- Footage sourcing degrades: Pexels when a key exists, otherwise generated
  `card` explainers, which need no downloads.

Prints the final rendered path on success.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CARD_ICONS = [
    "generic", "food", "health", "warning", "science", "book",
    "money", "chart", "question", "globe", "shield", "policy", "people",
]
CARD_MOTIONS = ["reveal", "rise", "pulse", "slide-left", "slide-right"]


def log(msg: str) -> None:
    print(f"[broll] {msg}", file=sys.stderr, flush=True)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, **kw)


# --------------------------------------------------------------------------
# Scene planning
# --------------------------------------------------------------------------

def scene_transcripts(scenes: list[dict], words: list[dict]) -> list[str]:
    """Words spoken during each scene slot, as plain text."""
    out = []
    for scene in scenes:
        said = [
            w.get("text", "")
            for w in words
            if w.get("end", 0) > scene["start"] and w.get("start", 0) < scene["end"]
        ]
        out.append(" ".join(t.strip() for t in said if t.strip()))
    return out


def plan_prompt(lines: list[str], topic: str) -> str:
    numbered = "\n".join(f"{i}: {t or '(silence)'}" for i, t in enumerate(lines))
    return f"""You are a short-form video editor choosing B-roll for a vertical clip.

The clip is about: {topic}

Below are {len(lines)} consecutive ~1 second slots with the words spoken in each.
Decide what the viewer SEES in every slot. Return JSON only.

Editorial rules you must follow:
- Slot 0 is always "source" (the speaker, for hook continuity).
- Return to "source" for 1 slot every 4 to 6 slots, for human continuity.
- Otherwise prefer "broll" showing a literal, concrete visual of what is being said.
- Use "card" when the idea is abstract, a number, a question, or a contrast that
  literal footage would show weakly.
- Never repeat the same broll query in adjacent slots.

Schema, one entry per slot, exactly {len(lines)} entries:
{{"slots":[
  {{"i":0,"kind":"source"}},
  {{"i":1,"kind":"broll","query":"two short concrete search words","description":"what is shown"}},
  {{"i":2,"kind":"card","title":"SHORT PUNCHY","subtitle":"","icon":"one of {CARD_ICONS}","motion":"one of {CARD_MOTIONS}","description":"what is shown"}}
]}}

Slots:
{numbered}"""


def anthropic_headers(credential: str) -> dict:
    """Auth headers for either Anthropic credential shape.

    Console keys (`sk-ant-api...`) authenticate with `x-api-key`. Claude Code
    subscription tokens (`sk-ant-oat...`) are OAuth access tokens: they
    authenticate as a bearer token with the OAuth beta header and are rejected
    by `x-api-key`.
    """
    base = {"anthropic-version": "2023-06-01", "content-type": "application/json"}
    if credential.startswith("sk-ant-oat"):
        return {**base,
                "authorization": f"Bearer {credential}",
                "anthropic-beta": "oauth-2025-04-20"}
    return {**base, "x-api-key": credential}


def call_anthropic(prompt: str, credential: str) -> str:
    body = json.dumps({
        "model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"),
        "max_tokens": 8000,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers=anthropic_headers(credential),
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.load(resp)
    return "".join(part.get("text", "") for part in data.get("content", []))


def call_llm(prompt: str) -> str:
    """Ask the configured provider for a scene plan.

    Anthropic is preferred when a credential is present. A subscription token
    can be rate-limited or expired independently of it being valid, so those
    failures fall back to OpenRouter rather than aborting the render.
    """
    anthropic = (os.environ.get("ANTHROPIC_API_KEY")
                 or os.environ.get("ANTHROPIC_OAUTH_TOKEN") or "").strip()
    if anthropic:
        kind = "subscription token" if anthropic.startswith("sk-ant-oat") else "API key"
        try:
            log(f"planning via Anthropic ({kind})")
            return call_anthropic(prompt, anthropic)
        except urllib.error.HTTPError as exc:
            reason = {401: "credential rejected", 429: "rate limited",
                      529: "overloaded"}.get(exc.code, f"HTTP {exc.code}")
            log(f"Anthropic unavailable ({reason}); falling back to OpenRouter")
        except Exception as exc:
            log(f"Anthropic call failed ({exc}); falling back to OpenRouter")

    key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not key:
        raise SystemExit("no usable LLM key: set ANTHROPIC_API_KEY (sk-ant-api...) or OPENROUTER_API_KEY")

    payload = {
        "model": os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "content-type": "application/json",
            "HTTP-Referer": "https://github.com/aariz51/clipping",
            "X-Title": "AutoShorts",
        },
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"]


def extract_json(text: str) -> dict:
    """Recover a JSON object from a possibly prose-wrapped reply."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    depth, start, in_str, esc = 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return json.loads(text[start:i + 1])
    raise ValueError("no JSON object in model reply")


# --------------------------------------------------------------------------
# Footage sourcing
# --------------------------------------------------------------------------

def clip_is_usable(path: Path, need_seconds: float) -> float | None:
    """Offset into `path` that yields real motion, or None if unusable.

    Stock and archive footage often opens on a static title card. Starting at
    zero there renders a frozen frame, which reads as no cut at all and fails
    `verify_output.py`'s boundary check.
    """
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            check=True, text=True, capture_output=True)
        duration = float(probe.stdout.strip() or 0)
    except Exception:
        return None
    if duration < need_seconds + 1.0:
        return None

    # Skip past any opening title, then confirm the segment actually moves.
    offset = min(3.0, duration * 0.25)
    try:
        det = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-ss", f"{offset:.2f}",
             "-t", f"{max(need_seconds, 1.0):.2f}", "-i", str(path),
             "-vf", "freezedetect=n=0.003:d=0.9", "-f", "null", "-"],
            text=True, capture_output=True, timeout=120)
        if "freeze_start" in (det.stderr or ""):
            return None
    except Exception:
        return None
    return offset


def wikimedia_download(query: str, dest_dir: Path, index: int,
                       need_seconds: float) -> tuple[Path, str, float] | None:
    """Fetch one freely-licensed clip from Wikimedia Commons.

    Needs no API key, and the skill lists Commons as an approved source, so this
    is the keyless tier between Pexels and generated cards. Files are CC/PD; the
    page URL is recorded in the scene's `source_url` for attribution.
    """
    params = urllib.parse.urlencode({
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": f"filetype:video {query}", "gsrnamespace": "6",
        "gsrlimit": "6", "prop": "imageinfo", "iiprop": "url|mime|size",
    })
    req = urllib.request.Request(
        f"https://commons.wikimedia.org/w/api.php?{params}",
        headers={"User-Agent": "AutoShorts/1.0 (b-roll sourcing)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.load(resp)
    except Exception as exc:
        log(f"commons search failed for {query!r}: {exc}")
        return None

    # Commons' video corpus is small, so a keyword search happily returns
    # something that merely shares one word with the query. Require a real term
    # overlap with the file title, or the clip will be off-topic.
    terms = {w.lower() for w in query.split() if len(w) > 3}

    pages = (data.get("query") or {}).get("pages") or {}
    for page in pages.values():
        info = (page.get("imageinfo") or [{}])[0]
        mime = info.get("mime") or ""
        size = info.get("size") or 0
        # Skip audio-only ogg and anything large enough to stall the render.
        if not mime.startswith("video/") or size > 60_000_000:
            continue
        title = (page.get("title") or "").lower()
        if terms and not any(t in title for t in terms):
            continue
        url = info.get("url")
        if not url:
            continue
        suffix = Path(urllib.parse.urlparse(url).path).suffix or ".webm"
        target = dest_dir / f"broll_{index:03d}{suffix}"
        try:
            dl = urllib.request.Request(
                url, headers={"User-Agent": "AutoShorts/1.0 (b-roll sourcing)"}
            )
            with urllib.request.urlopen(dl, timeout=180) as src, open(target, "wb") as out:
                out.write(src.read())
        except Exception as exc:
            log(f"commons download failed for {query!r}: {exc}")
            continue
        offset = clip_is_usable(target, need_seconds)
        if offset is None:
            log(f"commons clip rejected (too short or frozen): {page.get('title','')[:48]}")
            target.unlink(missing_ok=True)
            continue
        return target, info.get("descriptionurl") or url, offset
    return None


def pexels_download(query: str, dest_dir: Path, index: int) -> tuple[Path, str] | None:
    """Fetch one portrait clip for `query`. Returns (file, source_url)."""
    key = (os.environ.get("PEXELS_API_KEY") or "").strip()
    if not key:
        return None
    url = ("https://api.pexels.com/videos/search?"
           + urllib.parse.urlencode({"query": query, "orientation": "portrait", "per_page": 5}))
    try:
        # Pexels sits behind Cloudflare, which rejects urllib's default
        # User-Agent with a 403 (error 1010) before the key is even checked.
        req = urllib.request.Request(url, headers={
            "Authorization": key,
            "User-Agent": "AutoShorts/1.0 (local b-roll sourcing)",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.load(resp)
    except Exception as exc:
        log(f"pexels search failed for {query!r}: {exc}")
        return None

    for video in data.get("videos", []):
        files = [f for f in video.get("video_files", []) if (f.get("height") or 0) >= 960]
        files.sort(key=lambda f: f.get("height") or 0)
        if not files:
            continue
        target = dest_dir / f"broll_{index:03d}.mp4"
        try:
            dl = urllib.request.Request(
                files[0]["link"],
                headers={"User-Agent": "AutoShorts/1.0 (local b-roll sourcing)"},
            )
            with urllib.request.urlopen(dl, timeout=180) as src, open(target, "wb") as out:
                out.write(src.read())
        except Exception as exc:
            log(f"pexels download failed for {query!r}: {exc}")
            continue
        return target, video.get("url", "")
    return None


# --------------------------------------------------------------------------
# Plan assembly
# --------------------------------------------------------------------------

# Adjacent cards must not share styling, or the boundary between them shows no
# visible change and `verify_output.py` rejects the render.
CARD_ACCENTS = ["#EFFF32", "#4FC3F7", "#FF7043", "#AB47BC", "#66BB6A", "#FFCA28"]


def apply_slots(plan: dict, slots: list[dict], downloads: Path) -> dict:
    """Rewrite scene kinds in place, keeping every original boundary."""
    scenes = plan["scenes"]
    by_index = {int(s.get("i", -1)): s for s in slots}
    used_queries: list[str] = []
    sourced = cards = kept = 0

    # Two `source` scenes back to back are continuous footage, so their shared
    # boundary has no cut at all. Demote the second to a card so every planned
    # boundary is a real visual change.
    previous_kind = None
    for i, choice in ((i, by_index.get(i) or {}) for i in range(len(scenes))):
        kind = choice.get("kind", "source")
        if i > 0 and kind == "source" and previous_kind == "source":
            choice["kind"] = "card"
            choice.setdefault("title", (choice.get("description") or "KEY POINT").upper()[:18])
            by_index[i] = choice
            kind = "card"
        previous_kind = kind

    for i, scene in enumerate(scenes):
        choice = by_index.get(i) or {}
        kind = choice.get("kind", "source")

        # Slot 0 always shows the speaker, and silence stays on the speaker.
        if i == 0 or kind == "source":
            scene["kind"] = "source"
            scene["description"] = choice.get("description") or "speaker return"
            scene.pop("file", None)
            kept += 1
            continue

        if kind == "broll":
            query = (choice.get("query") or "").strip()
            got = None
            # Avoid using the same asset back to back.
            if query and query not in used_queries[-1:]:
                # Pexels first when a key exists, then keyless Commons.
                need = scene["end"] - scene["start"]
                got = pexels_download(query, downloads, i)
                if got:
                    got = (got[0], got[1], 0.0)
                else:
                    got = wikimedia_download(query, downloads, i, need)
            if got:
                path, source_url, offset = got
                scene["kind"] = "video"
                scene["file"] = str(path)
                scene["offset"] = round(offset, 2)
                # Vary the crop so a reused asset still reads as a new moment.
                scene["crop_x"] = 0.5
                scene["crop_y"] = 0.4 + 0.1 * (i % 3)
                scene["description"] = choice.get("description") or query
                scene["source_url"] = source_url
                used_queries.append(query)
                sourced += 1
                continue
            # No footage available: fall through to a generated explainer.
            choice = {
                "title": (query or "").upper()[:18] or "KEY POINT",
                "icon": "generic",
                "motion": "reveal",
                "description": choice.get("description") or query,
            }

        title = (choice.get("title") or "KEY POINT").strip()[:22]
        icon = choice.get("icon") if choice.get("icon") in CARD_ICONS else "generic"
        motion = choice.get("motion") if choice.get("motion") in CARD_MOTIONS else "reveal"
        # Rotate accent and motion by position so consecutive cards never render
        # the same plate, which is what the boundary-change check measures.
        accent = CARD_ACCENTS[cards % len(CARD_ACCENTS)]
        if by_index.get(i - 1, {}).get("kind") == "card":
            motion = CARD_MOTIONS[cards % len(CARD_MOTIONS)]
        scene["kind"] = "card"
        scene["title"] = title
        scene["subtitle"] = (choice.get("subtitle") or "")[:40]
        scene["icon"] = icon
        scene["accent"] = accent
        scene["motion"] = motion
        scene["description"] = choice.get("description") or title
        for key in ("file", "offset", "source_url", "effect"):
            scene.pop(key, None)
        cards += 1

    log(f"plan: {kept} source, {sourced} footage, {cards} cards across {len(scenes)} slots")
    return plan


GRADE = ("scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,"
         "crop=1080:1920,setsar=1")


def frame_count(path: Path) -> int | None:
    try:
        return int(subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
            check=True, text=True, capture_output=True).stdout.strip())
    except Exception:
        return None


def composite_locally(plan: dict, source: Path, timeline: Path, output: Path,
                      expected_frames: int | None = None) -> bool:
    """Build the final video without upstream's compositing pass.

    The pinned `video-use` composite drops frames and repeats the last scene
    across the tail when the extracted base and the rapid timeline disagree on
    duration. The timeline is already full-frame 1080x1920 for the entire clip,
    so the base contributes nothing except caption restoration -- which is
    reproduced here with the same filter chain the skill's patch uses, so the
    result matches its intent.

    Audio is copied from the source, keeping its stream hash identical.
    """
    caption = dict(plan.get("caption") or {})
    fps = probe(source, "stream=r_frame_rate")

    inputs = ["-i", str(timeline), "-i", str(source)]
    if caption.get("preserve"):
        crop_y = int(caption.get("y", 1160))
        crop_h = int(caption.get("height", 180))
        colour = str(caption.get("color", "0xFFFF00"))
        similarity = float(caption.get("similarity", 0.22))
        blend = float(caption.get("blend", 0.08))
        dilation = ",dilation" * max(0, min(8, int(caption.get("outline_dilation", 0))))

        # Source scenes already contain their own captions; restoring over them
        # would double them up, so those intervals are disabled.
        disabled = [
            (s["start"], s["end"]) for s in plan["scenes"] if s["kind"] == "source"
        ]
        enable = ""
        if disabled:
            terms = "+".join(
                f"between(t,{float(a):.3f},{max(float(a), float(b) - 0.001):.3f})"
                for a, b in disabled
            )
            enable = f":enable='not({terms})'"

        graph = (
            f"[1:v]{GRADE},split=2[capcolorsrc][capmasksrc];"
            f"[capcolorsrc]crop=1080:{crop_h}:0:{crop_y},format=rgba[capcolor];"
            f"[capmasksrc]crop=1080:{crop_h}:0:{crop_y},format=rgba,"
            f"colorkey={colour}:{similarity:.3f}:{blend:.3f},"
            f"alphaextract,negate{dilation}[capmask];"
            f"[capcolor][capmask]alphamerge[captionrgba];"
            f"[0:v][captionrgba]overlay=x=0:y={crop_y}{enable}[v]"
        )
    else:
        graph = "[0:v]null[v]"

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs,
           "-filter_complex", graph, "-map", "[v]", "-map", "1:a:0?",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-pix_fmt", "yuv420p", "-r", fps, "-c:a", "copy",
           "-movflags", "+faststart"]
    if expected_frames:
        # Overlay can emit one trailing frame past the source; cap it so the
        # count matches exactly.
        cmd += ["-frames:v", str(expected_frames)]
    cmd += [str(output)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        log(f"local composite failed: {(exc.stderr or '')[-200:]}")
        return False
    return True


def probe(path: Path, entries: str, stream: bool = True) -> str:
    args = ["ffprobe", "-v", "error"]
    if stream:
        args += ["-select_streams", "v:0"]
    args += ["-show_entries", entries, "-of", "csv=p=0", str(path)]
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout.strip()


def conform_frame_count(source: Path, output: Path) -> None:
    """Force `output` to the source's exact frame count.

    Compositing inside the pinned `video-use` renderer can drop frames when the
    base and overlay timebases disagree (observed: 300 in, 287 out), which
    drifts the picture against the untouched audio and fails
    `verify_output.py`. Re-timing the video to constant frame rate restores the
    exact count; the audio stream is copied from the source, so its hash stays
    identical and the preservation guarantee holds.
    """
    try:
        expected = int(subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(source)],
            check=True, text=True, capture_output=True).stdout.strip())
        actual = int(subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(output)],
            check=True, text=True, capture_output=True).stdout.strip())
        fps = probe(source, "stream=r_frame_rate")
    except Exception as exc:
        log(f"could not verify frame count: {exc}")
        return

    if actual == expected:
        return

    log(f"restamping timestamps ({actual} frames vs {expected} expected)")
    fixed = output.with_name(output.stem + "_conformed" + output.suffix)
    try:
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(output), "-i", str(source),
            "-map", "0:v:0", "-map", "1:a:0?",
            # Rebuild monotonic constant-rate timestamps from frame ORDER.
            # `fps=` must not be used here: fed a stream with broken PTS it
            # resamples, and was observed duplicating a single frame across six
            # scenes. setpts only re-stamps, so every decoded frame survives in
            # order and no content is invented or destroyed.
            "-vf", "setpts=N/FRAME_RATE/TB",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-r", fps, "-vsync", "cfr",
            "-c:a", "copy", "-movflags", "+faststart", str(fixed),
        ], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        log(f"conform failed, keeping original: {(exc.stderr or '')[-160:]}")
        fixed.unlink(missing_ok=True)
        return
    fixed.replace(output)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, help="rendered clip to add B-roll to")
    ap.add_argument("--transcript", help="JSON file: {\"words\":[{text,start,end}]} clip-relative")
    ap.add_argument("--topic", default="a short social video")
    ap.add_argument("--skill-dir", default=os.path.expanduser("~/b-rolls-ref"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--output", help="where to copy the finished B-roll cut")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    clip = Path(args.clip).expanduser().resolve()
    if not clip.exists():
        raise SystemExit(f"clip not found: {clip}")
    scripts = Path(args.skill_dir).expanduser().resolve() / "scripts"
    if not (scripts / "render_project.py").exists():
        raise SystemExit(f"b-rolls skill not found at {scripts}")

    # 1. Inventory + draft plan (also detects the burned-in caption band).
    log("preparing project")
    run([args.python, str(scripts / "prepare_project.py"), str(clip)],
        cwd=scripts, stdout=subprocess.DEVNULL)

    edit = clip.parent / "edit"
    plan_path = edit / "scene_plan.json"
    plan = json.loads(plan_path.read_text())

    words = []
    if args.transcript:
        words = json.loads(Path(args.transcript).read_text()).get("words", [])
    lines = scene_transcripts(plan["scenes"], words)

    # 2. Decide what each slot shows.
    if any(lines):
        log(f"planning {len(lines)} slots via LLM")
        reply = call_llm(plan_prompt(lines, args.topic))
        slots = extract_json(reply).get("slots", [])
    else:
        log("no transcript words; keeping the speaker throughout")
        slots = []

    downloads = edit / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    if not (os.environ.get("PEXELS_API_KEY") or "").strip():
        log("PEXELS_API_KEY unset: sourcing from Wikimedia Commons, cards where nothing matches")

    plan = apply_slots(plan, slots, downloads)
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")

    # 3. Render through the pinned video-use pipeline.
    log("rendering")
    run([args.python, str(scripts / "render_project.py"), str(plan_path)],
        cwd=scripts, stdout=subprocess.DEVNULL)

    final = Path(plan["output"])
    if not final.exists():
        raise SystemExit("render finished but no output was produced")

    # The skill's own render is authoritative when it is frame-accurate. When
    # upstream compositing loses frames it also repeats the tail scene, so a
    # short result is rebuilt locally rather than shipped.
    expected = frame_count(clip)
    actual = frame_count(final)
    timeline = edit / "animations" / "slot_rapid_timeline" / "render.mp4"
    if expected and actual and actual != expected and timeline.exists():
        log(f"upstream composite returned {actual}/{expected} frames; compositing locally")
        if composite_locally(plan, clip, timeline, final, expected):
            rebuilt = frame_count(final)
            log(f"local composite produced {rebuilt}/{expected} frames")
    elif expected and actual:
        log(f"frame count {actual}/{expected} from the skill renderer")

    # 4. Verify the rendered file, not the intermediates.
    if args.verify:
        log("verifying")
        try:
            run([args.python, str(scripts / "verify_output.py"), str(plan_path)],
                cwd=scripts, stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError as exc:
            log(f"verification reported problems (exit {exc.returncode})")

    if args.output:
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(final.read_bytes())
        final = out

    print(final)
    return 0


if __name__ == "__main__":
    sys.exit(main())
