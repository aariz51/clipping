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
- Only real footage is used. Pexels first, then Wikimedia Commons as a keyless
  fallback; a beat with no usable footage holds on the speaker rather than
  substituting a generated graphic.

Prints the final rendered path on success.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

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
- Every other slot is "broll": real stock footage showing a literal, concrete
  visual of what is being said.
- The query must describe something filmable. Prefer physical nouns a camera can
  see -- a product, a place, an action -- over abstract ideas.
- Write queries for objects, food, packaging, documents, machinery, buildings and
  landscapes. Do NOT ask for people: no "man", "woman", "person", "shopper",
  "doctor", "family", "crowd". Footage containing people is discarded before it
  reaches the edit, so a query about people simply wastes the slot. Say
  "grocery shelves" rather than "shopper in aisle", "hands chopping vegetables"
  rather than "chef cooking".
- Never repeat the same broll query in adjacent slots.

Schema, one entry per slot, exactly {len(lines)} entries:
{{"slots":[
  {{"i":0,"kind":"source"}},
  {{"i":1,"kind":"broll","query":"two short concrete search words","description":"what is shown"}}
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


# A rate limit is a "wait", not a "give up". A batch run plans a scene list per
# clip back to back, which is exactly the shape that trips a per-minute limit,
# and failing the clip there would throw away a completed cut over a pause of a
# few seconds. Overload (529) behaves the same way.
RETRY_STATUSES = {429, 500, 502, 503, 529}
MAX_ATTEMPTS = 8


def retry_delay(exc: urllib.error.HTTPError, attempt: int) -> float:
    """How long to wait before retrying, preferring the server's own answer."""
    for header in ("retry-after", "anthropic-ratelimit-requests-reset"):
        raw = exc.headers.get(header) if exc.headers else None
        try:
            if raw:
                return max(1.0, min(float(raw), 120.0))
        except (TypeError, ValueError):
            pass
    # Exponential backoff, capped: 4, 8, 16, 32, 60, 60...
    return min(60.0, 2.0 ** (attempt + 1))


def call_llm(prompt: str) -> str:
    """Ask Anthropic for a scene plan.

    Anthropic only, by explicit choice. There is deliberately no fallback to
    another provider: a silent switch would mean clips were planned by a model
    the user did not choose, and the earlier fallback did exactly that when a
    token expired mid-run. A credential problem now stops the run and says so.
    """
    credential = (os.environ.get("ANTHROPIC_API_KEY")
                  or os.environ.get("ANTHROPIC_OAUTH_TOKEN") or "").strip()
    if not credential:
        raise SystemExit(
            "no Anthropic credential: set ANTHROPIC_API_KEY (sk-ant-api...) or "
            "ANTHROPIC_OAUTH_TOKEN in .env")

    kind = "subscription token" if credential.startswith("sk-ant-oat") else "API key"
    log(f"planning via Anthropic ({kind})")

    for attempt in range(MAX_ATTEMPTS):
        try:
            return call_anthropic(prompt, credential)
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRY_STATUSES:
                reason = ("credential rejected or expired" if exc.code == 401
                          else f"HTTP {exc.code}")
                raise SystemExit(
                    f"Anthropic unavailable ({reason}) - fix the credential in "
                    f".env and rerun; no other provider will be used") from None
            if attempt == MAX_ATTEMPTS - 1:
                raise SystemExit(
                    f"Anthropic still returning {exc.code} after "
                    f"{MAX_ATTEMPTS} attempts; try again later") from None
            wait = retry_delay(exc, attempt)
            log(f"Anthropic {exc.code}; waiting {wait:.0f}s "
                f"(attempt {attempt + 1}/{MAX_ATTEMPTS})")
            time.sleep(wait)
        except OSError as exc:
            # URLError subclasses OSError, and a connection reset part-way
            # through the response body arrives as a bare ConnectionResetError.
            # Both are transient and worth retrying; neither should cost a clip.
            if attempt == MAX_ATTEMPTS - 1:
                raise SystemExit(f"Anthropic unreachable: {exc}") from None
            wait = min(30.0, 2.0 ** (attempt + 1))
            log(f"Anthropic unreachable ({exc}); retrying in {wait:.0f}s")
            time.sleep(wait)

    raise SystemExit("Anthropic planning failed")


def salvage_slots(text: str) -> dict | None:
    """Recover whatever slot objects a malformed reply still contains.

    A single bad character -- an unescaped quote inside a description, or a
    reply cut short -- used to throw away the whole scene plan and with it the
    planning call that was already paid for. The slots are independent, so the
    valid ones are still usable and any gaps simply hold on the speaker.

    Spans are collected with a stack so objects *nested* inside the (possibly
    unterminated) outer wrapper are found; tracking only the outermost brace
    finds nothing when the wrapper never closes, which is the common failure.
    """
    stack, spans, in_str, esc = [], [], False, False
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
            stack.append(i)
        elif ch == "}" and stack:
            spans.append((stack.pop(), i + 1))

    slots = []
    for a, b in sorted(spans):
        try:
            obj = json.loads(text[a:b])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "i" in obj:
            slots.append(obj)

    # De-duplicate by slot index, keeping the first good reading of each.
    unique, seen = [], set()
    for obj in slots:
        key = obj.get("i")
        if key in seen:
            continue
        seen.add(key)
        unique.append(obj)
    return {"slots": unique} if unique else None


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
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    break

    # Whole-object parsing failed. Rather than lose the clip and the planning
    # call with it, keep every slot that is individually well-formed.
    rescued = salvage_slots(text)
    if rescued:
        log(f"scene plan was malformed; salvaged {len(rescued['slots'])} slot(s)")
        return rescued
    raise ValueError("no JSON object in model reply")


# --------------------------------------------------------------------------
# People screening
# --------------------------------------------------------------------------
#
# Stock footage regularly contains people even when the search term does not
# ask for any ("grocery aisle" returns shoppers). This screen rejects any clip
# showing a woman.
#
# It is deliberately **fail-closed**: a clip is kept only when every detected
# face across every sampled frame is *confidently* male. An uncertain face, an
# unreadable frame, a missing model or any error rejects the clip. That turns
# classifier error into lost footage rather than a woman appearing on screen,
# which is the right trade for a constraint the user cannot compromise on.
#
# Policy via BROLL_PEOPLE_POLICY:
#   no-women  (default) keep clips with no faces, or only confident male faces
#   no-people           keep only clips with no detectable faces at all
#   off                 no screening

GENDER_MODEL_URL = ("https://github.com/onnx/models/raw/main/validated/vision/"
                    "body_analysis/age_gender/models/gender_googlenet.onnx")
PERSON_MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
                    "object_detection_yolox/object_detection_yolox_2022nov.onnx")

# Validated against known speakers from this project: a female face read 0.73 on
# the female class, a male 0.73 on the male class. Requiring 0.65 on the male
# class keeps the ambiguous middle on the reject side.
MALE_CONFIDENCE = 0.65
# A person is present. Set low deliberately: a missed person is the failure that
# matters, a false positive only costs a clip.
PERSON_CONFIDENCE = 0.45
SCREEN_FRAMES = 8


def _cache_dir() -> Path:
    d = Path.home() / ".cache" / "autoshorts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cached_model(url: str, name: str):
    """Fetch a model once into the cache and load it."""
    import cv2
    model = _cache_dir() / name
    if not model.exists() or model.stat().st_size < 100_000:
        log(f"fetching {name} (one time)")
        urllib.request.urlretrieve(url, model)
    return cv2.dnn.readNetFromONNX(str(model))


def _person_score(net, frame) -> float:
    """Highest person score anywhere in `frame`.

    Deliberately a scalar, not boxes: this network's raw output is grid-relative
    and needs stride decoding to become pixel coordinates, so treating it as
    boxes invents people that are not there (a shelf of jars read as a crowd of
    26). The score is the part that is trustworthy without decoding.
    """
    import cv2
    import numpy as np

    size = 640
    h0, w0 = frame.shape[:2]
    if h0 < 8 or w0 < 8:
        return 0.0
    scale = min(size / h0, size / w0)
    resized = cv2.resize(frame, (max(1, int(w0 * scale)), max(1, int(h0 * scale))))
    canvas = np.ones((size, size, 3), np.uint8) * 114
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    blob = cv2.dnn.blobFromImage(canvas, 1.0, (size, size), (0, 0, 0),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    out = net.forward()[0]
    # Layout is [x, y, w, h, objectness, 80 COCO classes]; class 0 is person.
    return float((out[:, 4] * out[:, 5]).max())


def _person_regions(net, frame, threshold: float = PERSON_CONFIDENCE):
    """Regions of `frame` that contain a person, as (x, y, image) tiles.

    Detection is scale-sensitive: the network sees a 640x640 letterbox, so a
    1080p frame is shrunk ~3x and anyone at the back of a room becomes a
    handful of pixels. That is how an auditorium full of people passed as
    "1 face, male" -- the crowd was never detected at all.

    Sweeping the frame in overlapping tiles at native resolution puts distant
    people back at a size the network can see, and each tile doubles as the
    region to zoom into when verifying faces.
    """
    h, w = frame.shape[:2]
    regions = []
    if _person_score(net, frame) >= threshold:
        regions.append((0, 0, frame))

    tw, th = int(w / 2.5), int(h / 2.5)
    if tw < 16 or th < 16:
        return regions
    step_x, step_y = max(1, int(tw * 0.8)), max(1, int(th * 0.8))
    for y in range(0, max(1, h - th // 2), step_y):
        for x in range(0, max(1, w - tw // 2), step_x):
            tile = frame[y:min(h, y + th), x:min(w, x + tw)]
            if tile.size == 0:
                continue
            if _person_score(net, tile) >= threshold:
                regions.append((x, y, tile))
    return regions


def _person_present(net, frame, threshold: float = PERSON_CONFIDENCE) -> bool:
    """True when the frame contains a person."""
    return _person_score(net, frame) >= threshold or bool(
        _person_regions(net, frame, threshold))


def clip_is_allowed(video: Path, assets: Path, start: float = 0.0,
                    window: float | None = None) -> tuple[bool, str]:
    """Screen one downloaded clip. Returns (allowed, reason).

    Two stages, both fail-closed:

    1. Is anybody in shot? If not, the clip is fine.
    2. If somebody is, every person must be verifiable as male. A face that
       reads female, a face the classifier is unsure about, or a person whose
       face cannot be seen at all all reject the clip.

    Any error -- missing model, unreadable frame, detection failure -- also
    rejects. Losing a usable clip costs nothing but a speaker hold; letting one
    through cannot be undone once posted.
    """
    policy = (os.environ.get("BROLL_PEOPLE_POLICY") or "no-women").strip().lower()
    if policy == "off":
        return True, "screening off"

    try:
        import cv2
        import numpy as np
    except Exception as exc:
        return False, f"screening unavailable ({exc})"

    weights = assets / "face_detection_yunet_2023mar.onnx"
    if not weights.exists():
        return False, "face detector missing"

    try:
        detector = cv2.FaceDetectorYN.create(str(weights), "", (320, 320), 0.7)
        person_net = _cached_model(PERSON_MODEL_URL, "object_detection_yolox_2022nov.onnx")
        gender_net = (None if policy == "no-people"
                      else _cached_model(GENDER_MODEL_URL, "gender_googlenet.onnx"))
    except Exception as exc:
        return False, f"models unavailable ({exc})"

    try:
        duration = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(video)],
            check=True, text=True, capture_output=True).stdout.strip())
    except Exception as exc:
        return False, f"unreadable clip ({exc})"

    # Sample the stretch that will actually appear in the edit. Spreading the
    # samples over a 20 second asset when only ~1 second of it is used means a
    # person visible in the used moment can sit entirely between two samples --
    # which is exactly how a woman at a warehouse desk reached a finished clip.
    if window and window > 0:
        span_start = max(0.0, min(start, max(0.0, duration - window)))
        span = min(window, max(0.0, duration - span_start))
    else:
        span_start, span = 0.0, duration
    if span <= 0:
        span_start, span = 0.0, duration

    people_frames = 0
    verified_male = 0
    for i in range(SCREEN_FRAMES):
        t = span_start + span * (i + 0.5) / SCREEN_FRAMES
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(video),
             "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
            capture_output=True).stdout
        if not raw:
            continue
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        try:
            regions = _person_regions(person_net, frame)
        except Exception as exc:
            return False, f"person detection failed ({exc})"
        if not regions:
            continue

        people_frames += 1
        if policy == "no-people":
            return False, "contains a person"

        # Every region holding a person must be verifiable. Checking only the
        # faces the detector happens to find on the whole frame is what let a
        # crowd through: one readable male face "cleared" a room full of people.
        for (_rx, _ry, region) in regions:
            rh, rw = region.shape[:2]
            # Zoom small regions so a distant face is judged at usable detail.
            if rh < 480:
                factor = min(4.0, 480 / max(1, rh))
                region = cv2.resize(region, (int(rw * factor), int(rh * factor)))
                rh, rw = region.shape[:2]

            detector.setInputSize((rw, rh))
            try:
                _, faces = detector.detect(region)
            except Exception:
                return False, "face detection failed"
            if faces is None or not len(faces):
                # Somebody is in shot but their face cannot be read, so they
                # cannot be verified.
                return False, "person present but face not visible"

            for face in faces:
                x, y, fw, fh = (int(v) for v in face[:4])
                pad_f = int(max(fw, fh) * 0.25)
                crop = region[max(0, y - pad_f):y + fh + pad_f,
                              max(0, x - pad_f):x + fw + pad_f]
                if crop.size == 0:
                    return False, "face too close to frame edge to judge"
                blob = cv2.dnn.blobFromImage(crop, 1.0, (224, 224), (104, 117, 123),
                                             swapRB=False)
                gender_net.setInput(blob)
                raw = gender_net.forward().flatten()
                # This network's final layer is already a softmax: a confident
                # man comes back as [1.0, 0.0]. Re-softmaxing it squashed that
                # to [0.73, 0.27] and a woman to [0.38, 0.62], collapsing the
                # confidence scale so no face could ever read as clearly male
                # and every clip containing a person was rejected. Only
                # normalise when the output is not already a distribution.
                if raw.min() < 0.0 or abs(float(raw.sum()) - 1.0) > 0.01:
                    exp = np.exp(raw - raw.max())
                    raw = exp / exp.sum()
                male = float(raw[0])
                if male < MALE_CONFIDENCE:
                    return False, f"face not confidently male (male={male:.2f})"
                verified_male += 1

    if people_frames:
        return True, f"people in {people_frames} frame(s), {verified_male} face(s) all male"
    return True, "no people"


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
                       need_seconds: float,
                       assets: Path | None = None) -> tuple[Path, str, float] | None:
    """Fetch one freely-licensed clip from Wikimedia Commons.

    Needs no API key, and the skill lists Commons as an approved source, so this
    is the keyless fallback when Pexels has no match. Files are CC/PD; the
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
        target = dest_dir / asset_name(index, suffix)
        if not fetch_to(url, target,
                        {"User-Agent": "AutoShorts/1.0 (b-roll sourcing)"}):
            continue
        offset = clip_is_usable(target, need_seconds)
        if offset is None:
            log(f"commons clip rejected (too short or frozen): {page.get('title','')[:48]}")
            target.unlink(missing_ok=True)
            continue
        # Commons footage was previously used unscreened -- the people policy
        # only ever applied to Pexels. Same rules apply here.
        if assets is not None:
            allowed, reason = clip_is_allowed(target, assets, offset, need_seconds)
            if not allowed:
                log(f"rejected commons {query!r}: {reason}")
                target.unlink(missing_ok=True)
                continue
        return target, info.get("descriptionurl") or url, offset
    return None


# Assets are named per clip, not just per scene index. The edit directory is
# shared by every clip in a project, so plain `broll_009.mp4` means two runs --
# a batch render and a click in the app -- write and delete the same filename.
# One run then rejects and unlinks an asset the other has already written into
# its scene plan, and the render dies on a path that no longer exists.
ASSET_PREFIX = "broll"


def asset_name(index: int, suffix: str = ".mp4") -> str:
    return f"{ASSET_PREFIX}_{index:03d}{suffix}"


def fetch_to(url: str, target: Path, headers: dict, attempts: int = 3,
             timeout: int = 180) -> bool:
    """Download `url` to `target`, retrying transient network failures.

    Stock hosts sit behind CDNs that occasionally reset a connection mid-body.
    Without a retry a single reset discards the whole clip, which is a very
    expensive way to react to a hiccup that clears on the next attempt.
    """
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            if target.exists() and target.stat().st_size > 0:
                return True
            raise OSError("empty download")
        except Exception as exc:
            target.unlink(missing_ok=True)
            if attempt == attempts:
                log(f"download failed after {attempts} attempts: {exc}")
                return False
            time.sleep(2 ** attempt)
    return False


def pexels_download(query: str, dest_dir: Path, index: int,
                    assets: Path | None = None,
                    need_seconds: float | None = None) -> tuple[Path, str] | None:
    """Fetch one portrait clip for `query` that passes people screening.

    Results are tried in order and each is screened after download, so a clip
    containing a woman is discarded and the next candidate is tried rather than
    the beat falling straight back to the speaker.
    """
    key = (os.environ.get("PEXELS_API_KEY") or "").strip()
    if not key:
        return None
    # Ask for more results than needed: screening will reject some.
    url = ("https://api.pexels.com/videos/search?"
           + urllib.parse.urlencode({"query": query, "orientation": "portrait", "per_page": 12}))
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

    rejected = 0
    for video in data.get("videos", []):
        files = [f for f in video.get("video_files", []) if (f.get("height") or 0) >= 960]
        files.sort(key=lambda f: f.get("height") or 0)
        if not files:
            continue
        target = dest_dir / asset_name(index)
        if not fetch_to(files[0]["link"], target,
                        {"User-Agent": "AutoShorts/1.0 (local b-roll sourcing)"}):
            continue

        if assets is not None:
            allowed, reason = clip_is_allowed(target, assets, 0.0, need_seconds)
            if not allowed:
                log(f"rejected {query!r}: {reason}")
                target.unlink(missing_ok=True)
                rejected += 1
                continue

        return target, video.get("url", "")

    if rejected:
        log(f"no usable footage for {query!r} after {rejected} rejected by screening")
    return None


# --------------------------------------------------------------------------
# Plan assembly
# --------------------------------------------------------------------------

def apply_slots(plan: dict, slots: list[dict], downloads: Path,
                assets: Path | None = None) -> dict:
    """Rewrite scene kinds in place, keeping every original boundary.

    Only two kinds are produced: real stock footage, and the speaker. Generated
    explainer cards were removed -- a slot that cannot be filled with footage
    stays on the speaker rather than inventing a graphic.
    """
    scenes = plan["scenes"]
    by_index = {int(s.get("i", -1)): s for s in slots}
    used_queries: list[str] = []
    sourced = kept = unfilled = 0

    for i, scene in enumerate(scenes):
        choice = by_index.get(i) or {}
        kind = choice.get("kind", "source")

        # Slot 0 always shows the speaker, and silence stays on the speaker.
        if i == 0 or kind != "broll":
            scene["kind"] = "source"
            scene["description"] = choice.get("description") or "speaker return"
            for key in ("file", "offset", "source_url", "title", "subtitle",
                        "icon", "accent", "motion"):
                scene.pop(key, None)
            kept += 1
            continue

        query = (choice.get("query") or "").strip()
        got = None
        # Avoid using the same asset back to back.
        if query and query not in used_queries[-1:]:
            need = scene["end"] - scene["start"]
            got = pexels_download(query, downloads, i, assets, need)
            if got:
                got = (got[0], got[1], 0.0)
            else:
                got = wikimedia_download(query, downloads, i, need, assets)

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

        # Nothing found for this beat: hold on the speaker. Two source scenes in
        # a row simply play the original footage through, which reads naturally
        # even though the boundary carries no cut.
        scene["kind"] = "source"
        scene["description"] = f"speaker (no footage for {query!r})" if query else "speaker"
        for key in ("file", "offset", "source_url", "title", "subtitle",
                    "icon", "accent", "motion"):
            scene.pop(key, None)
        kept += 1
        unfilled += 1

    # Last line of defence: the renderer rejects the whole plan if any scene
    # points at a file that is not there, losing the entire clip. A missing
    # asset is only worth one beat, so drop it back to the speaker instead.
    dangling = 0
    for scene in scenes:
        if scene.get("kind") != "video":
            continue
        path = scene.get("file")
        if path and Path(path).exists():
            continue
        scene["kind"] = "source"
        scene["description"] = "speaker (asset went missing)"
        for key in ("file", "offset", "source_url", "crop_x", "crop_y"):
            scene.pop(key, None)
        sourced -= 1
        kept += 1
        dangling += 1
    if dangling:
        log(f"{dangling} sourced asset(s) vanished before render; held on speaker")

    log(f"plan: {sourced} footage, {kept} source "
        f"({unfilled} slots had no footage) across {len(scenes)} slots")
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
    ap.add_argument("--assets", default=str(Path(__file__).parent),
                    help="directory holding the detection models used to screen "
                         "footage for people")
    args = ap.parse_args()

    clip = Path(args.clip).expanduser().resolve()
    if not clip.exists():
        raise SystemExit(f"clip not found: {clip}")
    scripts = Path(args.skill_dir).expanduser().resolve() / "scripts"
    if not (scripts / "render_project.py").exists():
        raise SystemExit(f"b-rolls skill not found at {scripts}")

    # Each clip gets its own working directory. The skill's default is
    # `<clip parent>/edit`, which every clip in a project shares -- so a batch
    # render and a click in the app overwrite each other's scene_plan.json and
    # delete each other's downloads. Isolating per clip makes concurrent runs
    # safe, and keeps each clip's plan around for inspection afterwards.
    edit = clip.parent / f"edit_{clip.stem}"

    # 1. Inventory + draft plan (also detects the burned-in caption band).
    log("preparing project")
    run([args.python, str(scripts / "prepare_project.py"), str(clip),
         "--edit-dir", str(edit)],
        cwd=scripts, stdout=subprocess.DEVNULL)

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

    # Name this run's assets after the clip so a batch render and a click in
    # the app cannot overwrite or delete each other's downloads.
    global ASSET_PREFIX
    ASSET_PREFIX = f"broll_{clip.stem}"

    downloads = edit / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    if not (os.environ.get("PEXELS_API_KEY") or "").strip():
        log("PEXELS_API_KEY unset: sourcing from Wikimedia Commons only")

    plan = apply_slots(plan, slots, downloads, Path(args.assets))
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
