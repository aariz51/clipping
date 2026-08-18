"""Publish a rendered clip to connected Postiz channels.

Two steps, per the Postiz public API: upload the file, then create a post that
references the uploaded media against one or more integrations ("channels" in
the UI).

Subcommands:
  integrations            list connected channels
  post --video ... --content ... [--integration ID ...] [--when ISO] [--publish]

Posts are created as drafts unless --publish is given.

Auth comes from POSTIZ_API_KEY. The base URL defaults to the cloud API and can
be pointed at a self-hosted instance with POSTIZ_API_URL.
"""

from __future__ import annotations

import argparse
import datetime
import json
import mimetypes
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

DEFAULT_BASE = "https://api.postiz.com/public/v1"


def log(msg: str) -> None:
    print(f"[postiz] {msg}", file=sys.stderr, flush=True)


def base_url() -> str:
    return (os.environ.get("POSTIZ_API_URL") or DEFAULT_BASE).rstrip("/")


def api_key() -> str:
    key = (os.environ.get("POSTIZ_API_KEY") or "").strip()
    if not key:
        raise SystemExit(
            "POSTIZ_API_KEY is not set. In Postiz: Settings > Developers > Public API."
        )
    return key


def explain(detail: str) -> str:
    """Turn Postiz's validation payload into something readable.

    A rejected post answers with a JSON array of messages like
    `posts.0.settings.privacy_level must be a string`. Printed raw that is a
    wall of text; grouped by field it says plainly what is missing.
    """
    try:
        payload = json.loads(detail)
    except Exception:
        return detail[:400]
    message = payload.get("message", payload)
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        fields = {}
        for entry in message:
            field = str(entry).split(" ")[0]
            fields.setdefault(field.rsplit(".", 1)[-1], str(entry))
        return "; ".join(sorted(fields.values()))[:600]
    return str(message)[:400]


def request(path, method="GET", body=None, content_type=None, attempts=5):
    """Call the API, retrying transient failures.

    A body over the server's limit is dropped mid-write and raises a broken
    pipe rather than an HTTP status, and the API occasionally 5xxs under load.
    Both are worth retrying; a 4xx is not, since the request itself is wrong.
    """
    last = None
    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(f"{base_url()}{path}", data=body, method=method)
        req.add_header("Authorization", api_key())
        if content_type:
            req.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                raw = resp.read().decode("utf-8", "replace")
            return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            if exc.code < 500:
                raise SystemExit(
                    f"Postiz {method} {path} failed ({exc.code}): {explain(detail)}"
                ) from None
            last = f"{exc.code}: {explain(detail)}"
        except (urllib.error.URLError, OSError) as exc:
            # Broken pipe here almost always means the body was too large.
            last = f"{exc}"
        if attempt < attempts:
            # Postiz drops requests under load -- observed resetting and then
            # timing out a plain GET /integrations three times in a row. Five
            # attempts with a longer ceiling rides that out instead of failing
            # a publish the user is watching.
            wait = min(30, 2 ** attempt)
            log(f"{method} {path} failed ({last}); retrying in {wait}s")
            time.sleep(wait)
    raise SystemExit(f"Postiz {method} {path} failed after {attempts} attempts: {last}")


def multipart(field, path):
    """Encode one file as multipart/form-data."""
    boundary = f"----autoshorts{uuid.uuid4().hex}"
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{path.name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    return head + path.read_bytes() + tail, f"multipart/form-data; boundary={boundary}"


# Postiz rejects request bodies over 50 MB, and drops the connection rather
# than answering, which surfaces as a broken pipe. Leave headroom for the
# multipart envelope.
MAX_UPLOAD_MB = 45.0

# Quality ladder for shrinking an oversized upload. CRF 20 is visually
# transparent at phone size; each step trades a little detail for size. A
# constrained bitrate is the last resort because it degrades busy frames worst,
# and B-roll cuts are exactly that.
CRF_LADDER = [20, 23, 26]


def media_shape(video: Path) -> dict:
    """Resolution, frame rate, frame count and duration of a file."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
         "-show_entries", "format=duration", "-of", "json", str(video)],
        check=True, text=True, capture_output=True).stdout
    data = json.loads(out)
    stream = (data.get("streams") or [{}])[0]
    return {
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": stream.get("r_frame_rate"),
        "frames": int(stream.get("nb_read_frames") or 0),
        "duration": float((data.get("format") or {}).get("duration") or 0.0),
    }


def fit_for_upload(video: Path) -> Path:
    """Return a copy of `video` small enough to upload.

    The rendered file is never modified: an oversized clip is re-encoded to a
    separate temporary file and the original stays exactly as the pipeline
    produced it.

    Geometry is preserved rather than downscaled -- same resolution, same frame
    rate, same frame count -- so the edit is untouched and only compression
    efficiency changes. **Audio is stream-copied**, so the speaker's voice and
    the cloned outro line come through bit-identical.
    """
    size_mb = video.stat().st_size / 1_048_576
    if size_mb <= MAX_UPLOAD_MB:
        return video

    import tempfile
    before = media_shape(video)
    target = Path(tempfile.gettempdir()) / f"{video.stem}_upload.mp4"

    for crf in CRF_LADDER:
        log(f"{size_mb:.1f} MB exceeds {MAX_UPLOAD_MB:.0f} MB; re-encoding at CRF {crf}")
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
            "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            # Audio untouched: re-encoding it would alter the voice for no
            # meaningful size saving.
            "-c:a", "copy", "-movflags", "+faststart", str(target),
        ], check=True, capture_output=True, text=True)
        new_mb = target.stat().st_size / 1_048_576
        if new_mb <= MAX_UPLOAD_MB:
            after = media_shape(target)
            # Refuse to upload something that is not the same edit.
            if (after["width"], after["height"], after["frames"]) != (
                    before["width"], before["height"], before["frames"]):
                raise SystemExit(
                    f"re-encode changed the video: {before} -> {after}")
            log(f"re-encoded to {new_mb:.1f} MB at CRF {crf}, "
                f"{after['width']}x{after['height']} {after['frames']} frames unchanged")
            return target
        log(f"CRF {crf} still {new_mb:.1f} MB, trying harder")

    raise SystemExit(
        f"could not bring {video.name} under {MAX_UPLOAD_MB:.0f} MB without "
        f"degrading it further; shorten the clip or raise the limit")


def list_integrations():
    data = request("/integrations")
    return data if isinstance(data, list) else data.get("integrations", [])


def upload(video):
    video = fit_for_upload(video)
    size_mb = video.stat().st_size / 1_048_576
    log(f"uploading {video.name} ({size_mb:.1f} MB)")
    body, ctype = multipart("file", video)
    result = request("/upload", method="POST", body=body, content_type=ctype)
    if not isinstance(result, dict) or not (result.get("id") or result.get("path")):
        raise SystemExit(f"unexpected upload response: {json.dumps(result)[:300]}")
    return result


# Complete settings blocks per provider.
#
# Postiz validates the whole block server-side and answers a 400 listing every
# missing field, so incomplete settings fail the post outright rather than
# falling back to defaults. Each platform's required keys are declared here in
# full instead of being discovered one error at a time.
#
# TikTok specifically: `autoAddMusic` stays "no" -- an auto-added soundtrack
# would sit under the speaker's own audio -- and `content_posting_method`
# defaults to DIRECT_POST so a published post actually appears on the profile.
PROVIDER_DEFAULTS = {
    "tiktok": {
        # DIRECT_POST publishes straight to the profile, the way Instagram
        # behaves. UPLOAD only drops the file into the TikTok app's inbox for
        # the user to finish by hand, which makes scheduling pointless -- so
        # direct posting is the default and UPLOAD is opt-in.
        #
        # DIRECT_POST needs the connected TikTok app to hold the video.publish
        # scope. If TikTok rejects it, set TIKTOK_POSTING_METHOD=UPLOAD to fall
        # back to the inbox flow.
        "content_posting_method": (
            os.environ.get("TIKTOK_POSTING_METHOD") or "DIRECT_POST").strip().upper(),
        "privacy_level": "PUBLIC_TO_EVERYONE",
        "duet": True,
        "stitch": True,
        "comment": True,
        "autoAddMusic": "no",
        "brand_content_toggle": False,
        "brand_organic_toggle": False,
    },
    "instagram": {"post_type": "post"},
    "instagram-standalone": {"post_type": "post"},
    "youtube": {"type": "public", "title": ""},
    "pinterest": {"board": ""},
    "facebook": {},
    "linkedin": {},
    "linkedin-page": {},
    "x": {"who_can_reply_post": "everyone"},
    "threads": {},
    "mastodon": {},
    "bluesky": {},
}

# Providers that also want a short title alongside the body text.
TITLE_PROVIDERS = {"tiktok": 90, "youtube": 100, "pinterest": 100}


def provider_settings(integration, title=""):
    """Settings block for one channel.

    Falls back to just `__type` for a provider not listed, which is what Postiz
    accepts for the simple text networks.
    """
    identifier = (integration.get("identifier")
                  or integration.get("providerIdentifier") or "")
    settings = {"__type": identifier}

    for prefix, defaults in PROVIDER_DEFAULTS.items():
        if identifier == prefix or identifier.startswith(prefix + "-"):
            settings.update(defaults)
            break

    limit = next((n for pre, n in TITLE_PROVIDERS.items()
                  if identifier == pre or identifier.startswith(pre + "-")), None)
    if limit:
        # First line of the caption reads better as a title than a truncation
        # of the whole body.
        first_line = (title or "").strip().splitlines()[0] if title.strip() else ""
        settings["title"] = first_line[:limit]

    return settings


def create_post(media, content, integrations, when, publish=False):
    """Create the post in Postiz.

    Defaults to a **draft**. Publishing straight from automation is what gets
    accounts rate-limited or shadow-banned, so the clip lands in Postiz for
    review and the final publish is a deliberate human action. Pass
    `publish=True` only when that is genuinely wanted.
    """
    image = {k: v for k, v in media.items() if k in ("id", "path", "name")} or media
    # The API validates the date even for drafts and immediate posts, so it has
    # to be a real ISO 8601 timestamp rather than the literal string "now".
    date = when or datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.000Z")
    if publish:
        post_type = "schedule" if when else "now"
    else:
        post_type = "draft"
    payload = {
        "type": post_type,
        "date": date,
        # Both are required by the API and rejected when null.
        "shortLink": False,
        "tags": [],
        "posts": [
            {
                "integration": {"id": i["id"]},
                "value": [{"content": content, "image": [image]}],
                "settings": provider_settings(i, content),
            }
            for i in integrations
        ],
    }
    what = "publishing to" if publish else "saving draft for"
    log(f"{what} " + ", ".join(i.get("name") or i["id"] for i in integrations))
    return request("/posts", method="POST",
                   body=json.dumps(payload).encode(), content_type="application/json")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("integrations")
    sub.add_parser("selftest")

    p = sub.add_parser("post")
    p.add_argument("--video", required=True)
    p.add_argument("--content", default="")
    p.add_argument("--integration", action="append", default=[],
                   help="integration id; repeatable. Omit to post to every channel.")
    p.add_argument("--when", help="ISO timestamp to schedule; omit to post now")
    p.add_argument("--dry-run", action="store_true",
                   help="upload and resolve channels but create nothing")
    p.add_argument("--publish", action="store_true",
                   help="actually publish. Without this the post is saved as a "
                        "draft in Postiz for review, which is the default so "
                        "automation never posts straight to a live account.")
    args = ap.parse_args()

    if args.command == "selftest":
        return _selftest()

    if args.command == "integrations":
        found = list_integrations()
        print(json.dumps(found, indent=2))
        log(f"{len(found)} connected channel(s)")
        return 0

    video = Path(args.video).expanduser().resolve()
    if not video.exists():
        raise SystemExit(f"video not found: {video}")

    available = list_integrations()
    if not available:
        raise SystemExit("no channels connected in Postiz; connect one in the UI first")
    chosen = [i for i in available if i["id"] in args.integration] if args.integration else available
    if not chosen:
        raise SystemExit(
            "none of the requested integration ids exist. Available: "
            + ", ".join(f"{i.get('name')}={i['id']}" for i in available)
        )

    media = upload(video)
    if args.dry_run:
        log("dry run: uploaded but not published")
        print(json.dumps({"uploaded": media, "would_post_to": [i["id"] for i in chosen]}, indent=2))
        return 0

    print(json.dumps(
        create_post(media, args.content, chosen, args.when, publish=args.publish), indent=2))
    return 0




def _selftest():
    """Guard the two things that broke in production.

    Run: python postiz_post.py selftest
    """
    ok = True

    tiktok = provider_settings({"identifier": "tiktok"}, "A title\nsecond line")
    # DIRECT_POST is the default so a published post reaches the profile; the
    # inbox flow stays available through TIKTOK_POSTING_METHOD=UPLOAD.
    want_method = (os.environ.get("TIKTOK_POSTING_METHOD") or "DIRECT_POST").strip().upper()
    for key, want in [("autoAddMusic", "no"), ("content_posting_method", want_method),
                      ("brand_content_toggle", False), ("brand_organic_toggle", False)]:
        if tiktok.get(key) != want:
            print(f"FAIL tiktok {key}={tiktok.get(key)!r} expected {want!r}"); ok = False
    for key in ("privacy_level", "duet", "stitch", "comment", "title"):
        if key not in tiktok:
            print(f"FAIL tiktok missing {key}"); ok = False
    if tiktok.get("title") != "A title":
        print(f"FAIL tiktok title should be the first line, got {tiktok.get('title')!r}"); ok = False

    ig = provider_settings({"identifier": "instagram-standalone"})
    if ig.get("post_type") != "post":
        print("FAIL instagram post_type"); ok = False

    if provider_settings({"identifier": "brand-new-network"}) != {"__type": "brand-new-network"}:
        print("FAIL unknown provider should degrade to __type only"); ok = False

    # Drafts by default: publishing must be opt-in.
    media = {"id": "x", "path": "p"}
    ints = [{"id": "1", "identifier": "x"}]
    import unittest.mock as mock
    with mock.patch(f"{__name__}.request") as req:
        create_post(media, "hello", ints, None)
        assert req.call_args, "no request made"
        body = json.loads(req.call_args.kwargs["body"])
        if body["type"] != "draft":
            print(f"FAIL default type is {body['type']!r}, expected 'draft'"); ok = False
    with mock.patch(f"{__name__}.request") as req:
        create_post(media, "hello", ints, None, publish=True)
        body = json.loads(req.call_args.kwargs["body"])
        if body["type"] != "now":
            print(f"FAIL publish=True gave {body['type']!r}"); ok = False

    print("selftest: PASS" if ok else "selftest: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
