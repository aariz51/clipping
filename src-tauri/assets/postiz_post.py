"""Publish a rendered clip to connected Postiz channels.

Two steps, per the Postiz public API: upload the file, then create a post that
references the uploaded media against one or more integrations ("channels" in
the UI).

Subcommands:
  integrations            list connected channels
  post --video ... --content ... [--integration ID ...] [--when ISO]

Auth comes from POSTIZ_API_KEY. The base URL defaults to the cloud API and can
be pointed at a self-hosted instance with POSTIZ_API_URL.
"""

from __future__ import annotations

import argparse
import datetime
import json
import mimetypes
import os
import sys
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


def request(path, method="GET", body=None, content_type=None):
    req = urllib.request.Request(f"{base_url()}{path}", data=body, method=method)
    req.add_header("Authorization", api_key())
    if content_type:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            raw = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise SystemExit(f"Postiz {method} {path} failed ({exc.code}): {detail}") from None
    return json.loads(raw) if raw.strip() else {}


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


def list_integrations():
    data = request("/integrations")
    return data if isinstance(data, list) else data.get("integrations", [])


def upload(video):
    size_mb = video.stat().st_size / 1_048_576
    log(f"uploading {video.name} ({size_mb:.1f} MB)")
    body, ctype = multipart("file", video)
    result = request("/upload", method="POST", body=body, content_type=ctype)
    if not isinstance(result, dict) or not (result.get("id") or result.get("path")):
        raise SystemExit(f"unexpected upload response: {json.dumps(result)[:300]}")
    return result


def provider_settings(integration):
    """Per-provider settings block.

    Every post carries a `__type` naming its provider, and some providers
    require extra fields: Instagram and TikTok reject a post without knowing
    whether it is a feed post or a story/video.
    """
    identifier = (integration.get("identifier")
                  or integration.get("providerIdentifier") or "")
    settings = {"__type": identifier}
    if identifier.startswith("instagram"):
        settings["post_type"] = "post"
    return settings


def create_post(media, content, integrations, when):
    image = {k: v for k, v in media.items() if k in ("id", "path", "name")} or media
    # The API validates the date even for immediate posts, so "now" has to be a
    # real ISO 8601 timestamp rather than the literal string.
    date = when or datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.000Z")
    payload = {
        "type": "schedule" if when else "now",
        "date": date,
        # Both are required by the API and rejected when null.
        "shortLink": False,
        "tags": [],
        "posts": [
            {
                "integration": {"id": i["id"]},
                "value": [{"content": content, "image": [image]}],
                "settings": provider_settings(i),
            }
            for i in integrations
        ],
    }
    log("posting to " + ", ".join(i.get("name") or i["id"] for i in integrations))
    return request("/posts", method="POST",
                   body=json.dumps(payload).encode(), content_type="application/json")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("integrations")

    p = sub.add_parser("post")
    p.add_argument("--video", required=True)
    p.add_argument("--content", default="")
    p.add_argument("--integration", action="append", default=[],
                   help="integration id; repeatable. Omit to post to every channel.")
    p.add_argument("--when", help="ISO timestamp to schedule; omit to post now")
    p.add_argument("--dry-run", action="store_true",
                   help="upload and resolve channels but do not publish")
    args = ap.parse_args()

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

    print(json.dumps(create_post(media, args.content, chosen, args.when), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
