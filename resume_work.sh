#!/bin/zsh
# Wait for the Anthropic quota to return, then do the work in priority order:
#   1. The single clip Aariz asked for by name.
#   2. The full batch.
#
# Ordered deliberately: the batch consumes the whole 5-hour window, so if it
# runs first the requested clip waits another five hours behind it.
#
# Polls the API rather than trusting the clock, so a Mac that slept through the
# reset does not start against a limit that is still in force.

set -u
cd "$(dirname "$0")"

LOG="$HOME/broll-work/resume.log"
CLIP="$HOME/Documents/AutoShorts/AutoShorts_J_03EXyhYS8/clips/clip-01_flat.mp4"
TRANSCRIPT="$HOME/Documents/AutoShorts/AutoShorts_J_03EXyhYS8/clips/broll_transcript.json"
OUT="$HOME/Documents/AutoShorts/AutoShorts_J_03EXyhYS8/clips/clip-01_flat_broll.mp4"
mkdir -p "$(dirname "$LOG")"

set -a; . ./.env; set +a

say() { echo "[$(date '+%H:%M:%S')] $*" >>"$LOG"; }

probe() {
  ./.venv/bin/python - <<'PY'
import json, os, sys, urllib.request, urllib.error
tok = (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_OAUTH_TOKEN") or "").strip()
hdr = {"anthropic-version": "2023-06-01", "content-type": "application/json"}
if tok.startswith("sk-ant-oat"):
    hdr["authorization"] = f"Bearer {tok}"; hdr["anthropic-beta"] = "oauth-2025-04-20"
else:
    hdr["x-api-key"] = tok
req = urllib.request.Request(
    "https://api.anthropic.com/v1/messages",
    data=json.dumps({"model": os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
                     "max_tokens": 16,
                     "messages": [{"role": "user", "content": "ping"}]}).encode(),
    headers=hdr)
try:
    urllib.request.urlopen(req, timeout=60); print("ready"); sys.exit(0)
except urllib.error.HTTPError as e:
    util = (e.headers or {}).get("anthropic-ratelimit-unified-5h-utilization", "?")
    print(f"blocked {e.code} (5h utilisation {util})"); sys.exit(1)
except Exception as e:
    print(f"unreachable {e}"); sys.exit(1)
PY
}

say "waiting for Anthropic quota"
while true; do
  probe_result="$(probe)"
  [[ "$probe_result" == "ready" ]] && { say "quota available"; break; }
  say "$probe_result; checking again in 5 min"
  sleep 300
done

say "rendering the requested clip first: $(basename "$CLIP")"
caffeinate -i ./.venv/bin/python src-tauri/assets/broll_pipeline.py \
  --clip "$CLIP" --transcript "$TRANSCRIPT" \
  --topic "ultra-processed food, obesity and food industry addiction" \
  --output "$OUT" --assets src-tauri/assets >>"$HOME/broll-work/manual_clip.log" 2>&1
say "requested clip finished with exit $? -> $OUT"

say "starting the full batch"
caffeinate -i ./run_batch.sh "$HOME/broll-work/batch.log"
say "batch finished with exit $?"
