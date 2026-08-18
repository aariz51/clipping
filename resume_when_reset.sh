#!/bin/zsh
# Wait for the Anthropic 5-hour window to reset, then run the batch.
#
# Polls rather than trusting a fixed clock time: the reset timestamp the API
# reported is the *earliest* it could lift, and a machine that slept through
# 05:00 would otherwise start against a limit that is still in force. When the
# probe succeeds the quota is genuinely available, whatever the clock says.
#
# The probe is the cheapest possible request (max_tokens 16), so waiting costs
# essentially nothing from the quota it is waiting for.
#
# Detached and self-contained: survives the terminal, the session, and sleep.

set -u
cd "$(dirname "$0")"

LOG="$HOME/broll-work/resume.log"
BATCH_LOG="$HOME/broll-work/batch.log"
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
    urllib.request.urlopen(req, timeout=60)
    print("ready"); sys.exit(0)
except urllib.error.HTTPError as e:
    # 401 is not a rate limit and waiting will never clear it: an OAuth access
    # token lasts about 8 hours, so an overnight wait outlives the credential
    # it is waiting with. Say so instead of polling a dead token forever.
    if e.code == 401:
        print("expired"); sys.exit(2)
    util = (e.headers or {}).get("anthropic-ratelimit-unified-5h-utilization", "?")
    print(f"blocked {e.code} (5h utilisation {util})"); sys.exit(1)
except Exception as e:
    print(f"unreachable {e}"); sys.exit(1)
PY
}

say "waiting for the Anthropic 5-hour window to reset"
while true; do
  # Not `status`: that name is read-only in zsh and assigning to it kills the
  # script on the first loop.
  probe_result="$(probe)"
  if [[ "$probe_result" == "ready" ]]; then
    say "quota available - starting batch"
    break
  fi
  if [[ "$probe_result" == "expired" ]]; then
    say "ANTHROPIC CREDENTIAL EXPIRED - waiting cannot fix this."
    say "Put a fresh token in .env (ANTHROPIC_OAUTH_TOKEN) or a console key"
    say "(ANTHROPIC_API_KEY, which does not expire), then rerun this script."
    exit 2
  fi
  say "$probe_result; checking again in 5 min"
  sleep 300
done

# caffeinate: the render is hours of ffmpeg work and a sleeping Mac would stall
# it mid-clip. -i blocks idle sleep only, so closing the lid still sleeps.
say "batch starting (log: $BATCH_LOG)"
caffeinate -i ./run_batch.sh "$BATCH_LOG"
say "batch finished with exit $?"
