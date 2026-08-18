#!/bin/zsh
# Retitle everything already rendered, then render every clip still missing.
#
# Ordered so the cheap work lands first: retitling an existing render costs one
# short LLM call, while a new clip costs a full B-roll plan. If the quota runs
# out mid-way, the clips that already exist are the ones that got improved.
set -u
cd "$(dirname "$0")/src-tauri"
LOG="$HOME/broll-work/finish_all.log"
mkdir -p "$(dirname "$LOG")"
export LLM_PROVIDER=claude MAX_CANDIDATES=25 BROLL_PEOPLE_POLICY=no-women
# Several clips at once: one clip's ffmpeg overlaps another's planning call, so
# the run finishes sooner and spends the quota while it is actually available.
export BATCH_CONCURRENCY="${BATCH_CONCURRENCY:-3}"

echo "=== waiting for any running retitle sweep ===" >>"$LOG"
while pgrep -f retitle_all >/dev/null; do sleep 20; done

# Several passes: a clip that failed on a transient rate limit is retried on
# the next sweep, and anything already rendered is skipped, so repeated passes
# converge on "everything done" without repeating any encoding.
RESUME_PASSES="${RESUME_PASSES:-6}"
for pass in $(seq 1 "$RESUME_PASSES"); do
  echo "=== batch pass $pass/$RESUME_PASSES $(date) ===" >>"$LOG"
  caffeinate -i cargo test --lib --release -- --ignored --nocapture render_parallel >>"$LOG" 2>&1
  # Nothing left to do? Stop early rather than spinning.
  if grep -q "BATCH COMPLETE: 0 rendered" "$LOG" 2>/dev/null; then
    tail -1 "$LOG" | grep -q "0 failed" && { echo "=== nothing left to render ===" >>"$LOG"; break; }
  fi
  sleep 60
done
echo "=== all passes finished $(date) ===" >>"$LOG"
