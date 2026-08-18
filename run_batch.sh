#!/bin/zsh
# Render every candidate of every transcribed project, unattended.
#
# Detached on purpose: the batch runs for hours and two earlier runs were
# killed when the launching session went away. `nohup` + `setsid` keeps it
# alive; the log is the only thing to watch.
#
# Resumable — a clip whose finished file already exists is skipped, so this can
# be re-run after an interruption without repeating any encoding.
#
# Nothing is posted. Rendering only.

set -u
cd "$(dirname "$0")/src-tauri"

LOG="${1:-$HOME/broll-work/batch.log}"
mkdir -p "$(dirname "$LOG")"

# Anthropic only. LLM_PROVIDER is set explicitly so the batch never picks up a
# different provider from a stale environment.
export LLM_PROVIDER=claude
export MAX_CANDIDATES="${MAX_CANDIDATES:-25}"
export BROLL_PEOPLE_POLICY="${BROLL_PEOPLE_POLICY:-no-women}"

echo "=== batch started $(date) (MAX_CANDIDATES=$MAX_CANDIDATES) ===" >>"$LOG"
exec cargo test --lib --release -- --ignored --nocapture batch_render_all >>"$LOG" 2>&1
