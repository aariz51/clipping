"""Place sound effects under a finished clip.

Effects are chosen from the edit itself, not sprinkled at random:

  whoosh     on scene changes, so a hard cut lands softly
  attention  once at the top, on the hook
  riser      leading into the biggest gap before a statement (a reveal)
  boom       on the strongest emphasis beat
  stinger    where the speaker's pace jumps, which usually marks a surprise

Everything is mixed well under the voice (-18 dB or lower) so speech stays the
loudest thing in the mix -- effects that fight the words cost retention rather
than adding to it.

Usage:
  sfx_mix.py --video IN.mp4 --scenes scene_plan.json --transcript words.json \
             --kit DIR --output OUT.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Gain per effect, in dB below the programme. Transitions sit lowest because
# there are many of them; a single boom can afford to be heard.
GAIN_DB = {
    "whoosh": -22.0,
    "attention": -16.0,
    "riser": -20.0,
    "boom": -15.0,
    "stinger": -18.0,
    "pop": -19.0,
}
# Never place two effects closer than this, or the mix turns into clutter.
MIN_GAP = 1.6
MAX_EFFECTS = 12


def log(msg: str) -> None:
    print(f"[sfx] {msg}", file=sys.stderr, flush=True)


def duration_of(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        check=True, text=True, capture_output=True).stdout.strip()
    return float(out or 0.0)


def plan_effects(scenes: list[dict], words: list[dict], duration: float) -> list[tuple[float, str]]:
    """Decide what plays when, from the structure of the edit."""
    picks: list[tuple[float, str]] = []

    # 1. One attention hit on the hook, just after the first word lands.
    if words:
        picks.append((max(0.0, float(words[0].get("start", 0.0)) - 0.12), "attention"))

    # 2. Whooshes on cuts between different kinds of scene. A cut from speaker
    #    to footage is a real visual change; footage to footage often is not.
    prev_kind = None
    for scene in scenes:
        kind = scene.get("kind")
        start = float(scene.get("start", 0.0))
        if prev_kind is not None and kind != prev_kind and start > 0.6:
            picks.append((start - 0.18, "whoosh"))
        prev_kind = kind

    # 3. The longest silence usually precedes the payoff: build into it.
    if len(words) > 4:
        gaps = []
        for a, b in zip(words, words[1:]):
            gap = float(b.get("start", 0)) - float(a.get("end", 0))
            if gap > 0.32:
                gaps.append((gap, float(b.get("start", 0))))
        gaps.sort(reverse=True)
        if gaps:
            _, at = gaps[0]
            picks.append((max(0.0, at - 1.9), "riser"))
            picks.append((at - 0.05, "boom"))
        # A sharp jump in speaking rate reads as a surprise.
        if len(gaps) > 2:
            picks.append((max(0.0, gaps[1][1] - 0.1), "stinger"))

    # Thin by importance, not by clock order. Cuts are frequent, so a plain
    # first-come filter fills the whole budget with whooshes and the reveal
    # beats -- the ones that actually carry meaning -- never get placed.
    PRIORITY = {"attention": 0, "riser": 1, "boom": 1, "stinger": 2, "whoosh": 3, "pop": 3}
    MAX_PER_KIND = {"whoosh": 4}

    picks = [(at, n) for at, n in picks if 0 <= at <= duration - 0.25]
    picks.sort(key=lambda p: (PRIORITY.get(p[1], 3), p[0]))

    chosen: list[tuple[float, str]] = []
    counts: dict[str, int] = {}
    for at, name in picks:
        if len(chosen) >= MAX_EFFECTS:
            break
        if counts.get(name, 0) >= MAX_PER_KIND.get(name, MAX_EFFECTS):
            continue
        if any(abs(at - other) < MIN_GAP for other, _ in chosen):
            continue
        chosen.append((at, name))
        counts[name] = counts.get(name, 0) + 1

    chosen.sort()
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--kit", required=True)
    ap.add_argument("--scenes")
    ap.add_argument("--transcript")
    ap.add_argument("--output", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    video = Path(args.video).expanduser().resolve()
    kit = Path(args.kit).expanduser().resolve()
    if not video.exists():
        raise SystemExit(f"video not found: {video}")

    scenes = []
    if args.scenes and Path(args.scenes).exists():
        scenes = json.loads(Path(args.scenes).read_text()).get("scenes", [])
    words = []
    if args.transcript and Path(args.transcript).exists():
        words = json.loads(Path(args.transcript).read_text()).get("words", [])

    duration = duration_of(video)
    picks = plan_effects(scenes, words, duration)
    if not picks:
        log("nothing to place; copying through")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(video), "-c", "copy", args.output], check=True)
        print(args.output)
        return 0

    available = []
    for at, name in picks:
        path = kit / f"{name}.wav"
        if path.exists():
            available.append((at, name, path))
        else:
            log(f"missing {name}.wav, skipping")
    if not available:
        raise SystemExit("no effect files found in the kit")

    for at, name, _ in available:
        log(f"{name:10s} at {at:6.2f}s")

    inputs = ["-i", str(video)]
    for _, _, path in available:
        inputs += ["-i", str(path)]

    parts, labels = [], []
    for idx, (at, name, _) in enumerate(available, start=1):
        gain = GAIN_DB.get(name, -18.0)
        delay = int(at * 1000)
        parts.append(f"[{idx}:a]adelay={delay}|{delay},volume={gain}dB[s{idx}]")
        labels.append(f"[s{idx}]")
    mix = ("[0:a]" + "".join(labels)
           + f"amix=inputs={len(labels) + 1}:duration=first:normalize=0[aout]")
    filt = ";".join(parts + [mix])

    if args.dry_run:
        print(json.dumps({"placements": [{"at": a, "effect": n} for a, n, _ in available]}, indent=1))
        return 0

    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs,
        "-filter_complex", filt,
        "-map", "0:v", "-map", "[aout]",
        # Video is copied, so adding sound costs no picture quality at all.
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", args.output,
    ], check=True, capture_output=True, text=True)
    log(f"mixed {len(available)} effect(s)")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
