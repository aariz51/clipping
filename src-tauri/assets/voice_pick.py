"""Choose a reference voice sample from a clip for voice cloning.

Picks the speaker whose voice the outro should be spoken in. When the clip
contains more than one speaker the female voice wins, per the brief; with only
male voices the best male sample is used instead.

Speaker identity comes from the transcript's diarisation labels when present.
Gender is inferred from median fundamental frequency (F0), which separates
adult voices reliably enough for this choice:

    typical adult female  165-255 Hz
    typical adult male     85-155 Hz

Prints JSON: {"start": s, "end": s, "speaker": "...", "f0": Hz, "gender": "..."}
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import wave

import numpy as np

# Lower bound of the typical adult female band. Measured samples: a female
# interview read 190 Hz, a male speaker 135-160 Hz, so 165 keeps the borderline
# male frames on the male side. This only decides *between* speakers -- with a
# single speaker in the clip their voice is used whatever it is classified as.
FEMALE_F0_THRESHOLD = 165.0

# Cloning quality degrades on very short references and gains little beyond a
# few seconds of continuous speech.
MIN_REF_SECONDS = 4.0
MAX_REF_SECONDS = 9.0


def log(msg: str) -> None:
    print(f"[voice] {msg}", file=sys.stderr, flush=True)


def load_mono(path: str, start: float, duration: float, rate: int = 16000) -> np.ndarray:
    """Decode a slice of audio to mono float32 in [-1, 1]."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", path,
           "-t", f"{duration:.3f}", "-ac", "1", "-ar", str(rate),
           "-f", "wav", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    # Skip the RIFF header by parsing rather than assuming a fixed offset.
    import io
    with wave.open(io.BytesIO(raw), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def f0_stats(samples: np.ndarray, rate: int = 16000) -> tuple[float, float] | None:
    """Median pitch and its relative spread.

    Spread matters as much as the median: a window holding one clean speaker has
    tightly clustered pitch, while one containing cross-talk, music or a speaker
    change is scattered. Cloning from a scattered window produced a voice a full
    third below the speaker (152 Hz from a 186 Hz talker), so the spread is used
    to reject those windows.
    """
    pitches = _voiced_pitches(samples, rate)
    if len(pitches) < 5:
        return None
    arr = np.array(pitches)
    median = float(np.median(arr))
    if median <= 0:
        return None
    # Interquartile range is robust to the odd octave-error frame.
    iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
    return median, iqr / median


def _voiced_pitches(samples: np.ndarray, rate: int = 16000) -> list[float]:
    """Per-frame pitch estimates over voiced frames.

    Autocorrelation is used rather than a library estimator to avoid pulling in
    another dependency; only coarse values are needed to pick a speaker.
    """
    frame = int(0.04 * rate)          # 40 ms
    hop = int(0.02 * rate)            # 20 ms
    lo = int(rate / 300)              # 300 Hz ceiling
    hi = int(rate / 70)               # 70 Hz floor
    if samples.size < frame:
        return []

    pitches: list[float] = []
    energies = []
    frames = []
    for i in range(0, samples.size - frame, hop):
        seg = samples[i:i + frame]
        energies.append(float(np.sqrt(np.mean(seg ** 2))))
        frames.append(seg)
    if not frames:
        return []

    # Only analyse frames with real energy; silence and breaths produce noise.
    speech_floor = max(0.02, float(np.median(energies)) * 0.8)

    for seg, energy in zip(frames, energies):
        if energy < speech_floor:
            continue
        seg = seg - seg.mean()
        corr = np.correlate(seg, seg, mode="full")[frame - 1:]
        if corr[0] <= 0:
            continue
        corr /= corr[0]
        window = corr[lo:hi]
        if window.size == 0:
            continue
        peak = int(np.argmax(window)) + lo
        # Weak periodicity means unvoiced; skip rather than guess.
        if corr[peak] < 0.3:
            continue
        pitches.append(rate / peak)

    return pitches


def median_f0(samples: np.ndarray, rate: int = 16000) -> float | None:
    stats = f0_stats(samples, rate)
    return stats[0] if stats else None


def speech_windows(words: list[dict], clip_end: float) -> list[tuple[float, float]]:
    """Continuous stretches of speech, split at pauses and label changes.

    Windows are the unit that gets pitch-measured. Local Whisper emits no
    diarisation -- every word carries the same speaker label -- so speakers are
    separated afterwards by pitch rather than by label.
    """
    spans: list[tuple[float, float]] = []
    if not words:
        return spans
    current = words[0].get("speaker") or "spk0"
    start = float(words[0].get("start", 0.0))
    last = float(words[0].get("end", 0.0))
    for w in words[1:]:
        spk = w.get("speaker") or "spk0"
        ws, we = float(w.get("start", 0.0)), float(w.get("end", 0.0))
        if spk != current or ws - last > 0.6:
            if last > start:
                spans.append((start, min(last, clip_end)))
            current, start = spk, ws
        last = we
    if last > start:
        spans.append((start, min(last, clip_end)))
    return spans


def group_by_pitch(measured: list[dict]) -> list[list[dict]]:
    """Split measured windows into speakers using their pitch.

    With no diarisation available, pitch is the only signal separating talkers.
    Windows are sorted by F0 and cut at the largest gap; the split is kept only
    when the two groups are far enough apart to be different people rather than
    one person's natural range (which spans roughly 30 Hz).
    """
    if len(measured) < 2:
        return [measured]
    ordered = sorted(measured, key=lambda m: m["f0"])
    gaps = [(ordered[i + 1]["f0"] - ordered[i]["f0"], i) for i in range(len(ordered) - 1)]
    widest, at = max(gaps, key=lambda g: g[0])
    # One speaker's frame-to-frame median rarely jumps 40 Hz between windows;
    # two speakers of different registers usually do.
    if widest < 40.0:
        return [ordered]
    return [ordered[: at + 1], ordered[at + 1 :]]


def speaker_windows(words: list[dict], clip_end: float) -> dict[str, list[tuple[float, float]]]:
    """Group consecutive words into continuous windows per speaker."""
    groups: dict[str, list[tuple[float, float]]] = {}
    if not words:
        return groups
    current = words[0].get("speaker") or "spk0"
    start = float(words[0].get("start", 0.0))
    last = float(words[0].get("end", 0.0))
    for w in words[1:]:
        spk = w.get("speaker") or "spk0"
        ws, we = float(w.get("start", 0.0)), float(w.get("end", 0.0))
        # A speaker change or a long pause ends the window.
        if spk != current or ws - last > 0.6:
            if last > start:
                groups.setdefault(current, []).append((start, min(last, clip_end)))
            current, start = spk, ws
        last = we
    if last > start:
        groups.setdefault(current, []).append((start, min(last, clip_end)))
    return groups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="clip to analyse")
    ap.add_argument("--transcript", help="JSON with clip-relative words")
    ap.add_argument("--out", help="write the chosen reference sample here (wav)")
    args = ap.parse_args()

    duration = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", args.audio],
        check=True, text=True, capture_output=True).stdout.strip())

    words = []
    if args.transcript:
        try:
            words = json.load(open(args.transcript)).get("words", [])
        except Exception as exc:
            log(f"could not read transcript: {exc}")

    spans = speech_windows(words, duration)
    if not spans:
        # No transcript: scan the clip at fixed offsets so multiple speakers can
        # still be found by pitch.
        step = MAX_REF_SECONDS
        spans = [(t, min(duration, t + step))
                 for t in np.arange(0.0, max(step, duration - step), step)]

    # Measure every usable window; speakers are separated from these afterwards.
    measured = []
    for span in sorted(spans, key=lambda s: s[1] - s[0], reverse=True)[:12]:
        length = min(MAX_REF_SECONDS, span[1] - span[0])
        if length < 1.5:
            continue
        try:
            samples = load_mono(args.audio, span[0], length)
        except subprocess.CalledProcessError:
            continue
        stats = f0_stats(samples)
        if stats is None:
            continue
        f0, spread = stats
        measured.append({
            "start": span[0],
            "end": span[0] + length,
            "seconds": length,
            "f0": round(f0, 1),
            "spread": round(spread, 3),
            "gender": "female" if f0 >= FEMALE_F0_THRESHOLD else "male",
            # Clean pitch matters more than length: cloning from a scattered
            # window produced a voice a third below the real speaker.
            "quality": round(min(length / MAX_REF_SECONDS, 1.0) * (1.0 - min(spread, 1.0)), 3),
        })

    # Each pitch group is treated as one speaker; its best window represents it.
    candidates = []
    for index, group in enumerate(group_by_pitch(measured)):
        if not group:
            continue
        best = max(group, key=lambda m: m["quality"])
        best = dict(best)
        best["speaker"] = f"spk{index}"
        best["windows"] = len(group)
        candidates.append(best)

    if not candidates:
        print(json.dumps({"error": "no usable voice found"}))
        return 1

    females = [c for c in candidates if c["gender"] == "female"]
    pool = females or candidates
    # Among eligible speakers prefer the longest usable reference.
    choice = max(pool, key=lambda c: c["quality"])
    choice["speakers_found"] = len(candidates)
    choice["had_female"] = bool(females)

    log(f"{len(candidates)} speaker(s); chose {choice['speaker']} "
        f"({choice['gender']}, {choice['f0']} Hz, {choice['seconds']:.1f}s, "
        f"spread {choice['spread']}, quality {choice['quality']})")

    if args.out:
        # Reference must be at least MIN_REF_SECONDS; extend into surrounding
        # audio when the chosen window is short.
        want = max(MIN_REF_SECONDS, choice["seconds"])
        start = max(0.0, min(choice["start"], duration - want))
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", args.audio,
             "-t", f"{want:.3f}", "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le",
             args.out],
            check=True)
        choice["reference"] = args.out

    print(json.dumps(choice))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)
