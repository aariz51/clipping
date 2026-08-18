"""Synthesise a royalty-free sound-effect kit.

Generated from first principles rather than downloaded, for two reasons: the
named references (Among Us, Vine Boom) are copyrighted game and meme audio that
would expose a monetised account to claims, and synthesis needs no network, no
API key and no licence attribution. These are the same primitives the originals
are built from -- a swept riser, a filtered whoosh, a low boom -- so they serve
the same editorial purpose.

Writes 16-bit 44.1kHz mono WAVs into the target directory.
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import wave
from pathlib import Path

SR = 44100


def _write(path: Path, samples: list[float]) -> None:
    peak = max(1e-9, max(abs(s) for s in samples))
    norm = [s / peak * 0.89 for s in samples]
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b"".join(
            struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32767)) for s in norm))
    print(f"[sfx] wrote {path.name} ({len(samples)/SR:.2f}s)", file=sys.stderr)


def _noise(seed: int):
    """Deterministic white noise. A fixed seed keeps renders reproducible."""
    state = seed
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield (state / 0x3FFFFFFF) - 1.0


def riser(seconds: float = 2.0) -> list[float]:
    """Rising tone + noise sweep. Builds tension into a reveal."""
    n = int(SR * seconds)
    rng = _noise(7)
    out, phase = [], 0.0
    for i in range(n):
        t = i / n
        freq = 180 * (2 ** (3.2 * t))          # ~180Hz -> ~1.7kHz
        phase += 2 * math.pi * freq / SR
        env = t ** 1.7                          # slow start, hard finish
        hiss = next(rng) * 0.35 * (t ** 2.5)
        out.append((math.sin(phase) * 0.55 + hiss) * env)
    # Short fade-out so it butts cleanly against the next beat.
    for i in range(int(SR * 0.04)):
        out[-1 - i] *= i / (SR * 0.04)
    return out


def whoosh(seconds: float = 0.55) -> list[float]:
    """Band-swept noise. Smooths a hard cut between scenes."""
    n = int(SR * seconds)
    rng = _noise(11)
    out, lp = [], 0.0
    for i in range(n):
        t = i / n
        # One-pole low-pass whose cutoff sweeps up then down.
        cutoff = 0.02 + 0.55 * math.sin(math.pi * t)
        lp += cutoff * (next(rng) - lp)
        env = math.sin(math.pi * t) ** 1.4
        out.append(lp * env)
    return out


def boom(seconds: float = 0.9) -> list[float]:
    """Low sine drop. Lands weight under an important statement."""
    n = int(SR * seconds)
    out, phase = [], 0.0
    for i in range(n):
        t = i / n
        freq = 118 * math.exp(-3.1 * t) + 34    # pitch drops fast
        phase += 2 * math.pi * freq / SR
        env = math.exp(-4.2 * t)
        click = math.exp(-260 * t) * 0.5        # transient so it cuts through
        out.append(math.sin(phase) * env + click)
    return out


def attention(seconds: float = 0.7) -> list[float]:
    """Bright two-tone stab. Grabs the eye at the hook."""
    n = int(SR * seconds)
    out = []
    for i in range(n):
        t = i / n
        env = math.exp(-6.5 * t)
        a = math.sin(2 * math.pi * 880 * i / SR)
        b = math.sin(2 * math.pi * 1320 * i / SR) * 0.6
        out.append((a + b) * env)
    return out


def stinger(seconds: float = 0.8) -> list[float]:
    """Dissonant stab for something unexpected."""
    n = int(SR * seconds)
    out = []
    for i in range(n):
        t = i / n
        env = math.exp(-5.0 * t)
        a = math.sin(2 * math.pi * 440 * i / SR)
        b = math.sin(2 * math.pi * 622 * i / SR)   # tritone against the root
        out.append((a + b * 0.85) * env * 0.7)
    return out


def pop(seconds: float = 0.25) -> list[float]:
    """Short comedic pop, for a light beat."""
    n = int(SR * seconds)
    out, phase = [], 0.0
    for i in range(n):
        t = i / n
        freq = 420 * math.exp(-6.0 * t) + 150
        phase += 2 * math.pi * freq / SR
        out.append(math.sin(phase) * math.exp(-13.0 * t))
    return out


KIT = {
    "riser": (riser, "build suspense into a reveal"),
    "whoosh": (whoosh, "smooth a transition between scenes"),
    "boom": (boom, "weight under an important statement"),
    "attention": (attention, "grab attention on the hook"),
    "stinger": (stinger, "mark something unexpected"),
    "pop": (pop, "light comedic beat"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory to write the kit into")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for name, (fn, purpose) in KIT.items():
        _write(out / f"{name}.wav", fn())
        print(f"  {name:10s} {purpose}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
