"""Speak a line in a cloned voice, using Chatterbox.

Runs in its own interpreter (`~/tts-venv`, Python 3.11) because the TTS stack
does not yet build on the 3.14 environment the rest of the app uses. Invoked by
`outro.py` as a subprocess, so nothing here is imported by the main pipeline.

Reads a reference wav of the target speaker and writes the spoken line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def log(msg: str) -> None:
    print(f"[tts] {msg}", file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="wav of the voice to clone")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    # Chatterbox exaggeration/cfg defaults are tuned for expressive reads; a
    # calmer delivery suits a short download prompt.
    ap.add_argument("--exaggeration", type=float, default=0.4)
    ap.add_argument("--cfg-weight", type=float, default=0.5)
    args = ap.parse_args()

    reference = Path(args.reference)
    if not reference.exists():
        raise SystemExit(f"reference not found: {reference}")

    import torch
    from chatterbox.tts import ChatterboxTTS

    # MPS gives a large speedup on Apple silicon but some ops still fall back;
    # CPU is the safe default when it is unavailable.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log(f"loading model on {device}")
    try:
        model = ChatterboxTTS.from_pretrained(device=device)
    except Exception as exc:
        if device == "cpu":
            raise
        log(f"{device} load failed ({exc}); retrying on cpu")
        device = "cpu"
        model = ChatterboxTTS.from_pretrained(device=device)

    log(f"generating: {args.text!r}")
    wav = model.generate(
        args.text,
        audio_prompt_path=str(reference),
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
    )

    import torchaudio
    torchaudio.save(args.out, wav.cpu(), model.sr)
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
