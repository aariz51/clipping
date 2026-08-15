"""Append a branded outro to a rendered clip.

The outro shows the app logo and name, and speaks a download line in the voice
of a speaker taken from the clip itself -- female when the clip contains one,
per the brief.

Pipeline:
  1. `voice_pick.py` selects a reference speaker sample from the clip.
  2. Chatterbox clones that voice to read the download line (see `tts_clone.py`,
     run in its own interpreter because the TTS stack needs Python <= 3.12).
  3. A logo card is rendered with Pillow and encoded for exactly the length of
     the spoken line plus a short hold.
  4. Card and clip are concatenated.

Every stage degrades rather than fails: without a working voice the outro is
rendered silent, and if the outro cannot be built at all the original clip is
returned untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HOLD_SECONDS = 1.1        # beat after the line so the logo does not cut away
MIN_OUTRO_SECONDS = 2.5
MAX_OUTRO_SECONDS = 7.0


def log(msg: str) -> None:
    print(f"[outro] {msg}", file=sys.stderr, flush=True)


def speakable(app_name: str) -> str:
    """Rewrite an app name so the TTS pronounces it correctly.

    Brand names are written for the eye, not the ear. `LabelWise` was read as
    "labor-wise" until the internal capital was split into two words, and a
    colon is voiced as an awkward stop, so it becomes a comma pause.
    """
    import re

    spoken = app_name.replace(":", ",").replace("&", " and ")
    # Split camelCase / PascalCase runs: LabelWise -> Label Wise.
    spoken = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", spoken)
    # Keep initialisms intact but separate them from a following word.
    spoken = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", spoken)
    return re.sub(r"\s+", " ", spoken).strip(" ,")


def probe_float(path: Path, entries: str, stream: bool = False) -> float | None:
    args = ["ffprobe", "-v", "error"]
    if stream:
        args += ["-select_streams", "v:0"]
    args += ["-show_entries", entries, "-of", "csv=p=0", str(path)]
    try:
        out = subprocess.run(args, check=True, text=True, capture_output=True).stdout
        return float(out.strip().split(",")[0])
    except Exception:
        return None


def render_card(logo: Path | None, app_name: str, width: int, height: int,
                out_png: Path) -> bool:
    """Draw the end card: logo above the app name, plus a call to action."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        log(f"Pillow unavailable: {exc}")
        return False

    card = Image.new("RGB", (width, height), (10, 12, 16))
    draw = ImageDraw.Draw(card)

    def font(size: int):
        for path in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                     "/System/Library/Fonts/HelveticaNeue.ttc",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue
        return ImageFont.load_default()

    centre_y = height // 2
    logo_px = int(width * 0.42)

    if logo and logo.exists():
        try:
            art = Image.open(logo).convert("RGBA")
            art.thumbnail((logo_px, logo_px), Image.LANCZOS)
            # Rounded mask so a square icon reads as an app tile.
            mask = Image.new("L", art.size, 0)
            ImageDraw.Draw(mask).rounded_rectangle(
                [0, 0, art.size[0] - 1, art.size[1] - 1],
                radius=int(min(art.size) * 0.22), fill=255)
            if art.mode == "RGBA":
                mask = Image.composite(mask, Image.new("L", art.size, 0),
                                       art.split()[-1])
            card.paste(art, ((width - art.size[0]) // 2,
                             centre_y - art.size[1] - int(height * 0.03)), mask)
        except Exception as exc:
            log(f"could not draw logo: {exc}")

    name_font = font(int(width * 0.082))
    bbox = draw.textbbox((0, 0), app_name, font=name_font)
    while bbox[2] - bbox[0] > width * 0.86 and name_font.size > 20:
        name_font = font(name_font.size - 4)
        bbox = draw.textbbox((0, 0), app_name, font=name_font)
    draw.text(((width - (bbox[2] - bbox[0])) // 2, centre_y + int(height * 0.02)),
              app_name, font=name_font, fill=(255, 255, 255))

    # Where to get it. Kept to one line so it stays readable at phone size.
    cta = "Download on the App Store & Google Play"
    cta_font = font(int(width * 0.042))
    cbox = draw.textbbox((0, 0), cta, font=cta_font)
    while cbox[2] - cbox[0] > width * 0.9 and cta_font.size > 14:
        cta_font = font(cta_font.size - 2)
        cbox = draw.textbbox((0, 0), cta, font=cta_font)
    draw.text(((width - (cbox[2] - cbox[0])) // 2,
               centre_y + int(height * 0.02) + (bbox[3] - bbox[1]) + int(height * 0.035)),
              cta, font=cta_font, fill=(150, 230, 200))

    card.save(out_png)
    return True


def build_outro(card_png: Path, voice_wav: Path | None, seconds: float,
                width: int, height: int, fps: str, out_mp4: Path) -> bool:
    """Encode the still card into a clip, with the spoken line if present."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-loop", "1", "-framerate", fps, "-t", f"{seconds:.3f}", "-i", str(card_png)]
    if voice_wav and voice_wav.exists():
        cmd += ["-i", str(voice_wav)]
    else:
        # Silent track keeps the concat demuxer happy: every segment must have
        # the same stream layout.
        cmd += ["-f", "lavfi", "-t", f"{seconds:.3f}", "-i",
                "anullsrc=channel_layout=stereo:sample_rate=48000"]

    # Gentle zoom so the card is not a dead freeze frame.
    zoom = (f"scale={width*2}:-1,zoompan=z='min(zoom+0.0009,1.12)'"
            f":d={max(1, int(float(eval(fps)) * seconds))}:s={width}x{height}"
            f":fps={fps},setsar=1,format=yuv420p")
    # Pad the voice with silence to the card's full length. Without this the
    # spoken line is shorter than the card and `-shortest` would clip the hold,
    # cutting away from the logo the moment the line ends.
    cmd += ["-filter_complex", f"[0:v]{zoom}[v];[1:a]apad[a]",
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-t", f"{seconds:.3f}", str(out_mp4)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as exc:
        log(f"outro encode failed: {(exc.stderr or '')[-200:]}")
        return False


def concat(clip: Path, outro: Path, out: Path, width: int, height: int, fps: str) -> bool:
    """Join clip and outro, re-encoding so both segments share a format."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(clip), "-i", str(outro),
           "-filter_complex",
           f"[0:v]scale={width}:{height},setsar=1,fps={fps},format=yuv420p[v0];"
           f"[1:v]scale={width}:{height},setsar=1,fps={fps},format=yuv420p[v1];"
           "[0:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a0];"
           "[1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a1];"
           "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]",
           "-map", "[v]", "-map", "[a]",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
           "-movflags", "+faststart", str(out)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as exc:
        log(f"concat failed: {(exc.stderr or '')[-200:]}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--app-name", required=True)
    ap.add_argument("--logo")
    ap.add_argument("--transcript", help="clip-relative words, for speaker windows")
    ap.add_argument("--output", required=True)
    ap.add_argument("--line", help="override the spoken line")
    ap.add_argument("--tts-python", default=os.path.expanduser("~/tts-venv/bin/python"))
    ap.add_argument("--assets", default=str(Path(__file__).parent))
    args = ap.parse_args()

    clip = Path(args.clip).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    if not clip.exists():
        raise SystemExit(f"clip not found: {clip}")

    work = out.parent / f".outro_{out.stem}"
    work.mkdir(parents=True, exist_ok=True)

    width = int(probe_float(clip, "stream=width", stream=True) or 1080)
    height = int(probe_float(clip, "stream=height", stream=True) or 1920)
    fps_raw = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate", "-of", "csv=p=0", str(clip)],
        check=True, text=True, capture_output=True).stdout.strip() or "30000/1001"

    line = args.line or f"Download {speakable(args.app_name)}"

    # 1. Pick whose voice to use.
    reference = work / "reference.wav"
    picker = Path(args.assets) / "voice_pick.py"
    pick_cmd = [sys.executable, str(picker), "--audio", str(clip),
                "--out", str(reference)]
    if args.transcript:
        pick_cmd += ["--transcript", args.transcript]
    voice_info = {}
    try:
        res = subprocess.run(pick_cmd, check=True, text=True, capture_output=True)
        voice_info = json.loads(res.stdout.strip().splitlines()[-1])
        log(f"voice: {voice_info.get('gender')} @ {voice_info.get('f0')} Hz "
            f"({voice_info.get('speakers_found')} speaker(s) found)")
    except Exception as exc:
        log(f"voice selection failed: {exc}")

    # 2. Clone it.
    voice_wav = work / "line.wav"
    cloner = Path(args.assets) / "tts_clone.py"
    if reference.exists() and Path(args.tts_python).exists() and cloner.exists():
        try:
            subprocess.run(
                [args.tts_python, str(cloner), "--reference", str(reference),
                 "--text", line, "--out", str(voice_wav)],
                check=True, text=True, capture_output=True, timeout=900)
            log(f"cloned line: \"{line}\"")
        except subprocess.TimeoutExpired:
            log("voice cloning timed out; outro will be silent")
        except subprocess.CalledProcessError as exc:
            log(f"voice cloning failed: {(exc.stderr or '')[-200:]}")
    else:
        log("voice cloning unavailable; outro will be silent")

    spoken = probe_float(voice_wav, "format=duration") if voice_wav.exists() else None
    seconds = min(MAX_OUTRO_SECONDS,
                  max(MIN_OUTRO_SECONDS, (spoken or 0.0) + HOLD_SECONDS))

    # 3. Card, 4. encode, 5. join.
    card = work / "card.png"
    logo = Path(args.logo).expanduser().resolve() if args.logo else None
    if logo and not logo.exists():
        log(f"logo not found, continuing without it: {logo}")
        logo = None
    if not render_card(logo, args.app_name, width, height, card):
        raise SystemExit("could not render the end card")

    outro_mp4 = work / "outro.mp4"
    if not build_outro(card, voice_wav if voice_wav.exists() else None,
                       seconds, width, height, fps_raw, outro_mp4):
        raise SystemExit("could not build the outro")

    if not concat(clip, outro_mp4, out, width, height, fps_raw):
        raise SystemExit("could not append the outro")

    log(f"appended {seconds:.1f}s outro -> {out.name}")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
