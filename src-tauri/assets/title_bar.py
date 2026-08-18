"""Render a persistent title banner and burn it into a clip.

A short title held at the top of the frame for the whole clip tells a scroller
what the video is about before they have listened to a word -- the pattern used
by the ad in Aariz's reference (a white plate reading "The Truth About Bottled
Water" above the speaker).

Placement is measured, not assumed. The banner must never cover the speaker's
face, so faces are detected across sampled frames and the banner is fitted into
the clear space above the topmost face. If that space is too small the font
shrinks, and only if it still cannot fit does the banner sit at the very top.

Usage:
  title_bar.py --video IN.mp4 --text "The Truth About X" --output OUT.mp4
               [--assets DIR] [--png-only PATH]

Prints the written video path on success.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

# Sampling for face detection. Faces move, so the banner is placed against the
# highest face seen anywhere in the clip rather than in one arbitrary frame.
# Sampled densely on purpose: B-roll is screened to contain no people, so the
# speaker may occupy only a third of the scenes. Ten samples across a 30 second
# clip repeatedly landed entirely on footage and reported "no face", which would
# have let the banner sit wherever it liked.
PROBE_FRAMES = 28
# Gap left between the banner and the topmost face, as a fraction of height.
FACE_CLEARANCE = 0.02
# The banner never grows past this share of the frame.
MAX_BANNER_FRAC = 0.20
MIN_FONT = 34
# Only avoid faces the detector is confident about. Measured on a real clip, a
# 0.63-scoring "face" sat at y=6 inside the letterbox bar where a face cannot
# be, while the genuine speaker detections all scored 0.86-0.92 around y=281+.
# Treating that noise as a face made the constraint unsatisfiable and would have
# pushed the banner off the frame entirely.
FACE_SCORE_MIN = 0.80


def log(msg: str) -> None:
    print(f"[title] {msg}", file=sys.stderr, flush=True)


def pick_font(size: int):
    from PIL import ImageFont

    import os
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def probe(video: Path) -> tuple[int, int, float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-show_entries", "format=duration",
         "-of", "json", str(video)],
        check=True, text=True, capture_output=True).stdout
    data = json.loads(out)
    stream = (data.get("streams") or [{}])[0]
    return (int(stream.get("width") or 1080),
            int(stream.get("height") or 1920),
            float((data.get("format") or {}).get("duration") or 0.0))


def topmost_face(video: Path, assets: Path, duration: float, height: int) -> int | None:
    """Smallest y of any face across sampled frames, or None if no face is seen.

    Returning None means the banner may use the full top margin -- there is no
    face to avoid.
    """
    weights = assets / "face_detection_yunet_2023mar.onnx"
    if not weights.exists():
        log("face model missing; placing banner in the top margin only")
        return None
    try:
        import cv2
        import numpy as np
    except Exception as exc:
        log(f"OpenCV unavailable ({exc}); placing banner in the top margin only")
        return None

    try:
        detector = cv2.FaceDetectorYN.create(str(weights), "", (320, 320), 0.6)
    except Exception as exc:
        log(f"detector unavailable ({exc})")
        return None

    # YuNet has an operating range: a face filling much of a 1080x1920 frame is
    # too large for it and detection returns nothing at all. Measured on a real
    # clip, full resolution found 0 faces while the same frame downscaled to 480
    # or 720 wide found them reliably. So sweep several scales and keep the
    # highest face any of them sees -- missing a face here would put the banner
    # over the speaker, which is the one outcome to avoid.
    scales = (720, 480, 320)
    top = None
    for i in range(PROBE_FRAMES):
        t = duration * (i + 0.5) / PROBE_FRAMES
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(video),
             "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
            capture_output=True).stdout
        if not raw:
            continue
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        h0, w0 = frame.shape[:2]
        if not h0 or not w0:
            continue
        for target_w in scales:
            if target_w > w0:
                continue
            scale = target_w / w0
            small = cv2.resize(frame, (target_w, max(1, int(h0 * scale))))
            try:
                detector.setInputSize((small.shape[1], small.shape[0]))
                _, faces = detector.detect(small)
            except Exception:
                continue
            if faces is None:
                continue
            for face in faces:
                if float(face[-1]) < FACE_SCORE_MIN:
                    continue
                # Back to full-resolution coordinates, then to the target frame.
                y_full = int(face[1] / scale)
                y = int(y_full * height / h0) if h0 else y_full
                top = y if top is None else min(top, y)
    return top


def wrap(draw, text: str, font, max_width: int) -> list[str]:
    words = text.split()
    lines, current = [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def render_banner(text: str, width: int, height: int, limit_y: int) -> "object":
    """Build a full-frame RGBA overlay holding the title plate.

    `limit_y` is the lowest pixel the banner may occupy. The font shrinks until
    the plate fits above it, so a tight frame yields a smaller title rather than
    a covered face.
    """
    from PIL import Image, ImageDraw

    text = " ".join(text.split()).upper()
    margin_x = int(width * 0.05)
    top_y = int(height * 0.025)
    max_text_w = width - 2 * margin_x - int(width * 0.06)

    size = int(height * 0.045)
    while size >= MIN_FONT:
        font = pick_font(size)
        probe_img = Image.new("RGBA", (8, 8))
        draw = ImageDraw.Draw(probe_img)
        lines = wrap(draw, text, font, max_text_w)
        if len(lines) > 2:
            size -= 4
            continue
        line_h = int(size * 1.22)
        pad_y = int(size * 0.42)
        plate_h = len(lines) * line_h + 2 * pad_y
        if top_y + plate_h <= limit_y and plate_h <= height * MAX_BANNER_FRAC:
            break
        size -= 4
    else:
        # Nothing fits the clear space; use the smallest size and accept the
        # top margin. Better a small title than none.
        font = pick_font(MIN_FONT)
        probe_img = Image.new("RGBA", (8, 8))
        draw = ImageDraw.Draw(probe_img)
        lines = wrap(draw, text, font, max_text_w)[:2]
        line_h = int(MIN_FONT * 1.22)
        pad_y = int(MIN_FONT * 0.42)
        plate_h = len(lines) * line_h + 2 * pad_y

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    # Text only -- no plate behind it. A filled white box reads as a sticker
    # pasted on the video; bare type sits in the frame. Legibility over
    # unpredictable footage comes from a heavy outline plus a soft shadow
    # instead of a background fill.
    stroke = max(2, int(size * 0.085))
    y = top_y + pad_y
    for line in lines:
        w = draw.textlength(line, font=font)
        x = (width - w) / 2
        # Shadow first, slightly offset, to lift the type off busy footage.
        d.text((x + stroke * 0.9, y + stroke * 0.9), line, font=font,
               fill=(0, 0, 0, 140))
        d.text((x, y), line, font=font, fill=(255, 255, 255, 255),
               stroke_width=stroke, stroke_fill=(0, 0, 0, 235))
        y += line_h

    return overlay, top_y + plate_h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--output")
    ap.add_argument("--assets", default=str(Path(__file__).parent))
    ap.add_argument("--png-only", help="write the overlay PNG here and stop")
    args = ap.parse_args()

    video = Path(args.video).expanduser().resolve()
    if not video.exists():
        raise SystemExit(f"video not found: {video}")
    assets = Path(args.assets).expanduser().resolve()

    width, height, duration = probe(video)
    face_top = topmost_face(video, assets, duration, height)
    if face_top is None:
        limit_y = int(height * MAX_BANNER_FRAC)
        log(f"no face detected; banner limited to top {MAX_BANNER_FRAC:.0%}")
    else:
        limit_y = max(int(height * 0.06), face_top - int(height * FACE_CLEARANCE))
        log(f"topmost face at y={face_top}; banner must end above y={limit_y}")

    # Two modes, chosen by measurement:
    #
    #   overlay - a face leaves clear space at the top, so the plate is drawn
    #             straight onto the picture.
    #   band    - the face sits too high to clear (measured y=71 on a real cut),
    #             so the picture is shifted down into a shorter frame and the
    #             title gets a dedicated band above it. This is the layout in
    #             the reference ad, and it makes covering the face impossible
    #             rather than merely unlikely.
    overlay, bottom = render_banner(args.text, width, height, limit_y)
    use_band = face_top is not None and bottom > face_top

    if use_band:
        band = min(int(height * MAX_BANNER_FRAC), bottom + int(height * 0.01))
        log(f"face at y={face_top} leaves no clear space; using a {band}px title band")
        # Re-render the plate against the band, vertically centred in it.
        overlay, bottom = render_banner(args.text, width, height, band)
    else:
        log(f"banner occupies y=0..{bottom} (face clear)")

    if args.png_only:
        overlay.save(args.png_only)
        print(args.png_only)
        return 0

    import tempfile
    png = Path(tempfile.gettempdir()) / f"{video.stem}_title.png"
    overlay.save(png)

    output = Path(args.output) if args.output else video.with_name(
        f"{video.stem}_titled.mp4")

    # Speaker footage is a 9:16 crop of a 16:9 source, so it is only ~607px
    # wide before being blown up to 1080 -- measured at a 6x sharpness drop
    # against the native cut, while B-roll (natively 1080+) stays crisp. That
    # upscale cannot be undone, but a mild unsharp mask restores much of the
    # perceived detail. Kept gentle (0.7) so already-sharp B-roll does not halo.
    SHARPEN = "unsharp=5:5:0.7:5:5:0.0"

    if use_band:
        # Trim an equal strip off the bottom and push the picture down, so the
        # frame size is unchanged and nothing in the upper picture is hidden.
        # The bottom strip is safe to lose: captions sit around 0.65-0.72 of the
        # height, well above the trimmed region.
        vf = (f"crop={width}:{height - band}:0:0,"
              f"pad={width}:{height}:0:{band}:black,{SHARPEN}")
        filt = f"[0:v]{vf}[base];[base][1:v]overlay=0:0:format=auto[v]"
    else:
        filt = f"[0:v]{SHARPEN}[base];[base][1:v]overlay=0:0:format=auto[v]"

    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video), "-i", str(png),
        "-filter_complex", filt,
        "-map", "[v]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "slow", "-crf", "16",
        "-pix_fmt", "yuv420p",
        # Audio untouched, so the speaker and any cloned outro line are intact.
        "-c:a", "copy", "-movflags", "+faststart", str(output),
    ], check=True, capture_output=True, text=True)

    png.unlink(missing_ok=True)
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
