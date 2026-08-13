"""Render burned-in captions as transparent PNG overlays.

ffmpeg's `drawtext` filter requires a build with libfreetype, which many
distributions (including current Homebrew) omit. `overlay` is a core filter
present in every build, so captions are rendered here with Pillow and
composited as an image sequence instead.

Reads a JSON spec on stdin, writes PNGs plus an ffmpeg concat list into the
requested directory, and prints the concat list path on success. Prints
nothing and exits non-zero on failure, so the caller can fall back.

Spec:
  {"width": 1214, "height": 2160, "duration": 96.0, "style": "classic-outline",
   "chunks": [{"text": "SO I WAS", "start": 0.0, "end": 0.6}, ...],
   "out_dir": "/tmp/..."}
"""

import json
import os
import sys

# Font candidates in preference order: a heavy weight reads best at small sizes
# on a phone screen.
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

# Mirrors the drawtext styles so switching renderers does not change the look.
# fill/stroke/box are RGBA; box=None means no background plate.
STYLES = {
    "classic-outline": {"fill": (255, 255, 0, 255), "stroke": (0, 0, 0, 255), "box": None, "y": 0.65},
    "modern-box":      {"fill": (255, 255, 255, 255), "stroke": None, "box": (0, 0, 0, 176), "y": 0.72},
    "minimal-shadow":  {"fill": (255, 255, 255, 255), "stroke": None, "box": None, "y": 0.70, "shadow": True},
    "vibrant-cyan":    {"fill": (0, 255, 255, 255), "stroke": None, "box": None, "y": 0.70, "shadow": True},
    "vibrant-yellow-box": {"fill": (0, 0, 0, 255), "stroke": None, "box": (255, 255, 0, 224), "y": 0.72},
    "vibrant-green":   {"fill": (57, 255, 20, 255), "stroke": (0, 0, 0, 255), "box": None, "y": 0.70},
    "vibrant-red":     {"fill": (255, 59, 48, 255), "stroke": (0, 0, 0, 255), "box": None, "y": 0.70},
}


def pick_font(size):
    from PIL import ImageFont

    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_chunk(text, width, height, style, font):
    """One transparent frame carrying a single caption."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if not text:
        return img

    draw = ImageDraw.Draw(img)
    stroke_w = max(2, round(font.size * 0.08)) if style.get("stroke") else 0

    # Wrap to keep captions clear of the frame edges.
    max_w = int(width * 0.86)
    words = text.split()
    lines, current = [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), trial, font=font, stroke_width=stroke_w)
        if bbox[2] - bbox[0] > max_w and current:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)

    line_h = int(font.size * 1.2)
    total_h = line_h * len(lines)
    y = int(height * style.get("y", 0.7)) - total_h // 2

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font, stroke_width=stroke_w)
        w = bbox[2] - bbox[0]
        x = (width - w) // 2

        if style.get("box"):
            pad_x = int(font.size * 0.35)
            pad_y = int(font.size * 0.18)
            draw.rectangle(
                [x - pad_x, y - pad_y, x + w + pad_x, y + line_h + pad_y // 2],
                fill=style["box"],
            )
        if style.get("shadow"):
            off = max(2, int(font.size * 0.05))
            draw.text((x + off, y + off), line, font=font, fill=(0, 0, 0, 140))

        draw.text(
            (x, y), line, font=font, fill=style["fill"],
            stroke_width=stroke_w, stroke_fill=style.get("stroke"),
        )
        y += line_h

    return img


def main():
    try:
        spec = json.load(sys.stdin)
    except Exception as exc:
        print("bad spec: {}".format(exc), file=sys.stderr)
        return 1

    try:
        from PIL import Image  # noqa: F401
    except Exception as exc:
        print("Pillow unavailable: {}".format(exc), file=sys.stderr)
        return 1

    width = int(spec["width"])
    height = int(spec["height"])
    duration = float(spec["duration"])
    out_dir = spec["out_dir"]
    chunks = spec.get("chunks") or []
    style = STYLES.get(spec.get("style"), STYLES["modern-box"])

    if width <= 0 or height <= 0 or duration <= 0 or not chunks:
        print("nothing to render", file=sys.stderr)
        return 1

    os.makedirs(out_dir, exist_ok=True)

    # Scale with frame width so captions read the same on any resolution.
    font = pick_font(max(28, min(96, int(width * 0.072))))

    # Identical text reuses one PNG: two-word chunks repeat a lot across a clip.
    cache = {}

    def png_for(text):
        key = text or ""
        if key not in cache:
            path = os.path.join(out_dir, "cap_{:04d}.png".format(len(cache)))
            render_chunk(key, width, height, style, font).save(path)
            cache[key] = path
        return cache[key]

    blank = png_for("")

    # Build a gapless timeline: concat has no notion of absent frames, so silent
    # stretches are explicit transparent entries.
    timeline = []
    cursor = 0.0
    for chunk in sorted(chunks, key=lambda c: c["start"]):
        start = max(0.0, float(chunk["start"]))
        end = min(duration, float(chunk["end"]))
        if end <= start:
            continue
        if start > cursor + 0.01:
            timeline.append((blank, start - cursor))
        text = (chunk.get("text") or "").strip()
        timeline.append((png_for(text), end - max(start, cursor)))
        cursor = end

    if cursor < duration:
        timeline.append((blank, duration - cursor))

    if not timeline:
        print("empty timeline", file=sys.stderr)
        return 1

    list_path = os.path.join(out_dir, "captions.txt")
    with open(list_path, "w") as fh:
        for path, dur in timeline:
            fh.write("file '{}'\n".format(path.replace("'", "'\\''")))
            fh.write("duration {:.3f}\n".format(max(dur, 0.02)))
        # concat needs the final entry repeated for its duration to apply.
        fh.write("file '{}'\n".format(timeline[-1][0].replace("'", "'\\''")))

    print(list_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("captions failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
