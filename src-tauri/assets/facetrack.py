"""Speaker-aware crop tracking for AutoShorts.

Samples frames across a clip range, detects the primary speaker's face with
YuNet, and emits either a static crop offset or a piecewise-linear ffmpeg
`crop` x/y expression that follows the speaker.

Prints a single JSON object to stdout. Any failure prints {"mode": "none"} so
the caller can fall back to the plain centre crop.
"""

import argparse
import json
import sys


def log(msg):
    print(msg, file=sys.stderr)


def emit(payload):
    print(json.dumps(payload))
    sys.exit(0)


def pick_face(faces, prev_cx, prev_cy, frame_w):
    """Choose the primary speaker among detections.

    Scores by detector confidence and face area, then penalises jumping away
    from the previously tracked face so a background listener does not steal
    the track mid-sentence.
    """
    best, best_score = None, -1.0
    for f in faces:
        x, y, w, h = float(f[0]), float(f[1]), float(f[2]), float(f[3])
        conf = float(f[14])
        if w <= 0 or h <= 0:
            continue
        cx, cy = x + w / 2.0, y + h / 2.0
        # Area normalised against frame width so the score is resolution independent.
        area = (w * h) / float(frame_w * frame_w)
        score = conf + 6.0 * area
        if prev_cx is not None:
            drift = abs(cx - prev_cx) / float(frame_w)
            score -= 1.5 * drift
        if score > best_score:
            best, best_score = (cx, cy, w, h), score
    return best


def sample_track(cap, cv2, detector, start, end, sample_fps, frame_w, frame_h):
    """Walk the clip sequentially, returning [(t_relative, cx, cy)] samples."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0 or fps > 240:
        fps = 30.0
    stride = max(1, int(round(fps / float(sample_fps))))

    # Seek to the clip start once, then read sequentially (far faster than
    # per-sample seeking, and avoids keyframe seek drift).
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)

    # Detection runs on a downscaled frame for speed; coordinates are scaled back.
    det_w = 480
    scale = det_w / float(frame_w)
    det_h = max(1, int(round(frame_h * scale)))
    detector.setInputSize((det_w, det_h))

    samples = []
    prev_cx = prev_cy = None
    idx = 0
    max_frames = int((end - start) * fps) + 2

    while idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            t_rel = max(0.0, pos - start)
            if t_rel > (end - start):
                break
            small = cv2.resize(frame, (det_w, det_h))
            try:
                _, faces = detector.detect(small)
            except Exception:
                faces = None
            if faces is not None and len(faces):
                prev_small_cx = prev_cx * scale if prev_cx is not None else None
                prev_small_cy = prev_cy * scale if prev_cy is not None else None
                best = pick_face(faces, prev_small_cx, prev_small_cy, det_w)
                if best is not None:
                    cx = best[0] / scale
                    cy = best[1] / scale
                    samples.append((t_rel, cx, cy))
                    prev_cx, prev_cy = cx, cy
        idx += 1

    return samples


def find_cuts(values, threshold):
    """Indices where the track jumps far enough to be a camera cut, not motion.

    A person cannot move a quarter of the frame between two samples, so a jump
    that large means the shot changed. Those boundaries must not be smoothed
    across, otherwise the crop slowly slides after every cut.
    """
    return [i for i in range(1, len(values))
            if abs(values[i] - values[i - 1]) > threshold]


def smooth(values, window, cuts=()):
    """Centred moving average that never averages across a cut boundary."""
    if window <= 1 or len(values) < 3:
        return list(values)
    # Segment bounds: [start, end) runs of continuous shot.
    bounds = [0] + list(cuts) + [len(values)]
    out = [0.0] * len(values)
    half = window // 2
    for b in range(len(bounds) - 1):
        seg_lo, seg_hi = bounds[b], bounds[b + 1]
        for i in range(seg_lo, seg_hi):
            lo = max(seg_lo, i - half)
            hi = min(seg_hi, i + half + 1)
            chunk = values[lo:hi]
            out[i] = sum(chunk) / float(len(chunk))
    return out


def build_keyframes(times, positions, max_keys, deadzone, cuts=()):
    """Reduce a dense track to a small set of keyframes.

    Drops points that a straight line between neighbours already predicts
    within `deadzone` pixels, which keeps the ffmpeg expression short and
    removes micro-jitter. Cut boundaries are always kept, and are bracketed by
    a pair of near-adjacent keys so the crop snaps rather than pans.
    """
    if not times:
        return []
    cutset = set(cuts)
    protected_times = set()
    keys = [(times[0], positions[0])]
    for i in range(1, len(times) - 1):
        if i in cutset:
            # Hold the outgoing framing right up to the cut, then jump.
            hold = max(keys[-1][0] + 1e-3, times[i] - 0.04)
            keys.append((hold, positions[i - 1]))
            keys.append((times[i], positions[i]))
            protected_times.add(hold)
            protected_times.add(times[i])
            continue
        t_prev, x_prev = keys[-1]
        span = times[i + 1] - t_prev
        if span <= 0:
            continue
        # Linear prediction from the last kept key to the next sample.
        ratio = (times[i] - t_prev) / span
        predicted = x_prev + (positions[i + 1] - x_prev) * ratio
        if abs(positions[i] - predicted) > deadzone:
            keys.append((times[i], positions[i]))
    keys.append((times[-1], positions[-1]))

    # Hard cap: thin uniformly if still too many. Cut snaps are load-bearing,
    # so drop only the interpolated motion keys between them.
    if len(keys) > max_keys:
        protected = [k for k in keys if k[0] in protected_times]
        movable = [k for k in keys if k[0] not in protected_times]
        room = max(2, max_keys - len(protected))
        if len(movable) > room:
            step = len(movable) / float(room)
            movable = [movable[int(i * step)] for i in range(room)]
        keys = sorted(set(protected + movable + [keys[0], keys[-1]]))
    return keys


def build_expr(keys, dim_in, dim_out):
    """Piecewise-linear ffmpeg expression in `t`, clamped to valid crop range."""
    if not keys:
        return None
    if len(keys) == 1:
        body = "{:.2f}".format(keys[0][1])
    else:
        body = "{:.2f}".format(keys[-1][1])
        for i in range(len(keys) - 2, -1, -1):
            t0, x0 = keys[i]
            t1, x1 = keys[i + 1]
            if t1 - t0 <= 1e-6:
                continue
            seg = "{:.2f}+({:.2f})*(t-{:.3f})".format(x0, (x1 - x0) / (t1 - t0), t0)
            body = "if(lt(t,{:.3f}),{},{})".format(t1, seg, body)
    return "max(0\\,min({}-{}\\,{}))".format(dim_in, dim_out, body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--sample-fps", type=float, default=4.0)
    args = ap.parse_args()

    try:
        import cv2
    except Exception as e:
        log("opencv unavailable: {}".format(e))
        emit({"mode": "none", "reason": "opencv-missing"})

    duration = args.end - args.start
    if duration <= 0:
        emit({"mode": "none", "reason": "empty-range"})

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        emit({"mode": "none", "reason": "open-failed"})

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if frame_w <= 0 or frame_h <= 0:
        cap.release()
        emit({"mode": "none", "reason": "no-dimensions"})

    # Target 9:16 window, matching the even-width rounding ffmpeg uses.
    out_w = min(frame_w, int(frame_h * 9 / 16))
    out_h = min(frame_h, int(frame_w * 16 / 9))
    out_w -= out_w % 2
    out_h -= out_h % 2

    if out_w >= frame_w and out_h >= frame_h:
        cap.release()
        emit({"mode": "none", "reason": "already-vertical"})

    try:
        detector = cv2.FaceDetectorYN.create(args.model, "", (320, 320), 0.6, 0.3, 5000)
    except Exception as e:
        cap.release()
        log("detector init failed: {}".format(e))
        emit({"mode": "none", "reason": "detector-failed"})

    samples = sample_track(cap, cv2, detector, args.start, args.end,
                           args.sample_fps, frame_w, frame_h)
    cap.release()

    # Require a reasonable hit rate before trusting the track.
    expected = max(1.0, duration * args.sample_fps)
    coverage = len(samples) / expected
    if len(samples) < 3 or coverage < 0.15:
        emit({"mode": "none", "reason": "insufficient-faces",
              "samples": len(samples), "coverage": round(coverage, 3)})

    times = [s[0] for s in samples]
    # Convert face centres into crop origins, then clamp into frame.
    raw_x = [min(max(s[1] - out_w / 2.0, 0.0), frame_w - out_w) for s in samples]
    raw_y = [min(max(s[2] - out_h / 2.0, 0.0), frame_h - out_h) for s in samples]

    # Shot changes are found on the raw track and honoured by both the smoother
    # and the keyframe builder, so the crop snaps at cuts instead of drifting.
    cuts = find_cuts(raw_x, out_w * 0.25)

    # Window scales with sample rate: ~1.5s of context smooths natural head motion.
    win = max(3, int(args.sample_fps * 1.5) | 1)
    sx = smooth(raw_x, win, cuts)
    sy = smooth(raw_y, win, cuts)

    horizontal = frame_w - out_w > 2
    vertical = frame_h - out_h > 2

    result = {"mode": "static", "out_w": out_w, "out_h": out_h,
              "samples": len(samples), "coverage": round(coverage, 3),
              "cuts": len(cuts)}

    # If the speaker barely moves, a fixed offset beats a dynamic expression:
    # no risk of drift, and the crop reads as a deliberate framing choice.
    travel_x = max(sx) - min(sx) if horizontal else 0.0
    travel_y = max(sy) - min(sy) if vertical else 0.0
    static_threshold = out_w * 0.06

    if travel_x <= static_threshold and travel_y <= static_threshold:
        result["x"] = int(round(sum(sx) / len(sx))) if horizontal else 0
        result["y"] = int(round(sum(sy) / len(sy))) if vertical else 0
        emit(result)

    deadzone = max(4.0, out_w * 0.015)
    result["mode"] = "dynamic"
    if horizontal:
        keys_x = build_keyframes(times, sx, 48, deadzone, cuts)
        result["x_expr"] = build_expr(keys_x, "in_w", "out_w")
        result["keys_x"] = len(keys_x)
    else:
        result["x"] = 0
    if vertical:
        keys_y = build_keyframes(times, sy, 48, deadzone, cuts)
        result["y_expr"] = build_expr(keys_y, "in_h", "out_h")
        result["keys_y"] = len(keys_y)
    else:
        result["y"] = 0

    emit(result)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        log("facetrack failed: {}".format(exc))
        print(json.dumps({"mode": "none", "reason": "exception"}))
