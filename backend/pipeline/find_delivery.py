"""Find the delivery inside a long clip and cut it out.

The wicket clipper hands us a ~35 s window of the match stream. Somewhere in
it is one bowled ball; the rest is the bowler walking back, fielders, the
celebration. The tracker (`process_video`) has only ever been run on short
single-delivery clips, so we trim the window down to that one ball first.

How the delivery is recognised
------------------------------
1. Run the ball detector on a sample of frames (FINDER_SAMPLE_HZ per second),
   looking only at the corridor around the calibrated pitch.
2. Look for a *release*: a ball detection where a bowler's hand would be —
   1.4–3.2 m above the bowler-end stumps and within 1.8 m either side —
   that is followed, over the next fraction of a second, by detections that
   chain together and travel DOWN the image (the ball dropping to the pitch).
   A ball tossed up in celebration or thrown back to the bowler does not
   start in that box and descend, so it is not picked.
3. Score each such release, keep the best ones.
4. Cut a couple of seconds either side with ffmpeg.

This module is self-contained on purpose: it does not import the legacy
Zone-div tracker modules (they hold per-frame filter state that assumes
consecutive frames), so it can run in the consumer process directly.

Usage:
    python -m pipeline.find_delivery --video window.mp4 --weights models/x.onnx \
        --calibration calib.json --out delivery.mp4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

# ── Tunables (all overridable by env so they can be tuned on real footage) ────
SAMPLE_HZ = float(os.environ.get("FINDER_SAMPLE_HZ", "15"))      # frames/s we run the detector on
CONF_MIN = float(os.environ.get("FINDER_CONF_MIN", "0.40"))      # ball confidence floor
TOP_K = 3                                                         # candidates kept per frame
LEAD_SEC = float(os.environ.get("FINDER_LEAD_SEC", "2.0"))       # kept before the release
TAIL_SEC = float(os.environ.get("FINDER_TAIL_SEC", "2.5"))       # kept after the last tracked point
MAX_CLIP_SEC = 10.0

# Geometry, in metres at the bowler's end (converted to pixels per clip).
_RETURN_CREASE_M = 2.64          # distance between the two calibrated crease corners
RELEASE_HEIGHT_M = (1.4, 3.2)    # hand height above the stump base at release
RELEASE_SIDE_M = 1.8             # how far either side of middle stump
MIN_DESCENT_M = 0.08             # ball must drop at least this far (in image) after release
MIN_BOUNCE_RISE_M = 0.25         # ...and coming back up this far afterwards marks a bounce
CHAIN_SEC = 1.2                  # how long after release we follow the ball
LINK_SPEED_M_S = 8.0             # max apparent ball speed when chaining detections
LINK_SLACK_M = 0.35
MIN_CHAIN_POINTS = 3
MODEL_IMGSZ = 1280               # scale the tracker's own detector runs at


@dataclass
class Candidate:
    release_sec: float           # time of the release inside the window
    end_sec: float               # last chained ball detection
    score: float
    n_points: int
    mean_conf: float
    descent_m: float
    points: list = field(default_factory=list)   # [(sec, x, y, conf)]

    @property
    def clip_start(self) -> float:
        return max(0.0, self.release_sec - LEAD_SEC)

    def clip_end(self, duration: float) -> float:
        end = min(duration, self.end_sec + TAIL_SEC)
        return min(end, self.clip_start + MAX_CLIP_SEC)


# ── ffmpeg ────────────────────────────────────────────────────────────────────
def find_ffmpeg() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


def cut_clip(src: Path, dst: Path, start: float, end: float) -> None:
    """Frame-accurate cut. Re-encodes at near-lossless quality so the cut
    lands exactly where asked regardless of where the keyframes are."""
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found (install it or `pip install imageio-ffmpeg`)")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{end - start:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "14", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-movflags", "+faststart",
        str(dst),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not dst.exists():
        raise RuntimeError(f"ffmpeg cut failed: {res.stderr[-500:]}")


# ── Geometry from the calibration ─────────────────────────────────────────────
def _geometry(calibration: dict, width: int, height: int) -> dict:
    bl, br = calibration["bowl_L"], calibration["bowl_R"]
    tl, tr = calibration["bat_L"], calibration["bat_R"]
    crease_w = math.hypot(br[0] - bl[0], br[1] - bl[1])
    if crease_w < 10:
        raise ValueError("calibration: bowler-end crease is degenerate")
    ppm = crease_w / _RETURN_CREASE_M            # pixels per metre at the bowler's end
    stump = calibration.get("bowl_stump") or [(bl[0] + br[0]) / 2, (bl[1] + br[1]) / 2]
    bat_y = min(tl[1], tr[1])
    xs = [bl[0], br[0], tl[0], tr[0]]
    # Corridor the detector looks at: wide enough for the bowler's arm and a
    # ball leaving the bat, tall enough for the release and the batter.
    x0 = max(0, int(min(xs) - 2.2 * ppm))
    x1 = min(width, int(max(xs) + 2.2 * ppm))
    y0 = max(0, int(min(stump[1] - 3.8 * ppm, bat_y - 1.6 * ppm)))
    y1 = min(height, int(stump[1] + 0.4 * ppm))
    return {"ppm": ppm, "stump": (float(stump[0]), float(stump[1])),
            "crop": (x0, y0, x1, y1)}


# ── Detector (ball only, no temporal filtering) ───────────────────────────────
class BallDetector:
    def __init__(self, weights_path: Path):
        import onnxruntime as ort
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(weights_path), providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        h, w = inp.shape[2], inp.shape[3]
        # Models exported with dynamic=True accept any size, so we can feed
        # just the pitch corridor (several times faster). Fixed-size exports
        # get the whole frame letterboxed, exactly like the tracker does.
        self.fixed_size = (int(h), int(w)) if isinstance(h, int) and isinstance(w, int) else None

    def _run(self, img: np.ndarray) -> np.ndarray:
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis]
        out = self.session.run(None, {self.input_name: blob})[0][0]
        if out.shape[0] < out.shape[1]:          # YOLOv8: [4+nc, N] -> [N, 4+nc]
            out = out.T
            scores = out[:, 4]                   # class 0 = ball
        else:                                    # YOLOv5: [N, 5+nc] with objectness
            cls = out[:, 5] if out.shape[1] > 5 else 1.0
            scores = out[:, 4] * cls
        if scores.size and (scores.min() < 0.0 or scores.max() > 1.0):
            scores = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack([out[:, 0], out[:, 1], scores])   # cx, cy, conf

    def detect(self, frame: np.ndarray, crop: tuple[int, int, int, int]) -> list[tuple[float, float, float]]:
        """Ball candidates in full-frame pixel coords: [(x, y, conf)], best first."""
        fh, fw = frame.shape[:2]
        if self.fixed_size is None:
            x0, y0, x1, y1 = crop
            ratio = MODEL_IMGSZ / max(fw, fh)    # same scale the tracker's letterbox uses
            roi = frame[y0:y1, x0:x1]
            rw = max(32, int(round(roi.shape[1] * ratio)))
            rh = max(32, int(round(roi.shape[0] * ratio)))
            roi = cv2.resize(roi, (rw, rh), interpolation=cv2.INTER_LINEAR)
            pw, ph = (-rw) % 32, (-rh) % 32      # pad to a multiple of the model stride
            img = cv2.copyMakeBorder(roi, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            dets = self._run(img)
            dets[:, 0] = dets[:, 0] / ratio + x0
            dets[:, 1] = dets[:, 1] / ratio + y0
        else:
            th, tw = self.fixed_size
            ratio = min(tw / fw, th / fh)
            nw, nh = int(round(fw * ratio)), int(round(fh * ratio))
            dw, dh = (tw - nw) // 2, (th - nh) // 2
            img = np.full((th, tw, 3), 114, dtype=np.uint8)
            img[dh:dh + nh, dw:dw + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            dets = self._run(img)
            dets[:, 0] = (dets[:, 0] - dw) / ratio
            dets[:, 1] = (dets[:, 1] - dh) / ratio

        dets = dets[dets[:, 2] >= CONF_MIN]
        if not len(dets):
            return []
        dets = dets[np.argsort(-dets[:, 2])]
        kept: list[tuple[float, float, float]] = []
        for x, y, c in dets:                     # cheap NMS: drop near-duplicates
            if all((x - kx) ** 2 + (y - ky) ** 2 > 12 ** 2 for kx, ky, _ in kept):
                kept.append((float(x), float(y), float(c)))
            if len(kept) >= TOP_K:
                break
        return kept


# ── Scan + candidate search ───────────────────────────────────────────────────
def scan(video: Path, detector: BallDetector, crop: tuple[int, int, int, int],
         progress: Optional[Callable[[int, int], None]] = None) -> tuple[list, dict]:
    """Run the detector over sampled frames. Returns (samples, video_info)
    where samples = [(sec, [(x, y, conf), ...]), ...]."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, int(round(fps / SAMPLE_HZ)))
    samples = []
    f = 0
    while True:
        if f % stride == 0:
            ok, frame = cap.read()
            if not ok:
                break
            samples.append((f / fps, detector.detect(frame, crop)))
            if progress is not None and len(samples) % 20 == 0:
                progress(f, total)
        elif not cap.grab():                     # skip without decoding to an image
            break
        f += 1
    cap.release()
    info = {"fps": float(fps), "frames": f, "width": width, "height": height,
            "duration_sec": f / fps if fps else 0.0, "stride": stride,
            "samples": len(samples)}
    return samples, info


def find_candidates(samples: list, geom: dict) -> list[Candidate]:
    """All plausible releases in the scanned window, best first."""
    ppm = geom["ppm"]
    sx, sy = geom["stump"]
    out: list[Candidate] = []

    for i, (t0, dets0) in enumerate(samples):
        for (x0, y0, c0) in dets0:
            height_m = (sy - y0) / ppm
            if not (RELEASE_HEIGHT_M[0] <= height_m <= RELEASE_HEIGHT_M[1]):
                continue
            if abs(x0 - sx) > RELEASE_SIDE_M * ppm:
                continue

            # Follow the ball forward in time, always taking the nearest
            # detection that a real ball could have reached.
            chain = [(t0, x0, y0, c0)]
            for (t, dets) in samples[i + 1:]:
                if t - t0 > CHAIN_SEC:
                    break
                lt, lx, ly, _ = chain[-1]
                reach = (LINK_SPEED_M_S * (t - lt) + LINK_SLACK_M) * ppm
                best = None
                for (x, y, c) in dets:
                    d = math.hypot(x - lx, y - ly)
                    if d <= reach and (best is None or d < best[0]):
                        best = (d, x, y, c)
                if best is not None:
                    chain.append((t, best[1], best[2], best[3]))

            if len(chain) < MIN_CHAIN_POINTS:
                continue
            # The ball must come DOWN the image after release (dropping to
            # the pitch), and that descent must come before any rise.
            ys = [p[2] for p in chain]
            lowest = int(np.argmax(ys))
            descent_m = (ys[lowest] - y0) / ppm
            if lowest == 0 or descent_m < MIN_DESCENT_M:
                continue
            # ...and it must actually be moving, not a static blob.
            path = sum(math.hypot(chain[k + 1][1] - chain[k][1], chain[k + 1][2] - chain[k][2])
                       for k in range(len(chain) - 1))
            if path < 0.5 * ppm:
                continue

            # Down-then-up is the bounce: the strongest sign this is a bowled
            # ball. Not required (a yorker or full toss never comes back up
            # before the bat), but it counts for a lot.
            rise_m = (ys[lowest] - min(ys[lowest:])) / ppm
            mean_conf = float(np.mean([p[3] for p in chain]))
            score = mean_conf * min(len(chain), 8) / 8.0 * (1.0 + min(descent_m, 1.0))
            if rise_m >= MIN_BOUNCE_RISE_M:
                score *= 1.5
            out.append(Candidate(
                release_sec=t0, end_sec=chain[-1][0], score=round(score, 4),
                n_points=len(chain), mean_conf=round(mean_conf, 4),
                descent_m=round(descent_m, 3),
                points=[(round(p[0], 3), int(p[1]), int(p[2]), round(p[3], 3)) for p in chain],
            ))

    # One delivery produces a candidate from each of its first few samples,
    # and the ball coming off the bat can look like a second "release".
    # Anything that starts while an earlier candidate is still being followed
    # is the same ball: fold it into the earlier one (earliest start wins,
    # best score is kept).
    out.sort(key=lambda c: c.release_sec)
    merged: list[Candidate] = []
    for c in out:
        if merged and c.release_sec <= merged[-1].end_sec + 0.5:
            m = merged[-1]
            m.end_sec = max(m.end_sec, c.end_sec)
            if c.score > m.score:
                m.score, m.n_points, m.mean_conf = c.score, c.n_points, c.mean_conf
        else:
            merged.append(c)
    merged.sort(key=lambda c: -c.score)
    return merged


def find_delivery(video: Path, weights_path: Path, calibration: dict,
                  progress: Optional[Callable[[int, int], None]] = None
                  ) -> tuple[list[Candidate], dict]:
    """Scan `video` and return (candidates best-first, video_info)."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    geom = _geometry(calibration, width, height)
    detector = BallDetector(weights_path)
    samples, info = scan(video, detector, geom["crop"], progress)
    info["crop"] = geom["crop"]
    info["cropped_inference"] = detector.fixed_size is None
    info["frames_with_ball"] = sum(1 for _, d in samples if d)
    return find_candidates(samples, geom), info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--calibration", required=True, type=Path,
                   help="JSON with bat_L, bat_R, bowl_L, bowl_R (and optionally the stumps)")
    p.add_argument("--out", type=Path, help="write the best delivery clip here")
    args = p.parse_args(argv)

    calib = json.loads(args.calibration.read_text(encoding="utf-8"))
    calib = calib.get("calibration", calib)
    cands, info = find_delivery(args.video, args.weights, calib)
    print(json.dumps({"video": info, "candidates": [
        {k: v for k, v in asdict(c).items() if k != "points"} for c in cands]}, indent=2))
    if not cands:
        print("no delivery found", file=sys.stderr)
        return 1
    if args.out:
        best = cands[0]
        cut_clip(args.video, args.out, best.clip_start, best.clip_end(info["duration_sec"]))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
