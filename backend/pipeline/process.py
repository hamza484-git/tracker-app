"""Headless `process_video()` entry point.

Wraps the legacy Zone-div 4-pass renderer (main_V2.1.py) into a
server-safe callable with:
  - no cv2.imshow / waitKey (calibration is passed in as data)
  - typed PipelineError subclasses on failure (spec §10)
  - progress + cancel callbacks
  - ffmpeg re-encode to H.264 + faststart (spec §9)
  - results.json built per spec §4

The legacy detector/filters modules are used as-is, including their module
globals — each job runs in its own subprocess (see worker.py / jobs.py), so
state cannot leak between jobs. Which legacy folder is bound (white-ball
Zone-div(final) or red-ball Zone-div RED) is decided by TRACKER_LEGACY_DIR
before this module is imported; both expose the same module API, the red one
additionally has ReleaseDetector.refine_from_track, called when present.

Resilience: when an event detector fails to lock (release / bounce / deviation),
that segment of the multi-pass render is skipped. The job still succeeds —
the JSON simply carries `null` for that event. Only zero ball detections is a
hard failure (NoBallDetectedError).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

# Legacy modules (resolved via TRACKER_LEGACY_DIR — white or red pipeline)
from config import GRAVITY_PX_PER_FRAME2, BOUNCE_MARKER_SIZE  # noqa: E402
import detector as _detector_mod  # noqa: E402 — for the IMGSZ fallback override
from detector import detect_objects  # noqa: E402
from filters import reset_filters  # noqa: E402
from grid_utils import compute_grid_corners, pixel_to_grid_cell  # noqa: E402
from impact_detector import ImpactDetector  # noqa: E402  (stub; kept for shape)
from logger import DetectionLogger  # noqa: E402
from release_detector import ReleaseDetector  # noqa: E402
from tracker import BounceDetector  # noqa: E402
from trajectory import Trajectory  # noqa: E402
from utils import overlay_ball_image  # noqa: E402
from visualizer import draw_detections, draw_release_to_bounce_parabola  # noqa: E402
from zones import build_homography, draw_zones  # noqa: E402

from .errors import (
    EncodingFailedError,
    HomographyFailedError,
    InvalidCalibrationError,
    InvalidVideoError,
    ModelLoadFailedError,
    NoBallDetectedError,
    PipelineCancelled,
)
from .zones_classify import classify_bounce


def get_session(weights_path: Path):
    """ONNX session from an explicit weights path (legacy load_model reads
    config.WEIGHTS_PATH, which the server overrides per ball type)."""
    import onnxruntime as ort

    if not Path(weights_path).exists():
        raise ModelLoadFailedError(f"Model not found: {weights_path}")
    try:
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        return ort.InferenceSession(str(weights_path), providers=providers)
    except Exception as e:  # ORT raises a variety of native errors
        raise ModelLoadFailedError(str(e)) from e


def _refine_release(release_detector, bounce_detector, trajectory) -> None:
    """RED pipeline only: once the bounce is known, re-derive the release from
    the bounce-anchored track (the streaming streak can lock onto the wind-up).
    No-op on the white pipeline, which has no refine_from_track."""
    if (
        bounce_detector.bounce_point is not None
        and hasattr(release_detector, "refine_from_track")
    ):
        release_detector.refine_from_track(
            trajectory._pts, bounce_detector.bounce_point
        )


# ─────────────────────────────────────────────────────────────────────────────
# Calibration validation
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED_MARKERS = ("bat_L", "bat_R", "bowl_L", "bowl_R")

# Stump base sits 1.22 m behind the popping crease; popping creases sit
# 17.68 − 2*1.22 = 15.24 m apart. The legacy homography needs synthetic
# bat_stump / bowl_stump (middle-stump base, behind/beyond the crease) — we
# compute them from the 4 user-placed crease corners.
_STUMP_OFFSET_FRAC = 1.22 / 15.24


def _validate_calibration(calib: dict, width: int, height: int) -> dict:
    if not isinstance(calib, dict):
        raise InvalidCalibrationError("calibration must be an object")
    missing = [k for k in REQUIRED_MARKERS if k not in calib]
    if missing:
        raise InvalidCalibrationError(f"missing markers: {missing}")
    out = {}
    for k in REQUIRED_MARKERS:
        v = calib[k]
        if not (hasattr(v, "__len__") and len(v) == 2):
            raise InvalidCalibrationError(f"{k} must be [x, y]")
        try:
            x, y = int(v[0]), int(v[1])
        except (TypeError, ValueError):
            raise InvalidCalibrationError(f"{k} coordinates must be integers")
        if not (0 <= x < width and 0 <= y < height):
            raise InvalidCalibrationError(
                f"{k}=({x},{y}) out of frame bounds {width}x{height}"
            )
        out[k] = (x, y)
    if out["bat_L"][0] >= out["bat_R"][0]:
        raise InvalidCalibrationError("bat_L must be left of bat_R")
    if out["bowl_L"][0] >= out["bowl_R"][0]:
        raise InvalidCalibrationError("bowl_L must be left of bowl_R")
    bat_mean_y = (out["bat_L"][1] + out["bat_R"][1]) / 2
    bowl_mean_y = (out["bowl_L"][1] + out["bowl_R"][1]) / 2
    if abs(bat_mean_y - bowl_mean_y) < 50:
        raise InvalidCalibrationError(
            "batter-end and bowler-end markers are not vertically separated"
        )
    # Middle-stump bases: prefer user-placed points; fall back to synthesis
    # from the 4 crease corners.
    for stump_key in ("bat_stump", "bowl_stump"):
        v = calib.get(stump_key)
        if v is None or not (hasattr(v, "__len__") and len(v) == 2):
            continue
        try:
            sx, sy = int(v[0]), int(v[1])
        except (TypeError, ValueError):
            raise InvalidCalibrationError(f"{stump_key} coordinates must be integers")
        if not (0 <= sx < width and 0 <= sy < height):
            raise InvalidCalibrationError(
                f"{stump_key}=({sx},{sy}) out of frame bounds {width}x{height}"
            )
        out[stump_key] = (sx, sy)

    if "bat_stump" not in out or "bowl_stump" not in out:
        bat_mid_x = (out["bat_L"][0] + out["bat_R"][0]) / 2
        bat_mid_y = (out["bat_L"][1] + out["bat_R"][1]) / 2
        bowl_mid_x = (out["bowl_L"][0] + out["bowl_R"][0]) / 2
        bowl_mid_y = (out["bowl_L"][1] + out["bowl_R"][1]) / 2
        dx = bat_mid_x - bowl_mid_x
        dy = bat_mid_y - bowl_mid_y
        out.setdefault("bat_stump", (
            int(round(bat_mid_x + dx * _STUMP_OFFSET_FRAC)),
            int(round(bat_mid_y + dy * _STUMP_OFFSET_FRAC)),
        ))
        out.setdefault("bowl_stump", (
            int(round(bowl_mid_x - dx * _STUMP_OFFSET_FRAC)),
            int(round(bowl_mid_y - dy * _STUMP_OFFSET_FRAC)),
        ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Overlay helpers — direct ports of main_V2.1._draw_progressive_overlays etc.
# ─────────────────────────────────────────────────────────────────────────────
def _draw_progressive_overlays(
    frame,
    frame_count: int,
    detections: list,
    release_detector: ReleaseDetector,
    bounce_detector: BounceDetector,
    trajectory: Trajectory,
    *,
    draw_parabolas: bool = False,  # kept for call-site compatibility (no-op)
    rel_to_bounce_progress: float = 1.0,
    bounce_to_dev_progress: float = 1.0,
    rel_to_bounce_pts=None,
    rel_to_bounce_frame_range=None,
):
    """Bake the growing post-bounce trajectory + direction markers.

    Ball detection boxes and per-frame confidence labels are intentionally
    NOT drawn — the frontend SVG overlay covers the interpretive layer.
    Parabolas, release/bounce/deviation pins, stumps and the pitching line
    are also handled by the SVG overlay so users can toggle them without
    re-rendering. `draw_parabolas` is kept on the signature so existing
    call-sites compile, but is intentionally ignored.

    When the ball has bounced but no deviation was detected, we still trace
    the ball's real post-bounce path (as long as detections keep landing)
    with a growing polyline, so the viewer can see the ball's path forward.
    """
    del draw_parabolas, rel_to_bounce_progress, bounce_to_dev_progress
    del rel_to_bounce_pts, rel_to_bounce_frame_range
    del detections
    bf_now = (
        bounce_detector.bounce_point[0]
        if bounce_detector.bounce_point else None
    )
    # Post-bounce polyline is ONLY a fallback for when no deviation point was
    # detected. When we have a deviation, the release→bounce→deviation parabola
    # from the SVG overlay already tells the full story — drawing a raw
    # polyline on top of it in SEG A adds noise.
    have_dev_now = bool(getattr(trajectory, "_dir_changes", None))
    if (
        bf_now is not None
        and frame_count >= bf_now
        and not have_dev_now
        and hasattr(trajectory, "draw_post_bounce_path")
    ):
        frame = trajectory.draw_post_bounce_path(
            frame,
            bounce_frame=bf_now,
            up_to_frame=frame_count,
        )

    # Bake release + bounce markers into the annotated video (per-ball-type
    # coloured PNG via utils.overlay_ball_image → TRACKER_BALL_TYPE env var).
    # Deviation is drawn below by trajectory.draw_direction_change_markers.
    from utils import overlay_ball_image  # local import — legacy TRACKER_LEGACY_DIR
    release = getattr(release_detector, "release", None)
    if release is not None and frame_count >= release.release_frame:
        overlay_ball_image(frame, int(release.x), int(release.y),
                           BOUNCE_MARKER_SIZE)
    if bounce_detector.bounce_point is not None:
        bf, bx, by = bounce_detector.bounce_point
        if frame_count >= bf:
            overlay_ball_image(frame, int(bx), int(by), BOUNCE_MARKER_SIZE)

    frame = trajectory.draw_direction_change_markers(frame, frame_count)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Stumps + wicket-to-wicket line overlay (SEG B and SEG C only — never SEG A).
# Sizing is driven by the calibrated crease width (real-world 1.32 m), so each
# end's stumps come out at the correct pixel scale automatically.
# ─────────────────────────────────────────────────────────────────────────────
_STUMP_DIAM_M      = 0.03493                       # 34.93 mm diameter
_STUMP_GAP_M       = 0.054                         # 54 mm edge-to-edge gap
_STUMP_PITCH_M     = _STUMP_DIAM_M + _STUMP_GAP_M  # center-to-center: 88.93 mm
_STUMP_SET_WIDTH_M = 3 * _STUMP_DIAM_M + 2 * _STUMP_GAP_M  # 212.79 mm outer-to-outer
_STUMP_HEIGHT_M    = 0.71  # 71.12 cm — true height; was over-trimmed before
                           # and read short at the batter (near) end.
# The user-calibrated L/R crease corners sit on the return crease lines,
# which are 1.32 m on each side of middle stump → 2.64 m apart end-to-end.
_RETURN_CREASE_M   = 2.64

# Off-white stumps (BGR). Highlight is near-white with a hint of warmth;
# shadow side is a soft grey for cylindrical shading; outline is darker grey
# to keep edges crisp against the pitch.
_STUMP_CREAM_LIGHT = (240, 245, 248)
_STUMP_CREAM_DARK  = (190, 200, 210)
_STUMP_OUTLINE     = (110, 120, 130)


def _draw_stumps_at(frame, base_xy, crease_w_pixels,
                    up_vec=(0.0, -1.0), alpha=0.75):
    """Draw a 3-stump set with bails at `base_xy`.

    Scale at this end comes entirely from this end's own popping crease width
    (`crease_w_pixels`, which spans the 2.64 m return-crease-to-return-crease
    distance). The two ends are sized independently — bowler-end small, batter-
    end larger — naturally giving the right perspective without any cross-end
    coupling.
    """
    if crease_w_pixels <= 0:
        return frame
    px_per_m = crease_w_pixels / _RETURN_CREASE_M
    stump_h  = _STUMP_HEIGHT_M * px_per_m
    stump_d  = max(2.0, _STUMP_DIAM_M * px_per_m)
    thick    = max(2, int(round(stump_d)))
    # Outer-stump centers sit at ±_STUMP_PITCH_M from the middle. With diameter
    # _STUMP_DIAM_M added, that lands the outer EDGES at ±_STUMP_SET_WIDTH_M/2,
    # matching the channel quadrilateral exactly.
    half_w = _STUMP_PITCH_M * px_per_m

    ux, uy = up_vec
    px_dir, py_dir = -uy, ux  # perpendicular to up, in the image plane

    cx, cy = float(base_xy[0]), float(base_xy[1])
    overlay = frame.copy()

    tops = []
    for offset in (-half_w, 0.0, half_w):
        bx = cx + px_dir * offset
        by = cy + py_dir * offset
        tx = bx + ux * stump_h
        ty = by + uy * stump_h
        # Outline (slightly thicker, dark) for crisp edge
        cv2.line(overlay, (int(bx), int(by)), (int(tx), int(ty)),
                 _STUMP_OUTLINE, thick + 2, lineType=cv2.LINE_AA)
        # Light fill (cream highlight)
        cv2.line(overlay, (int(bx), int(by)), (int(tx), int(ty)),
                 _STUMP_CREAM_LIGHT, thick, lineType=cv2.LINE_AA)
        # Thin dark stripe offset to one side fakes cylindrical shading
        sh = max(1, thick // 3)
        ox = px_dir * (thick * 0.25)
        oy = py_dir * (thick * 0.25)
        cv2.line(overlay,
                 (int(bx + ox), int(by + oy)),
                 (int(tx + ox), int(ty + oy)),
                 _STUMP_CREAM_DARK, sh, lineType=cv2.LINE_AA)
        tops.append((tx, ty))

    # Bails: short capsules sitting on top, slightly recessed
    bail_thick = max(2, int(thick * 0.75))
    dip_x = ux * stump_d * 0.4
    dip_y = uy * stump_d * 0.4
    for a, b in ((tops[0], tops[1]), (tops[1], tops[2])):
        x1 = a[0] + dip_x; y1 = a[1] + dip_y
        x2 = b[0] + dip_x; y2 = b[1] + dip_y
        cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)),
                 _STUMP_OUTLINE, bail_thick + 1, lineType=cv2.LINE_AA)
        cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)),
                 _STUMP_CREAM_DARK, bail_thick, lineType=cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    return frame


def _draw_stump_line(frame, bowl_stump, bat_stump,
                     bowl_crease_w, bat_crease_w, alpha=0.30):
    """Draw the wicket-to-wicket channel as a filled translucent quadrilateral
    spanning the outside edges of the bowler-end and batter-end stump sets.
    Per-end scale gives the right perspective foreshortening for free."""
    if bowl_crease_w <= 0 or bat_crease_w <= 0:
        return frame
    bsx, bsy = float(bowl_stump[0]), float(bowl_stump[1])
    btx, bty = float(bat_stump[0]),  float(bat_stump[1])
    dx, dy = btx - bsx, bty - bsy
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1.0:
        return frame
    ux, uy = dx / length, dy / length
    px_dir, py_dir = -uy, ux  # perpendicular in image plane

    half_w_bowl = (_STUMP_SET_WIDTH_M / 2) * (bowl_crease_w / _RETURN_CREASE_M)
    half_w_bat  = (_STUMP_SET_WIDTH_M / 2) * (bat_crease_w  / _RETURN_CREASE_M)

    # Quad corners, traversed in order: bowler-left → batter-left → batter-right
    # → bowler-right (so fillPoly draws a non-self-intersecting polygon).
    bowl_L = (bsx - px_dir * half_w_bowl, bsy - py_dir * half_w_bowl)
    bat_L  = (btx - px_dir * half_w_bat,  bty - py_dir * half_w_bat)
    bat_R  = (btx + px_dir * half_w_bat,  bty + py_dir * half_w_bat)
    bowl_R = (bsx + px_dir * half_w_bowl, bsy + py_dir * half_w_bowl)
    pts = np.array(
        [[int(p[0]), int(p[1])] for p in (bowl_L, bat_L, bat_R, bowl_R)],
        dtype=np.int32,
    )

    overlay = frame.copy()
    color = (255, 255, 255)  # white
    cv2.fillPoly(overlay, [pts], color, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    return frame


def _draw_stumps_overlay(frame, markers):
    """Stumps at both ends + wicket-to-wicket channel. Called from SEG B and
    SEG C only — never SEG A (which is the basic walkthrough)."""
    try:
        bowl_L = markers["bowl_L"]; bowl_R = markers["bowl_R"]
        bat_L  = markers["bat_L"];  bat_R  = markers["bat_R"]
        bowl_stump = markers["bowl_stump"]
        bat_stump  = markers["bat_stump"]
    except KeyError:
        return frame
    bowl_w = ((bowl_R[0] - bowl_L[0]) ** 2 + (bowl_R[1] - bowl_L[1]) ** 2) ** 0.5
    bat_w  = ((bat_R[0]  - bat_L[0])  ** 2 + (bat_R[1]  - bat_L[1])  ** 2) ** 0.5
    # Channel first so stumps render on top of the lines
    frame = _draw_stump_line(frame, bowl_stump, bat_stump, bowl_w, bat_w)
    frame = _draw_stumps_at(frame, bowl_stump, bowl_w)
    frame = _draw_stumps_at(frame, bat_stump,  bat_w)
    return frame


def _draw_bounce_marker_at(frame, pt):
    if pt is None:
        return frame
    bx, by = pt
    shadow_cy = by + int(BOUNCE_MARKER_SIZE * 0.85) - 8
    base_rx = max(2, int(BOUNCE_MARKER_SIZE * 0.42))
    base_ry = max(2, int(BOUNCE_MARKER_SIZE * 0.168))
    for k, alpha in ((1.6, 0.10), (1.3, 0.18), (1.0, 0.30)):
        rx = max(2, int(base_rx * k))
        ry = max(2, int(base_ry * k))
        overlay = frame.copy()
        cv2.ellipse(
            overlay, (int(bx), shadow_cy), (rx, ry), 0, 0, 360,
            (0, 0, 0), -1, lineType=cv2.LINE_AA,
        )
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    overlay_ball_image(frame, bx, by, BOUNCE_MARKER_SIZE)
    return frame


def _fit_constrained_quadratic_k(release_pt, bounce_pt, trajectory_points, frame_range):
    if not trajectory_points or len(trajectory_points) < 1 or frame_range is None:
        return 0.0, 0.0
    f_lo, f_hi = float(frame_range[0]), float(frame_range[1])
    if f_hi <= f_lo:
        return 0.0, 0.0
    x1, y1 = float(release_pt[0]), float(release_pt[1])
    x2, y2 = float(bounce_pt[0]), float(bounce_pt[1])
    ts_p = np.array(
        [(p[0] - f_lo) / (f_hi - f_lo) for p in trajectory_points], dtype=float
    )
    xs_p = np.array([p[1] for p in trajectory_points], dtype=float)
    ys_p = np.array([p[2] for p in trajectory_points], dtype=float)
    a = ts_p * (1.0 - ts_p)
    denom = float(np.sum(a * a))
    if denom <= 1e-9:
        return 0.0, 0.0
    bx = xs_p - ((1.0 - ts_p) * x1 + ts_p * x2)
    by = ys_p - ((1.0 - ts_p) * y1 + ts_p * y2)
    return float(np.sum(a * bx) / denom), float(np.sum(a * by) / denom)


def _point_on_constrained_quadratic(p1, p2, kx, ky, t):
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    a = t * (1.0 - t)
    return (1.0 - t) * x1 + t * x2 + kx * a, (1.0 - t) * y1 + t * y2 + ky * a


# ─────────────────────────────────────────────────────────────────────────────
# Segment C: frozen-frame zone + parabola animation
# ─────────────────────────────────────────────────────────────────────────────
def _run_segment_c(
    out_writer, video_path, fps,
    deviation_frame, release_pt, bounce_pt, deviation_pt,
    rel_to_bounce_pts, rel_to_bounce_frame_range, markers,
):
    if release_pt is None or bounce_pt is None or deviation_pt is None:
        return
    if deviation_frame is None:
        return
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, deviation_frame))
    ret, frozen = cap.read()
    cap.release()
    if not ret:
        return

    base = frozen.copy()
    H_world = build_homography(markers) if markers else None
    if H_world is not None:
        base = draw_zones(base, H_world, alpha=0.30, label=True)

    # SEG C: zoned frozen frame held for 0.5× of the source flight duration
    # (release→deviation) so the SVG parabola can sweep at half-speed, plus a
    # 1 s hold at the end. The 0.5× factor matches the user spec.
    release_f_for_c = rel_to_bounce_frame_range[0] if rel_to_bounce_frame_range else 0
    rel_to_dev_frames = max(1, deviation_frame - release_f_for_c)
    del release_pt, bounce_pt, deviation_pt, rel_to_bounce_pts
    del rel_to_bounce_frame_range
    anim_sec = 2.0 * rel_to_dev_frames / fps
    hold_sec = 1.0
    n_total = max(8, int(round(fps * (anim_sec + hold_sec))))
    for _ in range(n_total):
        out_writer.write(base.copy())


# ─────────────────────────────────────────────────────────────────────────────
# H.264 re-encode (spec §9)
# ─────────────────────────────────────────────────────────────────────────────
def _find_ffmpeg() -> Optional[str]:
    """Locate an ffmpeg binary: PATH first, then the imageio-ffmpeg bundle."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


def _reencode_h264(src: Path, dst: Path) -> bool:
    """Re-encode src→dst as H.264 + faststart. Returns True on success.

    If ffmpeg is missing or fails, returns False so the caller can fall back
    to shipping the raw OpenCV mp4v file. Playable in VLC and most desktop
    players; some browsers (notably Chrome) may not decode it.
    """
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        return False
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(src),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(dst),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except OSError:
        return False
    return result.returncode == 0


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def process_video(
    video_path: Path,
    output_dir: Path,
    calibration: dict,
    weights_path: Path,
    *,
    interp_method: str = "parabola",
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    job_id: Optional[str] = None,
    video_filename: Optional[str] = None,
) -> dict:
    """Process one delivery video end-to-end. See spec §3.1 and §4."""
    start_time = time.time()
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = Path(weights_path)

    def _check_cancel():
        if cancel_check is not None and cancel_check():
            raise PipelineCancelled()

    # ── Probe video ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise InvalidVideoError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total_frames <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise InvalidVideoError("zero frames or invalid dimensions")

    # First frame for calibration UI artifact
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise InvalidVideoError("could not read first frame")
    first_frame_path = output_dir / "first_frame.jpg"
    cv2.imwrite(str(first_frame_path), first_frame)

    # ── Validate calibration ─────────────────────────────────────────────────
    markers = _validate_calibration(calibration, width, height)
    try:
        grid_corners = compute_grid_corners({k: list(v) for k, v in markers.items()})
    except Exception as e:
        cap.release()
        raise HomographyFailedError(str(e)) from e
    if build_homography(markers) is None:
        cap.release()
        raise HomographyFailedError("degenerate marker layout")

    # ── Load model ───────────────────────────────────────────────────────────
    session = get_session(weights_path)

    # ─── PASS 1: analysis ────────────────────────────────────────────────────
    # Run once at the default IMGSZ (960); if the result lacks release or both
    # bounce+deviation, re-run at IMGSZ=1280 — the bigger letterbox gives the
    # detector ~1.78× more pixels on a small/blurred ball and recovers the
    # mid-flight frames that the 960 pass misses.
    impact_detector = ImpactDetector()  # stub, kept for API surface
    bounce_detector = BounceDetector()
    release_detector = ReleaseDetector()
    trajectory = Trajectory()
    detections_by_frame: dict[int, list] = {}
    ball_count = 0
    bat_count = 0

    # Stump-top y at bowler end — derived from calibration. ReleaseDetector
    # only sees ball candidates above this line; a real release happens at
    # hand height (always above the bails).
    _bowl_L = markers["bowl_L"]; _bowl_R = markers["bowl_R"]
    _bowl_crease_w = (
        (_bowl_R[0] - _bowl_L[0]) ** 2 + (_bowl_R[1] - _bowl_L[1]) ** 2
    ) ** 0.5
    _bowl_stump_top_y = (
        markers["bowl_stump"][1]
        - _STUMP_HEIGHT_M * (_bowl_crease_w / _RETURN_CREASE_M)
    )

    # Bat anchor at the batter's end — feeds the Trajectory source-level
    # filter as the escape-hatch reference point. A candidate that fails the
    # trajectory-extrapolation check is only accepted if it lands within a
    # small radius of (bat_zone_cx, bat_zone_cy), i.e. plausibly at the bat.
    # half_w / half_h are still passed (as legacy stubs for older Trajectory
    # class versions), but the current source filter only uses cx/cy.
    _bat_L_m = markers.get("bat_L"); _bat_R_m = markers.get("bat_R")
    _bat_stump_m = markers.get("bat_stump")
    _bat_zone_cx = _bat_stump_m[0] if _bat_stump_m is not None else None
    _bat_zone_cy = None
    if _bat_stump_m is not None and _bat_L_m and _bat_R_m:
        _bat_crease_w = (
            (_bat_R_m[0] - _bat_L_m[0]) ** 2 + (_bat_R_m[1] - _bat_L_m[1]) ** 2
        ) ** 0.5
        _bat_stump_h_px = _STUMP_HEIGHT_M * (_bat_crease_w / _RETURN_CREASE_M)
        # Aim for pad-height (~40% of stump height above the base) — that's
        # where a defensive contact usually lands.
        _bat_zone_cy = _bat_stump_m[1] - _bat_stump_h_px * 0.4

    def _apply_bat_zone(traj) -> None:
        if hasattr(traj, "set_bat_zone"):
            traj.set_bat_zone(
                cx=_bat_zone_cx, cy=_bat_zone_cy,
                half_w=None, half_h=None,
            )
    _apply_bat_zone(trajectory)

    def _run_pass1():
        nonlocal bounce_detector, release_detector, trajectory
        nonlocal detections_by_frame, ball_count, bat_count
        reset_filters()
        bounce_detector = BounceDetector()
        release_detector = ReleaseDetector()
        trajectory = Trajectory()
        _apply_bat_zone(trajectory)
        detections_by_frame = {}
        ball_count = 0
        bat_count = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        f = 0
        while True:
            _check_cancel()
            ret, frm = cap.read()
            if not ret:
                break
            dets = detect_objects(session, frm, f)
            detections_by_frame[f] = dets
            if any(d["class_name"] == "ball" for d in dets):
                ball_count += 1
            bat_count += sum(1 for d in dets if d["class_name"] == "bat")
            # Only feed release_detector ball candidates above stump top.
            rd_dets = [
                d for d in dets
                if d.get("class_name") != "ball"
                or 0.5 * (d["y1"] + d["y2"]) <= _bowl_stump_top_y
            ]
            release_detector.update(rd_dets, f)
            rel_f_now = (
                release_detector.release.release_frame
                if release_detector.release else None
            )
            bounce_detector.update(dets, f, release_frame=rel_f_now)
            bf_now = (
                bounce_detector.bounce_point[0]
                if bounce_detector.bounce_point else None
            )
            trajectory.update(dets, f, release_frame=rel_f_now, bounce_frame=bf_now)
            _refine_release(release_detector, bounce_detector, trajectory)
            if progress_callback is not None and f % 10 == 0:
                progress_callback(f, total_frames)
            f += 1
        return f

    _detector_mod.IMGSZ = 1280
    total = _run_pass1()
    # IMGSZ=1600 fallback disabled: current ONNX models are exported with a
    # fixed 1280×1280 input shape (yolo export imgsz=1280, no dynamic=True),
    # so re-running at 1600 fails with an ONNXRuntime shape mismatch. Re-export
    # the models with dynamic=True to re-enable the fallback.

    if ball_count == 0:
        cap.release()
        raise NoBallDetectedError("zero ball detections in the whole video")

    release = release_detector.release
    bounce = bounce_detector.bounce_point
    # Last-chance deviation detector (last-seen fallback) — runs after all
    # frames are processed. Cannot run per-frame because during streaming it
    # would fire on the first post-bounce frame (2 points post-bounce >= the
    # min-2 threshold) even though more frames are still incoming. The
    # bat_line_x / bat_line_y hints (midpoint of the batter's return-crease
    # markers + the batter-end middle stump base) tell the extrapolator where
    # the impact would have physically been, so the projection stops there
    # instead of running for a fixed number of frames.
    if hasattr(trajectory, "finalize_deviation") and bounce is not None:
        _bat_L = markers.get("bat_L"); _bat_R = markers.get("bat_R")
        _bat_stump = markers.get("bat_stump")
        _bat_x = (_bat_L[0] + _bat_R[0]) / 2 if _bat_L and _bat_R else None
        # Bat-impact height sits roughly at bail height above the crease base.
        # We use the batter-end stump-top y as a proxy: crease base minus the
        # stump-height in pixels (computed from calibration).
        _bat_y = None
        if _bat_stump is not None and _bat_L and _bat_R:
            _crease_w = (
                (_bat_R[0] - _bat_L[0]) ** 2 + (_bat_R[1] - _bat_L[1]) ** 2
            ) ** 0.5
            _stump_h_px = _STUMP_HEIGHT_M * (_crease_w / _RETURN_CREASE_M)
            # Bat contact zone: pad height (~40% of stump height) above the base.
            _bat_y = _bat_stump[1] - _stump_h_px * 0.4
        trajectory.finalize_deviation(
            bounce[0], bat_line_x=_bat_x, bat_line_y=_bat_y,
        )
        # If a detector fired but landed far from the bat — typically the
        # parabolic residual detector taking several frames of accumulated
        # divergence to trip AFTER a genuine hit — pull the deviation back
        # to the trajectory point closest to the batter's stump position.
        if hasattr(trajectory, "refine_deviation_to_bat"):
            trajectory.refine_deviation_to_bat(bounce[0])
    dev_changes = list(trajectory._dir_changes)
    have_release = release is not None
    have_bounce = bounce is not None
    have_dev = bool(dev_changes)

    release_f = release.release_frame if have_release else None
    bounce_f = bounce[0] if have_bounce else None
    dev_f = dev_changes[0][0] if have_dev else None

    # Prefer the earliest cluster of valid ball detections as release.
    # The streaming ReleaseDetector picks the first 3-frame streak whose
    # speed ratio stays within 0.5×–2× — that gate is conservative and on
    # jittery flight detections it can skip the real delivery cluster and
    # lock on the cleaner post-impact ball instead. After PASS 1 we have
    # every detection; scan for the EARLIEST cluster (≥3 ball detections
    # within an 8-frame window, all above the bowler-end stump top, with
    # ≥40 px total displacement so a static blob doesn't qualify) and
    # override the lock to that cluster's first frame.
    if have_release:
        _stump_top = (
            markers["bowl_stump"][1]
            - _STUMP_HEIGHT_M
            * (((markers["bowl_R"][0] - markers["bowl_L"][0]) ** 2
                + (markers["bowl_R"][1] - markers["bowl_L"][1]) ** 2) ** 0.5)
            / _RETURN_CREASE_M
        )
        _above = []  # (frame, x, y) of one ball detection per frame above stump
        for fnum in sorted(detections_by_frame):
            if fnum >= release_f:
                break
            for d in detections_by_frame[fnum]:
                if d.get("class_name") != "ball":
                    continue
                cx = 0.5 * (d["x1"] + d["x2"])
                cy = 0.5 * (d["y1"] + d["y2"])
                if cy <= _stump_top:
                    _above.append((fnum, cx, cy))
                    break
        for i in range(len(_above) - 2):
            f0, x0, y0 = _above[i]
            cluster = [(f0, x0, y0)]
            for fj, xj, yj in _above[i + 1:]:
                if fj - f0 > 8:
                    break
                cluster.append((fj, xj, yj))
            if len(cluster) < 3:
                continue
            xs = [p[1] for p in cluster]
            ys = [p[2] for p in cluster]
            span = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
            if span < 40:
                continue
            release.release_frame = int(f0)
            release.x = int(x0)
            release.y = int(y0)
            release_f = int(f0)
            print(
                f"[RELEASE OVERRIDE] earliest above-stump cluster: "
                f"F{f0} ({int(x0)},{int(y0)}) — {len(cluster)} pts, "
                f"span {span:.0f}px"
            )
            break

    # Strip pre-release ball detections that can't belong to the delivery.
    # The static-spot suppressor in filters.py needs >4 consecutive frames in
    # the same 12px cell to kill a spot, so true statics still leak 4 frames
    # of FPs into early seg_a. Once release is locked we know the ball was at
    # (release.x, release.y) at release_f, so any ball detection more than
    # _PRE_RELEASE_MAX_DIST px from that point on a frame before release is
    # definitely not the delivery — drop it from detections_by_frame so it
    # neither renders in PASS 2 nor pollutes the trajectory output.
    _PRE_RELEASE_MAX_DIST_SQ = 150 * 150
    if have_release:
        rx, ry = release.x, release.y
        for fnum in range(release_f):
            dets = detections_by_frame.get(fnum)
            if not dets:
                continue
            kept = []
            for d in dets:
                if d.get("class_name") == "ball":
                    cx = 0.5 * (d["x1"] + d["x2"])
                    cy = 0.5 * (d["y1"] + d["y2"])
                    if (cx - rx) ** 2 + (cy - ry) ** 2 > _PRE_RELEASE_MAX_DIST_SQ:
                        continue
                kept.append(d)
            detections_by_frame[fnum] = kept

    # ─── PASS 2: render annotated video (multi-segment, gated on events) ─────
    temp_video = output_dir / "_temp_out.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(temp_video), fourcc, fps, (width, height))
    csv_logger = DetectionLogger(output_dir=output_dir, video_fps=fps)

    # Reset detectors for progressive marker reveal during render
    reset_filters()
    bounce_detector = BounceDetector()
    release_detector = ReleaseDetector()
    trajectory = Trajectory()
    _apply_bat_zone(trajectory)

    seg_a_end = (
        min(dev_f + 30, total - 1) if have_dev
        else (min(bounce_f + 30, total - 1) if have_bounce
              else total - 1)
    )
    # SEG A is live playback from RELEASE-30 → deviation+30 (per user spec —
    # 30-frame lead-in so the bowler's run-up shows on screen before release).
    # Frames before seg_a_start_frame still pass through PASS 2 logic so the
    # CSV log + detector state stay consistent, but they're not written out.
    seg_a_start_frame = max(0, release_f - 30) if have_release else 0
    seg_a_frames_written = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for fnum in range(0, seg_a_end + 1):
        _check_cancel()
        ret, frame = cap.read()
        if not ret:
            break
        dets = detections_by_frame.get(fnum, [])
        release_detector.update(dets, fnum)
        rel_f = (
            release_detector.release.release_frame
            if release_detector.release else None
        )
        bounce_detector.update(dets, fnum, release_frame=rel_f)
        bf_now = (
            bounce_detector.bounce_point[0]
            if bounce_detector.bounce_point else None
        )
        trajectory.update(dets, fnum, release_frame=rel_f, bounce_frame=bf_now)
        _refine_release(release_detector, bounce_detector, trajectory)
        csv_logger.log(fnum, dets, grid_corners=grid_corners)
        if fnum < seg_a_start_frame:
            continue  # CSV logged, but skip writing this frame to SEG A.
        frame = _draw_progressive_overlays(
            frame, fnum, dets,
            release_detector, bounce_detector, trajectory,
            draw_parabolas=False,
        )
        out_writer.write(frame)
        seg_a_frames_written += 1

    # ─── PASS 3: SEG B (only when release + bounce both confirmed) ───────────
    seg_b_frames_written = 0
    if have_release and have_bounce:
        seg_b_start = max(0, release_f - 5)
        seg_b_end = min(dev_f + 5, total - 1) if have_dev else seg_a_end
        full_traj_pts = [p for p in trajectory._pts if p[0] <= seg_b_end]
        rel_to_bounce_pts = [
            (ff, x, y) for (ff, x, y) in full_traj_pts
            if release_f <= ff <= bounce_f
        ]

        # Pre-seed fresh detectors for the replay
        bounce_detector = BounceDetector()
        release_detector = ReleaseDetector()
        trajectory = Trajectory()
        _apply_bat_zone(trajectory)
        release_detector.release = release
        bounce_detector.bounce_point = bounce
        trajectory._dir_changes = list(dev_changes)
        trajectory._pts = list(full_traj_pts)
        big_fnum = seg_b_end + 10_000
        rb_span = max(1, bounce_f - release_f)
        bd_span = max(1, (dev_f - bounce_f) if have_dev else 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, seg_b_start)
        for fnum in range(seg_b_start, seg_b_end + 1):
            _check_cancel()
            ret, frame = cap.read()
            if not ret:
                break
            dets = detections_by_frame.get(fnum, [])
            for sub in (0, 1):
                f_eff = fnum + 0.5 * sub
                rb_prog = max(0.0, min(1.0, (f_eff - release_f) / rb_span))
                bd_prog = (
                    max(0.0, min(1.0, (f_eff - bounce_f) / bd_span))
                    if have_dev else 0.0
                )
                frame_out = _draw_progressive_overlays(
                    frame.copy(), big_fnum, dets,
                    release_detector, bounce_detector, trajectory,
                    draw_parabolas=True,
                    rel_to_bounce_progress=rb_prog,
                    bounce_to_dev_progress=bd_prog,
                    rel_to_bounce_pts=rel_to_bounce_pts,
                    rel_to_bounce_frame_range=(release_f, bounce_f),
                )
                out_writer.write(frame_out)
                seg_b_frames_written += 1

    cap.release()

    # ─── PASS 4: SEG C (frozen frame zones + parabolas) ──────────────────────
    if have_release and have_bounce and have_dev:
        release_pt = (release.x, release.y)
        bounce_pt = (bounce[1], bounce[2])
        deviation_pt = (dev_changes[0][1], dev_changes[0][2])
        _run_segment_c(
            out_writer, video_path, fps,
            deviation_frame=dev_f,
            release_pt=release_pt,
            bounce_pt=bounce_pt,
            deviation_pt=deviation_pt,
            rel_to_bounce_pts=[
                (ff, x, y) for (ff, x, y) in trajectory._pts
                if release_f <= ff <= bounce_f
            ],
            rel_to_bounce_frame_range=(release_f, bounce_f),
            markers={k: list(v) for k, v in markers.items()},
        )

    # ── Segment timestamps in the OUTPUT video timeline ─────────────────────
    # Carries per-segment event timestamps (release / bounce / deviation) so
    # the frontend SVG overlay can drive marker reveal in SEG A, parabola
    # sweep + traveling ball in SEG B, and the 0.5×-then-hold animation in
    # SEG C — each off a single `currentTime` value.
    seg_a_end_sec = seg_a_frames_written / fps
    seg_b_end_sec = seg_a_end_sec + seg_b_frames_written / fps

    # SEG A: starts 30 frames before release (or at frame 0 if release_f < 30).
    # Release shows up at output time (release_f - seg_a_start_frame) / fps;
    # bounce/deviation offset further by their source-frame distance.
    seg_a_lead_in_frames = release_f - seg_a_start_frame if have_release else 0
    seg_a_release_sec = (
        round(seg_a_lead_in_frames / fps, 4) if have_release else None
    )
    seg_a_bounce_sec = (
        round((seg_a_lead_in_frames + (bounce_f - release_f)) / fps, 4)
        if have_release and have_bounce else None
    )
    seg_a_deviation_sec = (
        round((seg_a_lead_in_frames + (dev_f - release_f)) / fps, 4)
        if have_release and have_dev else None
    )

    # SEG B: 0.5× replay starting at release_f - 5. Each input frame writes 2
    # output frames at fps, so input-frame offset N maps to N*2/fps output sec.
    if have_release and have_bounce:
        seg_b_release_sec   = round(seg_a_end_sec + 5 * 2 / fps, 4)
        seg_b_bounce_sec    = round(seg_a_end_sec + (bounce_f - release_f + 5) * 2 / fps, 4)
        seg_b_deviation_sec = (
            round(seg_a_end_sec + (dev_f - release_f + 5) * 2 / fps, 4)
            if have_dev else None
        )
    else:
        seg_b_release_sec = seg_b_bounce_sec = seg_b_deviation_sec = None

    # SEG C: 0.5× anim (rb sweep, then bd sweep), then 1 s hold.
    if have_release and have_bounce and have_dev:
        rb_anim = 2.0 * (bounce_f - release_f) / fps
        bd_anim = 2.0 * (dev_f - bounce_f) / fps
        seg_c_start_sec  = round(seg_b_end_sec, 4)
        seg_c_rb_end_sec = round(seg_b_end_sec + rb_anim, 4)
        seg_c_bd_end_sec = round(seg_b_end_sec + rb_anim + bd_anim, 4)
        seg_c_end_sec    = round(seg_b_end_sec + rb_anim + bd_anim + 1.0, 4)
    else:
        seg_c_start_sec = seg_c_rb_end_sec = seg_c_bd_end_sec = seg_c_end_sec = None

    segments_obj = {
        "seg_a_start_sec":     0.0,
        "seg_a_end_sec":       round(seg_a_end_sec, 4),
        "seg_a_release_sec":   seg_a_release_sec,
        "seg_a_bounce_sec":    seg_a_bounce_sec,
        "seg_a_deviation_sec": seg_a_deviation_sec,
        "seg_b_start_sec":     round(seg_a_end_sec, 4),
        "seg_b_end_sec":       round(seg_b_end_sec, 4),
        "seg_b_release_sec":   seg_b_release_sec,
        "seg_b_bounce_sec":    seg_b_bounce_sec,
        "seg_b_deviation_sec": seg_b_deviation_sec,
        "seg_c_start_sec":     seg_c_start_sec,
        "seg_c_rb_end_sec":    seg_c_rb_end_sec,
        "seg_c_bd_end_sec":    seg_c_bd_end_sec,
        "seg_c_end_sec":       seg_c_end_sec,
    }

    out_writer.release()
    csv_logger.close()

    # ─── ffmpeg H.264 re-encode (spec §9) ────────────────────────────────────
    final_video = output_dir / "output.mp4"
    encoding_warning = None
    if _reencode_h264(temp_video, final_video):
        try:
            temp_video.unlink()
        except OSError:
            pass
    else:
        # ffmpeg missing or failed — ship the raw OpenCV output so the job
        # still completes. Browser playback may not work; install ffmpeg for
        # proper H.264+faststart (see README).
        shutil.move(str(temp_video), str(final_video))
        encoding_warning = (
            "ffmpeg not found — annotated video uses mp4v codec, which "
            "some browsers cannot play. Install ffmpeg and re-run for "
            "H.264 playback."
        )

    # ─── Build results.json (spec §4) ────────────────────────────────────────
    trajectory_points = []
    seen_real = set()
    for (ff, x, y) in trajectory._pts:
        if ff in seen_real:
            continue
        seen_real.add(ff)
        # conf isn't tracked inside Trajectory; recover from detections_by_frame.
        conf = None
        for d in detections_by_frame.get(ff, []):
            if d["class_name"] == "ball":
                conf = round(float(d["conf"]), 4)
                break
        trajectory_points.append({
            "frame": int(ff), "x": int(x), "y": int(y),
            "interpolated": False, "conf": conf,
        })
    trajectory_points.sort(key=lambda p: p["frame"])

    def _ts(frame_idx: int) -> float:
        return round(frame_idx / fps, 4)

    release_obj = None
    if have_release:
        release_obj = {
            "frame": int(release.release_frame),
            "timestamp_sec": _ts(release.release_frame),
            "x": int(release.x), "y": int(release.y),
        }

    bounce_obj = None
    if have_bounce:
        bf, bx, by = bounce
        zone, length_m = classify_bounce(markers, (bx, by))
        cell = pixel_to_grid_cell((bx, by), grid_corners) if grid_corners else None
        bounce_obj = {
            "frame": int(bf), "timestamp_sec": _ts(bf),
            "x": int(bx), "y": int(by),
            "interpolated": False,
            "zone": zone,
            "length_m": length_m,
            "grid_cell": [int(cell[0]), int(cell[1])] if cell else None,
        }

    impact_obj = None  # ImpactDetector is a stub (spec §2.1 / §4)
    if impact_detector.impact is not None:  # future-proof; currently never True
        ip = impact_detector.impact
        impact_obj = {
            "frame": int(ip.frame),
            "timestamp_sec": _ts(ip.frame),
            "x": int(ip.x), "y": int(ip.y),
            "reason": getattr(ip, "reason", ""),
        }

    deviation_obj = None
    if have_dev:
        d_f, d_x, d_y = dev_changes[0]
        deviation_obj = {
            "frame": int(d_f),
            "timestamp_sec": _ts(int(d_f)),
            "x": int(d_x), "y": int(d_y),
        }

    frames_with_ball = ball_count
    frames_interpolated = sum(1 for p in trajectory_points if p["interpolated"])
    detection_rate = (
        frames_with_ball / max(1, frames_with_ball + frames_interpolated)
    )
    processing_time = round(time.time() - start_time, 2)

    # Ball type is set by jobs.py via env var so the frontend overlay can render
    # the right coloured markers + trajectory line without a second API call.
    _ball_type = os.environ.get("TRACKER_BALL_TYPE", "white").strip().lower()
    if _ball_type not in ("white", "red", "pink"):
        _ball_type = "white"

    results = {
        "schema_version": 1,
        "job_id": job_id or output_dir.name,
        "ball_type": _ball_type,
        "video": {
            "filename": video_filename or video_path.name,
            "fps": float(fps),
            "frame_count": int(total),
            "width": int(width),
            "height": int(height),
            "duration_sec": round(total / fps, 4) if fps else 0.0,
        },
        # URLs are filled in by the storage layer / API; pipeline leaves
        # filename-only references for the server to resolve into signed URLs.
        "outputs": {
            "annotated_video_url": "output.mp4",
            "csv_url": csv_logger.csv_path.name,
            "first_frame_url": "first_frame.jpg",
        },
        "calibration": {k: [int(v[0]), int(v[1])] for k, v in markers.items()},
        "events": {
            "release": release_obj,
            "bounce": bounce_obj,
            "impact": impact_obj,
            "deviation": deviation_obj,
        },
        "trajectory": trajectory_points,
        "segments": segments_obj,
        "stats": {
            "frames_with_ball": int(frames_with_ball),
            "frames_interpolated": int(frames_interpolated),
            "detection_rate": round(detection_rate, 4),
            "processing_time_sec": processing_time,
        },
        "warnings": [encoding_warning] if encoding_warning else [],
    }

    with open(output_dir / "results.json", "w") as f_out:
        json.dump(results, f_out, indent=2, sort_keys=True)

    if progress_callback is not None:
        progress_callback(total, total)
    return results
