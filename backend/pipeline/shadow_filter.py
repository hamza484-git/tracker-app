"""Ball-shadow rejection — shared by the 2D legacy and 3D reconstruct3d pipelines.

A real ball flying through the air casts a shadow onto the pitch. The YOLO
detector sometimes fires on BOTH the ball and its shadow in the same frame,
and if the shadow's score beats the ball's, the shadow blob poisons the
trajectory fit.

Signature of a shadow false-positive, in image coordinates:
  - same u as the ball (± small horizontal tolerance from an angled light)
  - strictly larger v (lower in the image, on the pitch surface)
  - the two detections are in the same frame

Rejection rule: in frames with multiple ball detections, drop any whose u is
close to another's and whose v is significantly larger. Return the rejections
with debug metadata (shadow_of_frame, Δu, Δv) so a caller can log them.

The core function `reject_shadows_uv` works on a format-agnostic list of
dicts carrying ``{"frame": int, "u": float, "v": float, ...}``. Two wrappers
adapt the common detection formats:

  reject_ball_shadows(dets)
      3D pipeline format: dicts already carry u, v, frame.

  reject_ball_shadows_legacy(dets_by_frame)
      Legacy process.py format: dicts carry x1, y1, x2, y2, class_name, conf.
      Only ball detections are filtered; non-ball detections (bats etc.) pass
      through untouched.

Both wrappers return (kept, rejected). ``rejected`` entries have the extra
``shadow_of_frame``, ``shadow_du_px``, ``shadow_dv_px`` fields for logging.
"""
from __future__ import annotations

from typing import Any


# Default tuning. Values chosen so a shadow directly below the ball (overhead
# sun) is caught, but a horizontally-separated second detection (two balls,
# false positive on the batsman) is kept — the shrink loop / bat-impact
# cutoff downstream can handle the latter case properly.
DEFAULT_V_SEPARATION_PX = 20.0
DEFAULT_U_TOLERANCE_PX  = 40.0


def reject_shadows_uv(
    dets: list[dict[str, Any]],
    v_separation_px: float = DEFAULT_V_SEPARATION_PX,
    u_tolerance_px: float = DEFAULT_U_TOLERANCE_PX,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Shadow-reject a list of detections carrying frame / u / v.

    Returns (kept, rejected). Non-shadow duplicates (horizontally separated,
    same v, etc.) are kept untouched.
    """
    by_frame: dict[int, list] = {}
    for d in dets:
        by_frame.setdefault(d["frame"], []).append(d)

    kept: list[dict] = []
    rejected: list[dict] = []
    for frame, group in by_frame.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        # Sort by v ascending — topmost detection in image is the ball candidate.
        ordered = sorted(group, key=lambda d: d["v"])
        ball = ordered[0]
        kept.append(ball)
        for d in ordered[1:]:
            du = abs(d["u"] - ball["u"])
            dv = d["v"] - ball["v"]
            if dv > v_separation_px and du < u_tolerance_px:
                rejected.append({
                    **d,
                    "shadow_of_frame": frame,
                    "shadow_du_px": round(du, 2),
                    "shadow_dv_px": round(dv, 2),
                })
            else:
                kept.append(d)
    kept.sort(key=lambda d: (d["frame"], d["v"]))
    return kept, rejected


def reject_ball_shadows(dets, v_separation_px=DEFAULT_V_SEPARATION_PX,
                       u_tolerance_px=DEFAULT_U_TOLERANCE_PX):
    """3D-pipeline wrapper: detections already carry frame / u / v."""
    return reject_shadows_uv(dets, v_separation_px, u_tolerance_px)


def reject_ball_shadows_legacy(
    dets_this_frame: list[dict[str, Any]],
    frame_index: int,
    v_separation_px: float = DEFAULT_V_SEPARATION_PX,
    u_tolerance_px: float = DEFAULT_U_TOLERANCE_PX,
) -> tuple[list[dict[str, Any]], int]:
    """Legacy-pipeline wrapper for ONE frame's detection list.

    Input: a list of detections with x1, y1, x2, y2, class_name, conf.
    Only entries with class_name == 'ball' are candidates for shadow
    rejection. Non-ball entries pass through unchanged. The ball detections
    are converted to u/v (bbox center), run through reject_shadows_uv, and
    then returned with their original schema preserved.

    Returns (kept_detections_this_frame, n_shadow_rejections).
    """
    balls = []
    non_balls = []
    for d in dets_this_frame:
        if d.get("class_name") == "ball":
            u = 0.5 * (d["x1"] + d["x2"])
            v = 0.5 * (d["y1"] + d["y2"])
            # Carry a reference back to the original so we can return the
            # legacy-format dict unchanged after filtering.
            balls.append({"frame": frame_index, "u": u, "v": v, "_orig": d})
        else:
            non_balls.append(d)

    if len(balls) <= 1:
        return dets_this_frame, 0

    kept_uv, rejected_uv = reject_shadows_uv(balls, v_separation_px, u_tolerance_px)

    kept_balls = [d["_orig"] for d in kept_uv]
    return non_balls + kept_balls, len(rejected_uv)
