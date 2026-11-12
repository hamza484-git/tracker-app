"""LBW-style OUT / NOT OUT verdict.

A faithful Python port of the browser's `PredictedPath` in
`frontend/src/features/tracking/PitchOverlay.tsx`. Same geometry, same
constants, same edge cases — so the backend verdict matches exactly what the
Review screen draws.

Method: the ball's post-bounce direction (a least-squares-ish fit through the
last few tracked points, falling back to bounce→deviation) is projected
forward to the batter's stump plane (bail height above `bat_stump`). If the
crossing lands within the stump set's screen half-width — widened by a
tolerance factor to absorb the irreducible 2D-vs-3D error of a single camera —
the ball would have hit: OUT. Otherwise NOT OUT.

Needs both a bounce and a deviation (the bat/pad contact). Without either,
there is no verdict (returns None), same as the browser drawing nothing.

⚠ The constants below MUST stay in sync with PitchOverlay.tsx.
"""
from __future__ import annotations

import math
from typing import Optional

# Real-world stump geometry (metres) — mirrors PitchOverlay.tsx / process.py.
STUMP_DIAM_M = 0.03493
STUMP_GAP_M = 0.054
STUMP_SET_WIDTH_M = 3 * STUMP_DIAM_M + 2 * STUMP_GAP_M   # 0.21279 m, outer-to-outer
STUMP_HEIGHT_M = 0.65
RETURN_CREASE_M = 2.64

# Tuning — MUST match PitchOverlay.tsx.
HIT_ZONE_TOLERANCE_FACTOR = 1.25
PROJECTION_DIRECTION_METHOD = "trajectory_fit"   # or "bounce_to_deviation"
TRAJECTORY_FIT_WINDOW = 5


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compute_verdict(results: dict) -> Optional[dict]:
    """Return the verdict dict, or None when it can't be decided.

    Takes the assembled results (events, trajectory, calibration, video) and
    returns e.g. {"decision": "OUT", "would_hit_stumps": true, ...}.
    """
    ev = results.get("events") or {}
    bounce = ev.get("bounce")
    deviation = ev.get("deviation")
    if not bounce or not deviation:
        return None

    cal = results.get("calibration") or {}
    bat_L, bat_R, bat_stump = cal.get("bat_L"), cal.get("bat_R"), cal.get("bat_stump")
    if not (bat_L and bat_R and bat_stump):
        return None

    # Direction (dx, dy): default bounce → deviation; prefer a fit through the
    # last TRAJECTORY_FIT_WINDOW post-bounce points (robust to a noisy deviation).
    dx = deviation["x"] - bounce["x"]
    dy = deviation["y"] - bounce["y"]
    direction_method = "bounce_to_deviation"
    if PROJECTION_DIRECTION_METHOD == "trajectory_fit":
        trailing = [
            p for p in results.get("trajectory", [])
            if bounce["frame"] <= p["frame"] <= deviation["frame"]
        ][-TRAJECTORY_FIT_WINDOW:]
        if len(trailing) >= 3:
            fdx = trailing[-1]["x"] - trailing[0]["x"]
            fdy = trailing[-1]["y"] - trailing[0]["y"]
            if abs(fdx) >= 1 or abs(fdy) >= 1:
                dx, dy = fdx, fdy
                direction_method = "trajectory_fit"
    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return None

    # Stump-plane screen geometry (identical to StumpSet's scale).
    bat_w = _dist(bat_L, bat_R)
    if bat_w <= 0:
        return None
    px_per_m = bat_w / RETURN_CREASE_M
    stump_h_px = STUMP_HEIGHT_M * px_per_m
    half_stump_w_px = (STUMP_SET_WIDTH_M / 2) * px_per_m
    # Camera behind the bowler → batter's stumps at the TOP of frame (smaller y).
    target_y = bat_stump[1] - stump_h_px

    # Stump-plane crossing — the point that decides hit vs miss.
    hit_x: Optional[float] = None
    hit_y: Optional[float] = None
    if deviation["y"] <= target_y + 1:          # already at/above bail height
        hit_x, hit_y = float(deviation["x"]), float(deviation["y"])
    elif abs(dy) > 1e-3:
        t = (target_y - deviation["y"]) / dy
        if t > 0.02:
            hit_x = deviation["x"] + dx * t
            hit_y = target_y

    would_hit = (
        hit_x is not None
        and abs(hit_x - bat_stump[0]) <= half_stump_w_px * HIT_ZONE_TOLERANCE_FACTOR
    )

    return {
        "decision": "OUT" if would_hit else "NOT OUT",
        "would_hit_stumps": bool(would_hit),
        "projected": hit_x is not None,
        "impact_x": None if hit_x is None else round(float(hit_x), 1),
        "impact_y": None if hit_y is None else round(float(hit_y), 1),
        "stump_center_x": round(float(bat_stump[0]), 1),
        "half_stump_width_px": round(float(half_stump_w_px), 1),
        "tolerance_factor": HIT_ZONE_TOLERANCE_FACTOR,
        "direction_method": direction_method,
    }
