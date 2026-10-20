"""Stage 3+4 — back-project 2D detections to 3D rays and fit a physics
trajectory (projectile + one bounce), writing the Unity playback JSON.

Data flow:
    detections.json + camera_pose.json
        │
        ├─ auto-detect release_frame  (first frame of longest coherent run)
        │
        ├─ window = [release-20, release+150] clipped to available detections
        │
        ├─ each detection (u, v) → ray in world 3D via camera intrinsics/pose
        │
        └─ optimize (p0, v0, t_bounce, restitution) such that
               ball(t)  under gravity, with one bounce reflection off Y=0,
               projects back to the observed (u, v) with minimum error
                          │
                          ▼
                    track_3d.json  (Unity)

Usage:
    python -m backend.pipeline.reconstruct3d.trajectory_fit \
        --detections samples/4_027_06_detections_v8m.json \
        --pose       samples/4_027_06_camera_pose.json \
        --video      "D:/intership/Input files/Red_ball/4_027_06.mp4" \
        --out-track  samples/4_027_06_track_3d.json \
        --out-check  samples/4_027_06_track_check.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from .world_points import (
    PITCH_LENGTH_M, POPPING_IN_FRONT_M, RETURN_HALF_WIDTH_M, BALL_RADIUS_M,
)
from ..shadow_filter import reject_ball_shadows   # re-exported for callers


G = np.array([0.0, -9.81, 0.0])   # gravity, world coords (Y is up)

# ─────────────────── Magnus physics (gated by --use-magnus) ──────────────────
# a_mag = C_MAG * (omega × v), with omega in rad/s and v in m/s.
# C_MAG is the ideal Rayleigh form for a smooth sphere of radius BALL_R, mass
# BALL_M, in air at density AIR_RHO. Real cricket balls have higher effective
# coupling due to the seam; that extra factor is absorbed into the fitted
# |omega| magnitude, which is why spin bounds (±200 rad/s) go above the
# physical ~190 rad/s ceiling of real cricket bowling.
BALL_R  = 0.036
BALL_M  = 0.156
AIR_RHO = 1.225
C_MAG   = (4.0 * np.pi * BALL_R**3 * AIR_RHO) / (3.0 * BALL_M)   # ≈ 1.53e-3

# ─────────────────── Quadratic drag (gated by --use-drag) ────────────────────
# a_drag = -K_DRAG * |v| * v,   K_DRAG = ρ·Cd·A / 2m
# Cd=0.5 is the standard sphere drag coefficient; cricket balls deviate ±15 %
# with seam orientation and surface wear but 0.5 is accurate enough that we
# fix it rather than add a fit parameter (which would trade off against the
# fitted Magnus magnitude and the release speed).
BALL_CD = 0.5
BALL_A  = np.pi * BALL_R**2                                      # ≈ 4.07e-3
K_DRAG  = (AIR_RHO * BALL_CD * BALL_A) / (2.0 * BALL_M)          # ≈ 7.99e-3


# ────────────────────────────── window selection ─────────────────────────────
def find_release_frame(dets, min_run_len=6, gap=3, min_step_px=15.0):
    """Return the first frame of the earliest detection run that carries a
    BOUNCE SIGNATURE — vertical pixel velocity sign change (ball dropping,
    then rising after hitting the pitch). That's the delivery. Post-impact
    deflections are horizontal-only runs and get skipped.

    Falls back to "longest run with fast motion" if no bounce signature is
    detected anywhere."""
    if not dets:
        return None
    ds = sorted(dets, key=lambda d: d["frame"])
    runs = []
    cur = [ds[0]]
    for d in ds[1:]:
        if d["frame"] - cur[-1]["frame"] <= gap:
            cur.append(d)
        else:
            runs.append(cur); cur = [d]
    runs.append(cur)

    fast_runs = []
    for r in runs:
        if len(r) < min_run_len:
            continue
        uv = np.array([[d["u"], d["v"]] for d in r], dtype=np.float64)
        step_lens = np.linalg.norm(np.diff(uv, axis=0), axis=1)
        if step_lens.mean() < min_step_px:
            continue
        fast_runs.append(r)

    if not fast_runs:
        return None

    # Prefer earliest run whose Y-velocity flips sign (ball down → up).
    for r in fast_runs:
        vy = np.diff([d["v"] for d in r])
        # Look for a clear sign change (not just noise): significant down then up.
        signs = np.sign(vy)
        # Ignore near-zero steps
        signs = signs[np.abs(vy) > 3.0]
        if len(signs) < 4:
            continue
        # Any transition from +1 (going down) to -1 (going up)?
        for i in range(len(signs) - 1):
            if signs[i] > 0 and signs[i + 1] < 0:
                return r[0]["frame"]

    # No bounce signature found → longest fast run.
    return max(fast_runs, key=len)[0]["frame"]


def select_window(dets, release_frame, before=20, after=150, max_gap=6):
    """Grab detections in [release-before, release+after] but drop anything
    separated from the release by a gap larger than max_gap frames — that
    excludes ball-in-hand detections before release and spurious late ones.

    Also deduplicates per frame: when the detector emits multiple candidates
    on one frame (e.g. a stationary false positive on the far stumps + the
    real airborne ball), we keep only the highest-confidence one."""
    lo, hi = release_frame - before, release_frame + after
    band_all = [d for d in dets if lo <= d["frame"] <= hi]
    by_frame: dict[int, dict] = {}
    for d in band_all:
        cur = by_frame.get(d["frame"])
        if cur is None or d.get("conf", 0.0) > cur.get("conf", 0.0):
            by_frame[d["frame"]] = d
    band = sorted(by_frame.values(), key=lambda d: d["frame"])
    if not band:
        return []
    # release_frame itself may not be in the detection list; find nearest
    idx0 = min(range(len(band)), key=lambda i: abs(band[i]["frame"] - release_frame))
    # walk forward while gap ≤ max_gap
    out = [band[idx0]]
    i = idx0 + 1
    while i < len(band) and band[i]["frame"] - out[-1]["frame"] <= max_gap:
        out.append(band[i]); i += 1
    # walk backward
    j = idx0 - 1
    while j >= 0 and out[0]["frame"] - band[j]["frame"] <= max_gap:
        out.insert(0, band[j]); j -= 1

    # Bat-impact cutoff: as soon as consecutive pixel-step vectors reverse by
    # more than 120° OR jump more than 3× in length, cut. That's the bat's
    # signature — the ball changes direction/speed discontinuously on contact.
    if len(out) >= 5:
        uvs = np.array([[d["u"], d["v"]] for d in out], dtype=float)
        steps = np.diff(uvs, axis=0)
        lens = np.linalg.norm(steps, axis=1) + 1e-9
        units = steps / lens[:, None]
        cos_120 = np.cos(np.deg2rad(120))
        for k in range(2, len(steps) - 1):
            cos_ang = float(np.dot(units[k], units[k + 1]))
            ratio   = float(lens[k + 1] / lens[k])
            if cos_ang < cos_120 or ratio > 3.0:
                out = out[:k + 2]   # keep points 0..k+1 (inclusive of the pre-impact step)
                break
    return out


# ─────────────────────────── ray back-projection ─────────────────────────────
def pixel_to_ray(uv, K, R, C):
    """Return (origin, direction) of the world-space ray through pixel uv.

    K (3×3) intrinsics, R (3×3) camera rotation (world→camera), C (3,) camera
    center in world coords. R.T maps camera→world.
    """
    u, v = uv
    Kinv = np.linalg.inv(K)
    d_cam = Kinv @ np.array([u, v, 1.0])
    d_world = R.T @ d_cam
    d_world = d_world / np.linalg.norm(d_world)
    return C, d_world


def project_world(P_world, K, R, t):
    """World point → pixel (u, v). t is camera translation (world→camera)."""
    p_cam = R @ P_world + t
    if p_cam[2] <= 0:
        return np.array([np.nan, np.nan])
    p_img = K @ p_cam
    return p_img[:2] / p_img[2]


# ─────────────────────────── trajectory + bounce ─────────────────────────────
def _solve_bounce_time(p0, v0, ball_r=0.036):
    """Analytical bounce time: first t > 0 where the ball's *centre* reaches
    ball_r above the ground. Solves 0.5*g*t^2 + v0y*t + (p0y - ball_r) = 0.
    Returns +inf if the ball never lands within a reasonable horizon."""
    a = 0.5 * G[1]         # -4.905
    b = v0[1]
    c = p0[1] - ball_r
    disc = b * b - 4 * a * c
    if disc < 0:
        return float("inf")
    sq = np.sqrt(disc)
    t1 = (-b - sq) / (2 * a)
    t2 = (-b + sq) / (2 * a)
    cands = [t for t in (t1, t2) if t > 1e-4]
    return min(cands) if cands else float("inf")


def ball_at_time(t, p0, v0, _tb_ignored, restitution_e, friction_f,
                 omega=None, use_drag=False):
    """Position at time t. Bounce time is derived from p0, v0 and Y=ball_radius,
    which enforces a physical bounce ON the ground (not above or below).

    When ``omega`` is None and ``use_drag`` is False, uses the closed-form
    gravity-only trajectory — byte-identical to the pre-Magnus implementation.
    If either Magnus or drag is enabled, falls back to the RK4 integrator;
    this is slower (per-call integration) so the batched
    ``ball_positions_at_times`` should be preferred inside LSQ."""
    needs_rk4 = (omega is not None and not np.allclose(omega, 0.0)) or use_drag
    if not needs_rk4:
        tb = _solve_bounce_time(p0, v0)
        if t <= tb:
            return p0 + v0 * t + 0.5 * G * t**2
        p_b = p0 + v0 * tb + 0.5 * G * tb**2
        v_b = v0 + G * tb
        v_after = np.array([v_b[0] * friction_f, -v_b[1] * restitution_e, v_b[2] * friction_f])
        dt = t - tb
        return p_b + v_after * dt + 0.5 * G * dt**2
    positions, _ = ball_positions_at_times(np.array([t]), p0, v0, restitution_e,
                                           friction_f, omega=omega, use_drag=use_drag)
    return positions[0]


def _accel(v, omega, use_drag):
    """Instantaneous acceleration under gravity (+Magnus, +drag as enabled)."""
    a = G.copy()
    if omega is not None:
        a = a + C_MAG * np.cross(omega, v)
    if use_drag:
        a = a - K_DRAG * np.linalg.norm(v) * v
    return a


def _rk4_step(p, v, dt, omega, use_drag):
    """Single RK4 step for (p, v) under _accel."""
    k1v = _accel(v,              omega, use_drag);  k1p = v
    k2v = _accel(v + 0.5*dt*k1v, omega, use_drag);  k2p = v + 0.5*dt*k1v
    k3v = _accel(v + 0.5*dt*k2v, omega, use_drag);  k3p = v + 0.5*dt*k2v
    k4v = _accel(v + dt*k3v,     omega, use_drag);  k4p = v + dt*k3v
    p_new = p + (dt / 6.0) * (k1p + 2*k2p + 2*k3p + k4p)
    v_new = v + (dt / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)
    return p_new, v_new


def ball_positions_at_times(times, p0, v0, restitution_e, friction_f,
                            omega=None, use_drag=False, dt=2e-3, ball_r=BALL_R):
    """Positions at requested times, under gravity (+Magnus, +drag as enabled),
    with one reflection off Y=ball_r. Returns (positions[N,3], bounce_time).

    - omega=None and use_drag=False → closed-form gravity path (fast, matches
      the pre-Magnus ball_at_time).
    - Otherwise → fixed-step RK4 at dt (default 2 ms, ~500 steps/second;
      cricket deflections converge well under 1 cm vs dt=0.5 ms), followed by
      linear interpolation onto `times`.

    The reflection is applied analytically at the ground-crossing instant
    (linearly interpolated between the two straddling RK4 samples) so that
    bounce timing accuracy is not limited by dt."""
    times = np.asarray(times, dtype=np.float64)
    N = len(times)
    needs_rk4 = (omega is not None and not np.allclose(omega, 0.0)) or use_drag
    if not needs_rk4:
        tb = _solve_bounce_time(p0, v0)
        out = np.empty((N, 3))
        for i, t in enumerate(times):
            if t <= tb:
                out[i] = p0 + v0 * t + 0.5 * G * t**2
            else:
                p_b = p0 + v0 * tb + 0.5 * G * tb**2
                v_b = v0 + G * tb
                v_after = np.array([v_b[0]*friction_f, -v_b[1]*restitution_e, v_b[2]*friction_f])
                dtp = t - tb
                out[i] = p_b + v_after * dtp + 0.5 * G * dtp**2
        return out, tb

    # RK4 path
    t_end = float(times[-1]) + 2.0 * dt
    omega = np.asarray(omega, dtype=np.float64) if omega is not None else None
    n_max = int(t_end / dt) + 4
    ts = np.empty(n_max); ts[0] = 0.0
    ps = np.empty((n_max, 3)); ps[0] = p0
    p = p0.astype(np.float64).copy()
    v = v0.astype(np.float64).copy()
    t = 0.0
    bounced = False
    tb = float("inf")
    i = 1
    while t < t_end and i < n_max:
        p_new, v_new = _rk4_step(p, v, dt, omega, use_drag)
        if (not bounced) and p_new[1] < ball_r and v[1] < 0:
            # linearly interpolate the Y=ball_r crossing within this step
            denom = (p[1] - p_new[1])
            frac = (p[1] - ball_r) / denom if abs(denom) > 1e-12 else 0.0
            frac = max(0.0, min(1.0, frac))
            t_cross = t + frac * dt
            p_cross = p + frac * (p_new - p)
            v_cross = v + frac * (v_new - v)
            v_after = np.array([v_cross[0]*friction_f,
                                -v_cross[1]*restitution_e,
                                v_cross[2]*friction_f])
            # record the bounce sample so interpolation stays smooth across it
            ts[i] = t_cross
            ps[i] = p_cross
            i += 1
            p, v, t = p_cross, v_after, t_cross
            bounced = True
            tb = t_cross
            continue
        p, v, t = p_new, v_new, t + dt
        ts[i] = t
        ps[i] = p
        i += 1

    ts = ts[:i]
    ps = ps[:i]
    out = np.empty((N, 3))
    for k in range(3):
        out[:, k] = np.interp(times, ts, ps[:, k])
    return out, tb


def fit_trajectory(times, uvs, K, R, t_vec, C, use_magnus=False, use_drag=False):
    """Least-squares fit: minimise sum of (proj - obs)^2 over all detections.

    Base parameters: p0(3), v0(3), t_bounce(1), restitution_e(1), friction_f(1).
    With ``use_magnus=True`` three extra parameters are added: ωx, ωy, ωz
    (ball spin in rad/s). ``use_drag=True`` turns on quadratic air drag with
    a fixed Cd=0.5 (no new fit parameters). With both flags off the solver
    uses the closed-form gravity-only trajectory and is byte-identical to the
    pre-Magnus / pre-drag code.
    """
    # Initial guess: release near bowler's popping crease at hand height,
    # ball delivered toward striker (Z=0) at ~35 m/s (fast-medium pace),
    # slight downward and slight side-arm angle.
    x0_base = [
        0.0,  2.0,  18.9,      # p0  (bowler's popping crease, hand height)
        0.0, -2.0, -35.0,      # v0  (mostly -Z toward striker, slight drop)
        (times[-1] - times[0]) * 0.55,   # t_bounce ≈ 55% of window
        0.50,                             # restitution
        0.70,                             # friction
    ]
    # Physically realistic bounds:
    #   restitution (Y) : real cricket ball 0.30 - 0.55
    #   tangential friction : real cricket ball 0.55 - 0.85 (retains most forward
    #     speed; 0.20 was letting fits collapse to "straight up" post-bounce).
    #   release Z (down-pitch) : bowler releases 0.5-2m in front of the bowler's
    #     crease (which is at Z=18.9); so 15-22m works for most deliveries.
    # v_z upper bound = -20 m/s (72 km/h — slowest realistic bowling delivery);
    # keeps the fit out of scale-collapsed local minima where the ball "moves"
    # at walking pace over 1 m instead of 20 m.
    lo_base = [-4.0, 0.5, 15.0,   -25, -20, -50, 0.05, 0.30, 0.55]
    hi_base = [+4.0, 3.5, 25.0,   +25,  +8, -20, 3.00, 0.55, 0.85]

    if use_magnus:
        # ±200 rad/s on each spin axis covers the full physical range of
        # cricket bowling (fast seamers 10-25 rev/s ≈ 60-160 rad/s;
        # wrist/finger spinners 20-30 rev/s ≈ 125-190 rad/s), plus headroom
        # for the seam's extra Magnus coupling absorbed into the fitted ω.
        x0 = np.array(x0_base + [0.0, 0.0, 0.0])
        lo = np.array(lo_base + [-200.0, -200.0, -200.0])
        hi = np.array(hi_base + [+200.0, +200.0, +200.0])
    else:
        x0 = np.array(x0_base)
        lo = np.array(lo_base)
        hi = np.array(hi_base)

    def residuals(params):
        p0 = params[0:3]; v0 = params[3:6]
        e = params[7]; f = params[8]
        omega = params[9:12] if use_magnus else None
        positions, _ = ball_positions_at_times(times, p0, v0, e, f,
                                                omega=omega, use_drag=use_drag)
        r = np.empty(2 * len(times))
        for i, (P, uv) in enumerate(zip(positions, uvs)):
            pr = project_world(P, K, R, t_vec)
            if np.any(np.isnan(pr)):
                r[2*i] = 1e3; r[2*i+1] = 1e3
            else:
                r[2*i]   = pr[0] - uv[0]
                r[2*i+1] = pr[1] - uv[1]
        return r

    result = least_squares(residuals, x0, bounds=(lo, hi), method="trf", max_nfev=500)
    return result


# ───────────────────────────────── export ────────────────────────────────────
def build_track_json(pose_json, fit_result, dets_window, times, fps,
                     sample_hz=240, use_magnus=False, use_drag=False):
    p0 = fit_result.x[0:3]; v0 = fit_result.x[3:6]
    e = float(fit_result.x[7]); f = float(fit_result.x[8])
    omega = np.asarray(fit_result.x[9:12]) if use_magnus else None
    duration = times[-1] - times[0]
    n_samples = max(2, int(duration * sample_hz) + 1)
    ts = np.linspace(0.0, duration, n_samples)

    positions, tb = ball_positions_at_times(ts, p0, v0, e, f,
                                             omega=omega, use_drag=use_drag)
    traj = [{"t": round(float(t), 4),
             "p": [round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)]}
            for t, p in zip(ts, positions)]
    # Guard against the no-bounce case: a fit where the ball never crosses
    # Y=ball_r within the integration window returns tb=inf, which would
    # crash np.array([tb]) → int(inf). Clamp to the trajectory endpoint and
    # mark the record so callers know bounce wasn't observed.
    bounce_observed = np.isfinite(tb)
    tb_eval = float(tb) if bounce_observed else float(ts[-1])
    bounce_arr, _ = ball_positions_at_times(np.array([tb_eval]), p0, v0, e, f,
                                             omega=omega, use_drag=use_drag)
    bounce_p = bounce_arr[0]
    tb = tb_eval if bounce_observed else float("nan")
    intr = pose_json["intrinsics"]
    W, H = pose_json["image_size"]
    return {
        "video": pose_json.get("video"),
        "fps": fps,
        "render": {
            "resolution": [W, H],
            "aspect_ratio": round(W / H, 6),
            "fov_vertical_deg": pose_json["fov_vertical_deg"],
            "note": "Unity camera MUST use this aspect_ratio (Camera.aspect) and fov_vertical_deg (Camera.fieldOfView). Render target should match `resolution` (or same aspect); background video plane must fill the same viewport 1:1 for the CG ball to line up pixel-for-pixel with the source.",
        },
        "camera": {
            "position_m": pose_json["extrinsics_opencv"]["camera_position_world_m"],
            "rotation_matrix_row_major": [
                float(x) for row in pose_json["extrinsics_opencv"]["R"] for x in row
            ],
            "fov_vertical_deg": pose_json["fov_vertical_deg"],
            "intrinsics_px": intr,
            "image_size": pose_json["image_size"],
        },
        "delivery": {
            "release_frame": int(dets_window[0]["frame"]),
            "last_frame":    int(dets_window[-1]["frame"]),
            "impact_frame":  int(dets_window[-1]["frame"]),   # last physics-consistent detection
            "release_pos_m": [round(float(x), 3) for x in p0],
            "release_vel_ms": [round(float(x), 3) for x in v0],
            "release_speed_ms": round(float(np.linalg.norm(v0)), 3),
            "bounce_observed": bool(bounce_observed),
            "bounce_time_s":   round(tb, 4) if bounce_observed else None,
            "bounce_pos_m":    [round(float(x), 3) for x in bounce_p]
                               if bounce_observed else None,
            "restitution":   round(e, 3),
            "tangential_friction": round(f, 3),
            "fit_cost_px2":  round(float(fit_result.cost), 3),
            "drag_enabled":  bool(use_drag),
            **({"drag_coefficient": BALL_CD} if use_drag else {}),
            **({
                "spin_omega_rad_s":   [round(float(x), 3) for x in omega],
                "spin_magnitude_rad_s": round(float(np.linalg.norm(omega)), 3),
                "spin_rev_s":           round(float(np.linalg.norm(omega) / (2*np.pi)), 3),
            } if use_magnus else {}),
        },
        "trajectory": traj,
    }


# ───────────────────────────── check overlay ─────────────────────────────────
def draw_check(video_path, window, times, fit, K, R, t_vec, C, frame_idx,
               use_magnus=False, use_drag=False):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("could not read check frame")
    out = frame.copy()

    p0 = fit.x[0:3]; v0 = fit.x[3:6]
    e = float(fit.x[7]); f = float(fit.x[8])
    omega = np.asarray(fit.x[9:12]) if use_magnus else None

    # densely sampled fitted trajectory reprojected
    n = 200
    ts_dense = np.linspace(0.0, times[-1] - times[0], n)
    dense_positions, tb = ball_positions_at_times(ts_dense, p0, v0, e, f,
                                                   omega=omega, use_drag=use_drag)
    prev = None
    for P in dense_positions:
        uv = project_world(P, K, R, t_vec)
        if not np.any(np.isnan(uv)):
            pt = (int(uv[0]), int(uv[1]))
            if prev is not None:
                cv2.line(out, prev, pt, (0, 200, 255), 2)   # orange = fitted
            prev = pt

    # bounce point
    bounce_arr, _ = ball_positions_at_times(np.array([tb]), p0, v0, e, f,
                                              omega=omega, use_drag=use_drag)
    P_b = bounce_arr[0]
    uv_b = project_world(P_b, K, R, t_vec)
    if not np.any(np.isnan(uv_b)):
        pt_b = (int(uv_b[0]), int(uv_b[1]))
        cv2.circle(out, pt_b, 10, (0, 255, 255), 2)         # yellow ring at bounce
        cv2.putText(out, "bounce", (pt_b[0]+12, pt_b[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # observed detections
    for d in window:
        cv2.circle(out, (int(d["u"]), int(d["v"])), 4, (0, 255, 0), -1)  # green dot

    cv2.rectangle(out, (0, 0), (out.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(out, "green = observed detection   orange = fitted 3D trajectory (reprojected)   "
                     "yellow ring = bounce",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ────────────────────────────────── main ─────────────────────────────────────
def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--detections", required=True, type=Path)
    p.add_argument("--pose",       required=True, type=Path)
    p.add_argument("--video",      required=True, type=Path)
    p.add_argument("--out-track",  required=True, type=Path)
    p.add_argument("--out-check",  required=True, type=Path)
    p.add_argument("--release-before", type=int, default=20)
    p.add_argument("--release-after",  type=int, default=150)
    p.add_argument("--release-frame", type=int, default=None,
                   help="Override auto-detected release frame")
    # Physics defaults are ON. Env vars override: TRACKER_USE_MAGNUS=0 and
    # TRACKER_USE_DRAG=0 disable per-process. --no-magnus / --no-drag disable
    # per-invocation (highest precedence).
    _env_magnus = os.environ.get("TRACKER_USE_MAGNUS", "1").strip() not in (
        "0", "false", "False", "FALSE", "no", "NO", "")
    _env_drag = os.environ.get("TRACKER_USE_DRAG", "1").strip() not in (
        "0", "false", "False", "FALSE", "no", "NO", "")
    _env_shadows = os.environ.get("TRACKER_REJECT_SHADOWS", "1").strip() not in (
        "0", "false", "False", "FALSE", "no", "NO", "")

    p.add_argument("--use-magnus", action="store_true", default=_env_magnus,
                   help="Enable Magnus force in the physics model (adds 3 "
                        "spin parameters: ωx, ωy, ωz in rad/s). Default ON "
                        "(override with --no-magnus or TRACKER_USE_MAGNUS=0).")
    p.add_argument("--no-magnus", dest="use_magnus", action="store_false",
                   help="Disable Magnus force (closed-form gravity path).")
    p.add_argument("--use-drag", action="store_true", default=_env_drag,
                   help="Enable quadratic air drag (fixed Cd=0.5, no new fit "
                        "parameters). Default ON (override with --no-drag or "
                        "TRACKER_USE_DRAG=0).")
    p.add_argument("--no-drag", dest="use_drag", action="store_false",
                   help="Disable air drag (closed-form gravity-only path).")
    p.add_argument("--reject-shadows", action="store_true", default=_env_shadows,
                   help="Drop detections that look like the ball's shadow on "
                        "the pitch. Default ON (override with "
                        "--no-reject-shadows or TRACKER_REJECT_SHADOWS=0).")
    p.add_argument("--no-reject-shadows", dest="reject_shadows",
                   action="store_false",
                   help="Disable shadow rejection (for debugging/comparison).")
    args = p.parse_args(argv)

    det_data = json.loads(args.detections.read_text(encoding="utf-8"))
    pose = json.loads(args.pose.read_text(encoding="utf-8"))
    fps = float(det_data.get("fps", 25.0))
    all_dets = det_data["detections"]

    if args.reject_shadows:
        n_before = len(all_dets)
        all_dets, shadow_rejections = reject_ball_shadows(all_dets)
        n_after = len(all_dets)
        if shadow_rejections:
            print(f"shadow rejection: dropped {n_before - n_after} detection(s) "
                  f"as pitch shadows (of {n_before})")
            for r in shadow_rejections[:10]:
                print(f"  frame {r['frame']:3d}: u={r['u']:.1f}, v={r['v']:.1f}  "
                      f"(Δu={r['shadow_du_px']:.1f}, Δv={r['shadow_dv_px']:.1f} below ball)")
            if len(shadow_rejections) > 10:
                print(f"  ... and {len(shadow_rejections) - 10} more")
        else:
            print(f"shadow rejection: no shadow-like detections found "
                  f"in {n_before} total")

    if args.release_frame is not None:
        release = args.release_frame
        release_source = "CLI"
    elif det_data.get("release_frame") is not None:
        release = int(det_data["release_frame"])
        release_source = "V8m pose-gated ReleaseDetector"
    else:
        release = find_release_frame(all_dets)
        release_source = "bounce-signature heuristic"
    print(f"release source: {release_source}")
    if release is None:
        print("could not auto-detect release; pass --release-frame N", file=sys.stderr)
        return 1
    window = select_window(all_dets, release, args.release_before, args.release_after)
    if len(window) < 5:
        print(f"window has only {len(window)} detections — too few for a fit", file=sys.stderr)
        return 1

    # Build K, R, t, C
    intr = pose["intrinsics"]
    K = np.array([[intr["fx"], 0, intr["cx"]],
                  [0, intr["fy"], intr["cy"]],
                  [0, 0, 1.0]])
    R = np.array(pose["extrinsics_opencv"]["R"], dtype=np.float64)
    tvec = np.array(pose["extrinsics_opencv"]["tvec"], dtype=np.float64)
    C = np.array(pose["extrinsics_opencv"]["camera_position_world_m"], dtype=np.float64)

    # ─── Shrink-until-physics-fits impact discovery ─────────────────────────
    # Try the fit with all detections from release_frame up to a candidate
    # impact. If any detection is more than TOLERANCE_PX from the physics
    # reprojection, drop that candidate and shrink to the previous DETECTED
    # frame (not the previous frame number). Repeat until the fit is
    # physics-consistent or too few points remain.
    TOLERANCE_PX   = 3.0
    MIN_FIT_POINTS = 6

    print(f"release estimate = frame {release}")
    print(f"initial window has {len(window)} detections (frames {window[0]['frame']}..{window[-1]['frame']})")

    fit = None
    per_err = None
    accepted_window = None
    detected_frames_desc = sorted({d["frame"] for d in window}, reverse=True)
    for candidate_impact in detected_frames_desc:
        cand_win = [d for d in window if d["frame"] <= candidate_impact]
        if len(cand_win) < MIN_FIT_POINTS:
            print(f"  too few points ({len(cand_win)}) — stopping shrink loop")
            break
        cand_times = np.array([(d["frame"] - cand_win[0]["frame"]) / fps for d in cand_win])
        cand_uvs   = np.array([[d["u"], d["v"]] for d in cand_win])
        cand_fit = fit_trajectory(cand_times, cand_uvs, K, R, tvec, C,
                                   use_magnus=args.use_magnus,
                                   use_drag=args.use_drag)
        cand_err = np.linalg.norm(cand_fit.fun.reshape(-1, 2), axis=1)
        max_err  = float(cand_err.max())
        print(f"  candidate impact = {candidate_impact:3d}  ({len(cand_win)} pts)  "
              f"max_err={max_err:.2f}px  mean_err={cand_err.mean():.2f}px  "
              f"{'ACCEPT' if max_err <= TOLERANCE_PX else 'reject'}")
        if max_err <= TOLERANCE_PX:
            fit = cand_fit
            per_err = cand_err
            accepted_window = cand_win
            break

    if fit is None:
        print(f"no candidate impact produced max_err <= {TOLERANCE_PX} px — no fit accepted",
              file=sys.stderr)
        return 1

    window = accepted_window
    times = np.array([(d["frame"] - window[0]["frame"]) / fps for d in window])
    uvs   = np.array([[d["u"], d["v"]] for d in window])
    print(f"ACCEPTED window: frames {window[0]['frame']}..{window[-1]['frame']} "
          f"({len(window)} pts, {times[-1]:.2f}s)  impact_frame={window[-1]['frame']}")
    print(f"fit converged: {fit.success}, iterations={fit.nfev}, "
          f"reprojection mean={per_err.mean():.2f}px  max={per_err.max():.2f}px")

    track_json = build_track_json(pose, fit, window, times, fps,
                                   use_magnus=args.use_magnus,
                                   use_drag=args.use_drag)
    args.out_track.parent.mkdir(parents=True, exist_ok=True)
    args.out_track.write_text(json.dumps(track_json, indent=2), encoding="utf-8")
    print(f"wrote track:  {args.out_track}")

    mid_frame = window[len(window)//2]["frame"]
    check = draw_check(args.video, window, times, fit, K, R, tvec, C, mid_frame,
                       use_magnus=args.use_magnus, use_drag=args.use_drag)
    args.out_check.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out_check), check)
    print(f"wrote check:  {args.out_check}")

    d = track_json["delivery"]
    print(f"\nrelease pos:  {d['release_pos_m']} m")
    print(f"release vel:  {d['release_vel_ms']} m/s   (speed {d['release_speed_ms']:.2f} m/s = {d['release_speed_ms']*3.6:.1f} km/h)")
    print(f"bounce at t={d['bounce_time_s']:.3f}s, pos {d['bounce_pos_m']} m")
    print(f"restitution={d['restitution']}, tangential friction={d['tangential_friction']}")
    if args.use_magnus:
        print(f"spin ω={d['spin_omega_rad_s']} rad/s   "
              f"|ω|={d['spin_magnitude_rad_s']:.1f} rad/s "
              f"({d['spin_rev_s']:.1f} rev/s)")
    if args.use_drag:
        print(f"drag enabled (Cd={BALL_CD})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
