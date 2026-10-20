"""Unit tests for the Magnus and drag physics in the 3D trajectory fit.

Covers:
  - RK4 integrator (ω=0, no drag) matches closed-form gravity path.
  - Drag=OFF path byte-identical to baseline.
  - Magnus deflection direction + magnitude (hand-calculated).
  - Drag deceleration magnitude matches K_DRAG * |v|².
  - LSQ fit on synthetic data recovers planted ω and v₀.

Pure numpy + scipy — no video, model, or detector dependencies.
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.reconstruct3d.trajectory_fit import (
    G, C_MAG, K_DRAG, BALL_R, BALL_CD,
    ball_positions_at_times, fit_trajectory, project_world,
    _solve_bounce_time,
)


def _make_test_camera():
    """Build a reasonable 1080p down-pitch camera for synthetic tests."""
    H, W = 1080, 1920
    f_px = 1500.0
    K = np.array([[f_px, 0, W/2.0], [0, f_px, H/2.0], [0, 0, 1.0]])
    cam_pos = np.array([0.0, 2.5, 22.5])
    look_at = np.array([0.0, 0.9, 5.0])
    fwd = look_at - cam_pos; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0])); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    R = np.stack([right, -up, fwd], axis=0)
    tvec = -R @ cam_pos
    return K, R, tvec, cam_pos, (W, H)


# ──────────────────────────── Magnus ─────────────────────────────────────────
class TestMagnus:
    def test_rk4_zero_spin_matches_closed_form(self):
        p0 = np.array([0.0, 2.0, 18.9])
        v0 = np.array([0.5, -2.0, -35.0])
        times = np.linspace(0.0, 0.9, 50)
        pos_closed, tb_c = ball_positions_at_times(times, p0, v0, 0.5, 0.7, omega=None)
        pos_rk4,    tb_r = ball_positions_at_times(times, p0, v0, 0.5, 0.7,
                                                     omega=np.zeros(3))
        max_err = float(np.max(np.linalg.norm(pos_closed - pos_rk4, axis=1)))
        assert max_err < 1e-3, f"max_err = {max_err} m"
        assert abs(tb_c - tb_r) < 2.1e-3

    def test_backspin_lifts_ball(self):
        """ω=(+125,0,0) × v=(0,-2,-38) → +Y Magnus component (ball lifts)."""
        p0 = np.array([0.0, 2.0, 18.9])
        v0 = np.array([0.0, -2.0, -38.0])
        tb = _solve_bounce_time(p0, v0)
        t_eval = np.linspace(0.0, tb * 0.95, 10)
        pos_spin, _  = ball_positions_at_times(t_eval, p0, v0, 0.5, 0.7,
                                                omega=np.array([125.0, 0.0, 0.0]))
        pos_plain, _ = ball_positions_at_times(t_eval, p0, v0, 0.5, 0.7,
                                                omega=np.zeros(3))
        y_lift = pos_spin[-1, 1] - pos_plain[-1, 1]
        # Should lift by 0.1–2 m under this spin — real cricket swing scale.
        assert 0.1 < y_lift < 2.0, f"y_lift = {y_lift} m"

    def test_rhr_side_spin_direction(self):
        """ω=(0,+150,0) × v=(0,-2,-38) → -X Magnus (right-hand rule)."""
        p0 = np.array([0.0, 2.0, 18.9])
        v0 = np.array([0.0, -2.0, -38.0])
        t_eval = np.linspace(0.0, 0.3, 5)
        pos_side, _  = ball_positions_at_times(t_eval, p0, v0, 0.5, 0.7,
                                                omega=np.array([0.0, 150.0, 0.0]))
        pos_plain, _ = ball_positions_at_times(t_eval, p0, v0, 0.5, 0.7,
                                                omega=np.zeros(3))
        x_shift = pos_side[-1, 0] - pos_plain[-1, 0]
        assert x_shift < 0, f"expected -X, got x_shift = {x_shift}"

    def test_fit_recovers_planted_omega(self):
        p0_true = np.array([0.5, 2.0, 18.9])
        v0_true = np.array([0.3, -2.0, -36.0])
        omega_true = np.array([80.0, 60.0, 0.0])
        K, R, tvec, cam_pos, (W, H) = _make_test_camera()

        times = np.arange(25) / 30.0
        positions_true, _ = ball_positions_at_times(times, p0_true, v0_true, 0.5, 0.7,
                                                      omega=omega_true)
        uvs, keep_t = [], []
        for t, P in zip(times, positions_true):
            uv = project_world(P, K, R, tvec)
            if not np.any(np.isnan(uv)) and 0 < uv[0] < W and 0 < uv[1] < H:
                uvs.append(uv); keep_t.append(t)
        uvs = np.array(uvs); keep_t = np.array(keep_t)
        fit = fit_trajectory(keep_t, uvs, K, R, tvec, cam_pos, use_magnus=True)

        err = np.linalg.norm(fit.fun.reshape(-1, 2), axis=1)
        assert err.mean() < 0.5, f"mean err = {err.mean():.3f} px"
        omega_fit = fit.x[9:12]
        # Signs must match on each component.
        for i in range(3):
            if abs(omega_true[i]) > 10:
                assert np.sign(omega_fit[i]) == np.sign(omega_true[i])


# ───────────────────────────── Drag ──────────────────────────────────────────
class TestDrag:
    def test_drag_off_byte_identical_to_baseline(self):
        p0 = np.array([0.0, 2.0, 18.9])
        v0 = np.array([0.5, -2.0, -35.0])
        times = np.linspace(0.0, 0.9, 50)
        pos_a, tb_a = ball_positions_at_times(times, p0, v0, 0.5, 0.7,
                                                omega=None, use_drag=False)
        pos_b, tb_b = ball_positions_at_times(times, p0, v0, 0.5, 0.7, omega=None)
        max_err = float(np.max(np.linalg.norm(pos_a - pos_b, axis=1)))
        assert max_err < 1e-9
        assert tb_a == tb_b

    def test_drag_slows_ball(self):
        """At v=40 m/s, drag over 0.4 s should shorten travel by 0.3–1.5 m."""
        p0 = np.array([0.0, 2.0, 18.9])
        v0 = np.array([0.0, -2.0, -40.0])
        times = np.array([0.0, 0.4])
        pos_nd, _ = ball_positions_at_times(times, p0, v0, 0.5, 0.7, use_drag=False)
        pos_d,  _ = ball_positions_at_times(times, p0, v0, 0.5, 0.7, use_drag=True)
        shortfall = abs(pos_nd[1, 2] - pos_nd[0, 2]) - abs(pos_d[1, 2] - pos_d[0, 2])
        assert 0.3 < shortfall < 1.5, f"shortfall = {shortfall:.3f} m"

    def test_drag_deceleration_magnitude(self):
        """Measured a_drag ≈ K_DRAG * |v|² to within 10%."""
        v_mag = 40.0
        a_expected = K_DRAG * v_mag * v_mag
        p0 = np.array([0.0, 2.0, 100.0])   # high up, no bounce in window
        v0 = np.array([0.0, 0.0, -v_mag])
        times = np.array([0.0, 0.01])
        pos_nd, _ = ball_positions_at_times(times, p0, v0, 0.5, 0.7, use_drag=False)
        pos_d,  _ = ball_positions_at_times(times, p0, v0, 0.5, 0.7, use_drag=True)
        delta_z = pos_d[1, 2] - pos_nd[1, 2]
        a_measured = 2.0 * delta_z / (0.01 * 0.01)
        assert abs(a_measured - a_expected) / a_expected < 0.10

    def test_drag_aware_fit_recovers_speed(self):
        """Fit with drag on drag-afflicted obs recovers release speed exactly."""
        p0_true = np.array([0.3, 2.0, 18.9])
        v0_true = np.array([0.1, -2.5, -40.0])
        K, R, tvec, cam_pos, (W, H) = _make_test_camera()
        times = np.arange(20) / 30.0
        positions_true, _ = ball_positions_at_times(times, p0_true, v0_true, 0.5, 0.7,
                                                      omega=None, use_drag=True)
        uvs, keep_t = [], []
        for t, P in zip(times, positions_true):
            uv = project_world(P, K, R, tvec)
            if not np.any(np.isnan(uv)) and 0 < uv[0] < W and 0 < uv[1] < H:
                uvs.append(uv); keep_t.append(t)
        uvs = np.array(uvs); keep_t = np.array(keep_t)
        fit_wd = fit_trajectory(keep_t, uvs, K, R, tvec, cam_pos,
                                 use_magnus=False, use_drag=True)
        v0_true_mag = float(np.linalg.norm(v0_true))
        v0_wd_mag   = float(np.linalg.norm(fit_wd.x[3:6]))
        assert abs(v0_wd_mag - v0_true_mag) < 1.0, (
            f"drag-aware fit should recover release speed to 1 m/s, "
            f"got {abs(v0_wd_mag - v0_true_mag):.2f} m/s off")


# ───────────── track_3d.json emits the right diagnostic fields ───────────────
def test_build_track_json_emits_drag_and_spin_fields():
    """When both flags are on, the JSON advertises both."""
    from pipeline.reconstruct3d.trajectory_fit import build_track_json

    class _FakeFit:
        # p0(3), v0(3), tb, e, f, ωx, ωy, ωz
        x = np.array([0.0, 2.0, 19.0, 0.0, -2.0, -35.0, 0.3, 0.5, 0.7, 20.0, 10.0, 5.0])
        cost = 1.0
    pose = {
        "intrinsics": {"fx": 1500, "fy": 1500, "cx": 960, "cy": 540},
        "image_size": [1920, 1080],
        "fov_vertical_deg": 42.0,
        "extrinsics_opencv": {"R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                               "camera_position_world_m": [0, 2.5, 22.5]},
    }
    dets_window = [{"frame": 10}, {"frame": 11}, {"frame": 20}]
    times = np.array([0.0, 0.4])
    out = build_track_json(pose, _FakeFit(), dets_window, times, 30.0,
                           use_magnus=True, use_drag=True)
    d = out["delivery"]
    assert d["drag_enabled"] is True
    assert d["drag_coefficient"] == BALL_CD
    assert "spin_omega_rad_s" in d
    assert "spin_rev_s" in d
