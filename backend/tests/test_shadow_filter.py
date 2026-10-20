"""Unit tests for backend.pipeline.shadow_filter.

Covers both the native (3D-pipeline) format and the legacy (process.py)
adapter. These are pure-function tests — no video / model dependencies.
"""
from __future__ import annotations

import pytest

from pipeline.shadow_filter import (
    reject_ball_shadows,
    reject_ball_shadows_legacy,
    reject_shadows_uv,
)


# ───────────────────────── native uv format ──────────────────────────────────
def test_single_detection_passes_through():
    dets = [{"frame": 10, "u": 500.0, "v": 300.0, "conf": 0.9}]
    kept, rej = reject_ball_shadows(dets)
    assert len(kept) == 1
    assert len(rej) == 0


def test_planted_shadow_removed_ball_kept():
    """Ball at (500, 300), shadow at (502, 450) — same u ± 2 px, 150 px below."""
    dets = [
        {"frame": 10, "u": 500.0, "v": 300.0, "conf": 0.80},  # ball (higher)
        {"frame": 10, "u": 502.0, "v": 450.0, "conf": 0.75},  # shadow
    ]
    kept, rej = reject_ball_shadows(dets)
    assert len(kept) == 1 and len(rej) == 1
    assert kept[0]["v"] == 300.0
    assert rej[0]["shadow_of_frame"] == 10
    assert rej[0]["shadow_du_px"] == 2.0
    assert rej[0]["shadow_dv_px"] == 150.0


def test_horizontally_spread_duplicates_both_kept():
    """Two detections far apart horizontally — not a shadow pair."""
    dets = [
        {"frame": 10, "u": 500.0, "v": 300.0, "conf": 0.80},
        {"frame": 10, "u": 900.0, "v": 460.0, "conf": 0.70},
    ]
    kept, rej = reject_ball_shadows(dets)
    assert len(kept) == 2
    assert len(rej) == 0


def test_tiny_v_separation_both_kept():
    """Below the v_separation_px threshold — treat as noise, keep both."""
    dets = [
        {"frame": 10, "u": 500.0, "v": 300.0, "conf": 0.80},
        {"frame": 10, "u": 502.0, "v": 310.0, "conf": 0.75},
    ]
    kept, rej = reject_ball_shadows(dets)
    assert len(kept) == 2
    assert len(rej) == 0


def test_mixed_three_detections():
    """Ball + its shadow + unrelated horizontal false positive."""
    dets = [
        {"frame": 10, "u": 500.0,  "v": 300.0, "conf": 0.85},  # ball
        {"frame": 10, "u": 501.0,  "v": 460.0, "conf": 0.70},  # shadow
        {"frame": 10, "u": 1200.0, "v": 400.0, "conf": 0.60},  # unrelated FP
    ]
    kept, rej = reject_ball_shadows(dets)
    assert len(kept) == 2
    assert len(rej) == 1
    assert rej[0]["v"] == 460.0
    assert {k["v"] for k in kept} == {300.0, 400.0}


def test_configurable_thresholds():
    """Tight u tolerance stops a slightly-offset shadow being caught."""
    dets = [
        {"frame": 1, "u": 500.0, "v": 300.0, "conf": 0.8},
        {"frame": 1, "u": 530.0, "v": 500.0, "conf": 0.7},
    ]
    # Default tolerance (40 px) catches it.
    _, rej = reject_ball_shadows(dets)
    assert len(rej) == 1
    # Tighter tolerance (20 px) does not.
    _, rej = reject_ball_shadows(dets, u_tolerance_px=20.0)
    assert len(rej) == 0


# ─────────────────── legacy process.py format adapter ────────────────────────
def test_legacy_single_ball_passes_through():
    dets = [{"x1": 490, "y1": 290, "x2": 510, "y2": 310,
             "class_name": "ball", "conf": 0.9}]
    kept, n_rej = reject_ball_shadows_legacy(dets, frame_index=42)
    assert kept == dets
    assert n_rej == 0


def test_legacy_shadow_dropped_ball_kept():
    """Ball bbox centered at (500, 300), shadow bbox centered at (501, 450)."""
    ball = {"x1": 490, "y1": 290, "x2": 510, "y2": 310,
            "class_name": "ball", "conf": 0.80}
    shadow = {"x1": 491, "y1": 440, "x2": 511, "y2": 460,
              "class_name": "ball", "conf": 0.75}
    dets = [ball, shadow]
    kept, n_rej = reject_ball_shadows_legacy(dets, frame_index=42)
    assert n_rej == 1
    assert ball in kept
    assert shadow not in kept


def test_legacy_non_ball_classes_pass_through_untouched():
    """Bat detections should never be filtered, even if positioned like shadow."""
    ball = {"x1": 490, "y1": 290, "x2": 510, "y2": 310,
            "class_name": "ball", "conf": 0.8}
    bat_below = {"x1": 491, "y1": 440, "x2": 511, "y2": 460,
                 "class_name": "bat", "conf": 0.75}
    kept, n_rej = reject_ball_shadows_legacy([ball, bat_below], frame_index=42)
    assert n_rej == 0
    assert ball in kept and bat_below in kept


def test_legacy_mixed_ball_bat_shadow():
    ball = {"x1": 490, "y1": 290, "x2": 510, "y2": 310,
            "class_name": "ball", "conf": 0.9}
    shadow = {"x1": 491, "y1": 440, "x2": 511, "y2": 460,
              "class_name": "ball", "conf": 0.6}
    bat = {"x1": 100, "y1": 500, "x2": 200, "y2": 600,
           "class_name": "bat", "conf": 0.8}
    kept, n_rej = reject_ball_shadows_legacy([ball, shadow, bat], frame_index=0)
    assert n_rej == 1
    assert bat in kept
    assert ball in kept
    assert shadow not in kept


def test_legacy_preserves_detection_schema():
    """Original dict fields must survive the round trip through the adapter."""
    ball = {"x1": 490, "y1": 290, "x2": 510, "y2": 310,
            "class_name": "ball", "conf": 0.8, "extra_field": "keep_me"}
    kept, _ = reject_ball_shadows_legacy([ball], frame_index=0)
    assert kept[0] is ball  # same object reference
    assert kept[0]["extra_field"] == "keep_me"
