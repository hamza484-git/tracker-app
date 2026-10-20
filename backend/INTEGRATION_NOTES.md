# Physics + shadow-rejection integration — operator notes

## What's integrated

| Feature | Where it runs | Default |
|---|---|---|
| Ball-shadow rejection | Live 2D pipeline (`pipeline/process.py` Pass 1) **and** 3D CLI | **ON** |
| Magnus force | 3D CLI only (`pipeline/reconstruct3d/trajectory_fit.py`) | **ON** |
| Quadratic air drag (Cd=0.5 fixed) | 3D CLI only | **ON** |

> **User impact today:** only shadow rejection reaches users (via `process.py`). Magnus and drag live in the 3D CLI, which is not yet wired to the frontend. See the strategy doc for the Phase 2–4 plan.

## Rollback — per feature

### Shadow rejection (live 2D pipeline)
```bash
# Disable globally
TRACKER_REJECT_SHADOWS=0

# Verify
grep shadow_rejection_enabled _tracker_work/jobs/<job_id>/results.json
# → "shadow_rejection_enabled": false
```
The 2D pipeline's `results.json` grows a `diagnostics` block:
```json
"diagnostics": {
  "shadow_rejections": 0,
  "shadow_rejection_enabled": true
}
```
Non-zero `shadow_rejections` means pitch shadows were caught on that job.

### Magnus force (3D CLI)
```bash
# One-shot
python -m backend.pipeline.reconstruct3d.trajectory_fit --no-magnus ...

# Globally
TRACKER_USE_MAGNUS=0
```

### Quadratic drag (3D CLI)
```bash
# One-shot
python -m backend.pipeline.reconstruct3d.trajectory_fit --no-drag ...

# Globally
TRACKER_USE_DRAG=0
```

### Everything off at once (back to pre-integration baseline)
```bash
TRACKER_REJECT_SHADOWS=0 TRACKER_USE_MAGNUS=0 TRACKER_USE_DRAG=0 python ...
```

## Precedence

1. Explicit CLI flag (`--no-magnus`, `--no-drag`, `--no-reject-shadows`)
2. Env var (`TRACKER_USE_MAGNUS=0`, etc.)
3. Default (ON)

## New `track_3d.json` fields (3D CLI)

```json
"delivery": {
  "drag_enabled": true,
  "drag_coefficient": 0.5,
  "bounce_observed": true,
  "bounce_time_s": 0.384,
  "bounce_pos_m": [-0.10, 0.036, 6.73],
  "spin_omega_rad_s": [-32.4, -101.8, -79.8],
  "spin_magnitude_rad_s": 133.3,
  "spin_rev_s": 21.2,
  "release_speed_ms": 48.08,
  "fit_cost_px2": 0.699
}
```

`bounce_observed=false` + `bounce_time_s=null` + `bounce_pos_m=null` means the fit converged but the ball never crossed Y=ball_radius inside the integration window (yorker / out-of-window fit). Previously this crashed `build_track_json`; now it returns cleanly.

## Tests

```bash
cd backend
python -m pytest tests/test_shadow_filter.py tests/test_trajectory_physics.py -v
```
Expected: 20 passed. These are pure-numpy/scipy tests — no video or model dependencies, safe to run in CI.

The pre-existing `test_pipeline_isolation.py` requires `TRACKER_LEGACY_DIR` + ONNX weights + a sample video — not part of the CI-safe set.

## Code touched

- **NEW** `backend/pipeline/shadow_filter.py` — shared filter with both-format support
- **NEW** `backend/tests/test_shadow_filter.py` — 11 unit tests
- **NEW** `backend/tests/test_trajectory_physics.py` — 9 Magnus + drag tests
- `backend/pipeline/process.py` — Pass 1 shadow filter call, env var, `diagnostics` field
- `backend/pipeline/reconstruct3d/trajectory_fit.py` — defaults flipped, env-var fallback, no-bounce guard, inline shadow filter removed (re-exported from shadow_filter.py)

## Known limits

- **Magnus + drag only run on the 3D CLI path.** No user video goes through that path today — the live FastAPI → `process.py` → 2D parabola path is unchanged. The physics work is production-ready but not production-reachable; wiring the frontend to use the 3D pipeline is a separate rollout (see strategy doc, Phases 2–4).
- **3D fit is slow with physics on** (~30–40 s per delivery vs ~0.1 s baseline). Needs warm-start + analytic Jacobian before user-facing use. Target p95 ≤ 3 s.
- **Shadow filter default is strict** (`v_separation_px=20`, `u_tolerance_px=40`) — catches shadows directly below the ball under overhead lighting. Shadows under angled light (sun to the side) with large horizontal offset would need wider `u_tolerance_px` and could add false rejections; left strict until real traffic proves otherwise.
- **`v_z` lower bound still −50 m/s** on fast deliveries. Known pin on 4_027_06 sample. Widening to −55 m/s is a follow-up.
- **`ω_z` bound still ±200 rad/s.** Known pin on 4_027_06. Widening to ±400 rad/s is a follow-up.
