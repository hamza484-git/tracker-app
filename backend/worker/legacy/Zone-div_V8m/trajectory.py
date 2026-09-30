"""
TRAJECTORY — post-bounce ball-position history plus deviation detection.

Stores every real ball detection as (frame, cx, cy). After bounce, watches
for a direction change (predicted vs actual velocity angle > threshold) and
records the first deviation frame.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional


class Trajectory:
    def __init__(self):
        self._pts: list[tuple[int, int, int]] = []   # (frame, x, y)
        # Direction-change markers: list of (frame, x, y) where the ball's
        # horizontal velocity flipped against the predicted direction.
        self._dir_changes: list[tuple[int, int, int]] = []
        # Optional bat-zone hint (populated by process.py after calibration).
        # Used only by the source-level detection filter below to decide
        # whether an off-trajectory candidate should still be accepted as a
        # real bat deflection.
        self._bat_zone_cx: Optional[float] = None
        self._bat_zone_cy: Optional[float] = None
        self._bat_zone_half_w: Optional[float] = None
        self._bat_zone_half_h: Optional[float] = None
        # Debug: reasons candidates were rejected. Not surfaced anywhere yet;
        # useful when tuning thresholds against a specific problem clip.
        self._filter_rejects: list[tuple[int, int, int, str]] = []

    # --- Source-level detection filter (2026-09-30) ----------------------------
    # Problem this solves: the ball detector occasionally scores a
    # false-positive (helmet, shadow, dark pitch patch, kit) higher than the
    # real ball on a given frame. Once that fake gets appended to _pts, every
    # downstream detector — gap, parabolic, last-seen — treats it as truth,
    # and the reported deviation lands wherever the false positive was.
    #
    # This filter runs INSIDE update() before a detection lands in _pts. It
    # only kicks in post-bounce, once we have enough context (>= 3 accepted
    # post-bounce points) to extrapolate an expected next position. From the
    # multiple ball candidates the detector emitted this frame, it picks the
    # one closest to that expected position. If even the best candidate is
    # too far from the extrapolation AND lies outside the bat zone (so it's
    # not a plausible deflection either), the whole frame is dropped.
    #
    # Kill-switch: set _ENABLE_SOURCE_FILTER = False to restore pre-filter
    # behaviour with zero risk.
    _ENABLE_SOURCE_FILTER = True
    # Minimum radius (px) allowed around the extrapolated position, before
    # scaling with the ball's speed. Keeps the filter from being too tight on
    # slow deliveries where the velocity-scaled slack alone would be tiny.
    _SOURCE_FILTER_MIN_RADIUS = 60
    # Multiplier on the ball's current speed × frame gap. 2.0 = allow the ball
    # to move at up to 2× its recent velocity before flagging as outlier.
    _SOURCE_FILTER_SPEED_MULT = 2.0
    # Additive slack (px) on top of the velocity envelope — a fixed detection
    # jitter budget.
    _SOURCE_FILTER_PAD = 30
    # Bat-zone escape hatch: even a far-off candidate is accepted if it lands
    # within the calibrated bat zone plus this padding — the ball genuinely
    # DOES jump when the bat deflects it, and that's what we want to catch.
    _SOURCE_FILTER_BAT_ZONE_PAD = 50

    def set_bat_zone(
        self,
        cx: Optional[float],
        cy: Optional[float],
        half_w: Optional[float],
        half_h: Optional[float],
    ) -> None:
        """Store the batter-end bat zone in pixel coords. Called by
        process.py after calibration. All None → the source filter's escape
        hatch is disabled (real deflections may then be rejected as noise)."""
        self._bat_zone_cx = cx
        self._bat_zone_cy = cy
        self._bat_zone_half_w = half_w
        self._bat_zone_half_h = half_h

    def _is_within_bat_zone(self, cx: float, cy: float) -> bool:
        if (self._bat_zone_cx is None or self._bat_zone_cy is None
                or self._bat_zone_half_w is None
                or self._bat_zone_half_h is None):
            # No zone configured → be permissive so we don't over-reject.
            return True
        pad = self._SOURCE_FILTER_BAT_ZONE_PAD
        return (abs(cx - self._bat_zone_cx) <= self._bat_zone_half_w + pad
                and abs(cy - self._bat_zone_cy) <= self._bat_zone_half_h + pad)

    def _select_post_bounce_candidate(
        self,
        candidates: list[tuple[dict, int, int]],
        frame_num: int,
        bounce_frame: int,
    ) -> Optional[tuple[dict, int, int]]:
        """Pick the best of several ball candidates for a post-bounce frame,
        or return None if all of them look like false positives.

        `candidates` is a list of (detection_dict, cx, cy). Pre-bounce or
        with too few points of context we fall back to the highest-confidence
        candidate, matching legacy behaviour.
        """
        # Post-bounce points already accepted into _pts.
        post_bounce = [p for p in self._pts if p[0] >= bounce_frame]
        # Not enough context to extrapolate → don't filter yet.
        if len(post_bounce) < 3:
            return max(candidates, key=lambda t: t[0].get("conf", 0.0))

        # Linear velocity from the last two accepted points.
        f0, x0, y0 = post_bounce[-2]
        f1, x1, y1 = post_bounce[-1]
        df = max(f1 - f0, 1)
        vx = (x1 - x0) / df
        vy = (y1 - y0) / df
        dt = frame_num - f1
        if dt <= 0:
            return max(candidates, key=lambda t: t[0].get("conf", 0.0))
        pred_x = x1 + vx * dt
        pred_y = y1 + vy * dt

        speed = (vx * vx + vy * vy) ** 0.5
        max_dist = max(
            self._SOURCE_FILTER_MIN_RADIUS,
            speed * dt * self._SOURCE_FILTER_SPEED_MULT,
        ) + self._SOURCE_FILTER_PAD

        # Rank candidates by proximity to predicted position.
        scored = sorted(
            candidates,
            key=lambda t: (t[1] - pred_x) ** 2 + (t[2] - pred_y) ** 2,
        )
        best = scored[0]
        _, bx, by = best
        best_dist = ((bx - pred_x) ** 2 + (by - pred_y) ** 2) ** 0.5

        if best_dist <= max_dist:
            return best
        # Best candidate is far from extrapolated path. Only accept it if it
        # plausibly landed at the bat — real deflection, not noise.
        if self._is_within_bat_zone(bx, by):
            return best
        self._filter_rejects.append((
            frame_num, int(bx), int(by),
            f"far {best_dist:.0f}>{max_dist:.0f} pred=({pred_x:.0f},{pred_y:.0f}) out of bat zone",
        ))
        return None

    # Minimum velocity magnitude (px/frame) on both predicted and actual
    # sides before we consider deviation meaningful (filters jitter on a
    # near-stationary ball).
    _MIN_SPEED = 1.5
    # Angular deviation threshold (degrees) between predicted and actual
    # velocity vectors. Anything beyond this is flagged as a deviation event.
    # Used by the legacy linear-velocity detector (fallback when parabolic
    # residual can't fit — e.g. too few post-bounce points).
    _DEVIATION_ANGLE_DEG = 10.0

    # --- Gap-based deviation detector (2026-09-15, primary) --------------------
    # Physical rationale: during bat impact the ball goes behind/into the bat and
    # is occluded + heavily motion-blurred for a few frames, then reappears on a
    # different trajectory. That gap in the tracked-ball timeline IS the impact
    # signal — the first post-gap frame is where the ball has separated from the
    # bat and become trackable again. This method is robust to noise because
    # it uses the absence of detections rather than fitting through them.
    #
    # Revert path: set _USE_GAP_DEVIATION = False to skip this method.
    _USE_GAP_DEVIATION = True
    # Minimum trajectory gap (in frames) that qualifies as a bat-impact
    # candidate. Smaller = more sensitive (catches quick impacts) but risks
    # firing on single-frame detector dropouts. 3 is a solid middle.
    _GAP_MIN_FRAMES = 3
    # Maximum gap size — beyond this, the tracker probably lost the ball for
    # non-impact reasons (subject leaving frame, hard occlusion, model failure).
    _GAP_MAX_FRAMES = 20
    # Maximum bounce-to-impact window (frames). Real cricket deliveries have
    # the ball reach the bat within ~10-20 frames of bounce at 30fps. Gaps
    # beyond this are probably unrelated tracking failures.
    _GAP_MAX_BOUNCE_TO_IMPACT = 25

    # --- Last-seen fallback (2026-09-15, tertiary) -----------------------------
    # For dead-defensive shots (ball hits bat/pad with no rebound) the ball
    # simply stops being visible after bounce. There's no gap-then-recovery, no
    # parabola to fit, no linear velocity change to compare against. Without
    # this fallback the pipeline reports deviation:null — technically correct
    # but not useful for the UI. This heuristic accepts the last tracked ball
    # position as the deviation marker, as long as tracking died close enough
    # to the bounce to plausibly represent an impact.
    #
    # Revert path: set _USE_LAST_SEEN_FALLBACK = False.
    _USE_LAST_SEEN_FALLBACK = True
    # Only fire if post-bounce trajectory has at most this many points.
    # For hard defensive shots the ball is lost between bounce and impact —
    # typically leaving 5–8 tracked frames after bounce. Anything longer than
    # this means gap/parabolic should have already fired.
    _LAST_SEEN_MAX_POST_BOUNCE = 12
    # Last tracked ball must be within this many frames of bounce — beyond
    # that, the ball probably wasn't lost from impact but from something else.
    _LAST_SEEN_MAX_FRAMES_FROM_BOUNCE = 15
    # After finding the last-tracked position, extrapolate the recent motion
    # forward. This is now the MAX number of frames the extrapolator searches;
    # it stops early when the projection crosses the batter's stance zone
    # (from calibration). 0 disables extrapolation. Bumped up because the
    # early-stop logic prevents over-projection.
    _LAST_SEEN_EXTRAPOLATE_FRAMES = 8
    # Absolute hard cap on marker displacement from last-tracked ball — even
    # with parabolic fit + calibration guidance, we never move the marker
    # further than this. Guards against runaway extrapolations on noisy fits.
    _LAST_SEEN_EXTRAPOLATE_MAX_PX = 120

    # --- Parabolic-residual deviation detector (2026-09-15, secondary) ---------
    # Fallback for clips where the ball IS tracked continuously through impact
    # (rare — hard shots without occlusion, or a model good enough to hold the
    # ball through motion blur). Fits a parabola to the first stable post-bounce
    # detections, extrapolates forward, flags where actual position diverges by
    # more than a threshold for at least N consecutive frames.
    #
    # Revert path: set _USE_PARABOLIC_DEVIATION = False.
    _USE_PARABOLIC_DEVIATION = True
    # Number of post-bounce detections used to fit the parabola.
    _PARABOLA_FIT_POINTS = 6
    # Pixel distance between actual detection and parabola extrapolation.
    _RESIDUAL_THRESHOLD_PX = 25.0
    # Consecutive over-threshold frames required before firing.
    _RESIDUAL_CONFIRM_FRAMES = 2

    def update(
        self,
        detections: list,
        frame_num: int,
        release_frame: Optional[int] = None,
        bounce_frame:  Optional[int] = None,
    ) -> None:
        balls = [d for d in detections if d.get("class_name") == "ball"]
        if not balls:
            return
        # Compute the centre for each candidate up front so both paths share
        # the same coord derivation.
        candidates = [
            (d, (d["x1"] + d["x2"]) // 2, (d["y1"] + d["y2"]) // 2)
            for d in balls
        ]

        # Source-level filter: post-bounce only, and only if enabled. Otherwise
        # keep legacy behaviour (pick the highest-confidence candidate).
        if (self._ENABLE_SOURCE_FILTER
                and bounce_frame is not None
                and frame_num >= bounce_frame):
            picked = self._select_post_bounce_candidate(
                candidates, frame_num, bounce_frame,
            )
            if picked is None:
                return  # all candidates look like false positives — drop frame
            _, cx, cy = picked
        else:
            _, cx, cy = max(candidates, key=lambda t: t[0].get("conf", 0.0))

        self._pts.append((frame_num, cx, cy))
        self._check_direction_change(release_frame, bounce_frame)

    def _check_last_seen_fallback(
        self,
        bounce_frame: int,
        bat_line_x: Optional[float] = None,
        bat_line_y: Optional[float] = None,
    ) -> bool:
        """
        Last-seen-position fallback for dead-defensive shots.

        When ball tracking dies within a few frames of bounce and never
        recovers, take the last tracked position as the deviation marker.
        This is approximate — the true impact is probably a few frames later
        than the last-seen point — but it's more useful than null and it
        lands close to the actual impact zone (bat/pad in front of stumps).

        Only fires when:
          * post-bounce trajectory has <= _LAST_SEEN_MAX_POST_BOUNCE points
          * the last point is within _LAST_SEEN_MAX_FRAMES_FROM_BOUNCE frames
            of the bounce (otherwise it's probably an unrelated tracking loss)

        Returns True if a deviation was recorded.
        """
        ordered = sorted(self._pts, key=lambda p: p[0])
        post_bounce = [p for p in ordered if p[0] >= bounce_frame]
        if len(post_bounce) < 2:
            # Need at least the bounce point plus one more so "last seen" is
            # different from "bounce" — otherwise we'd double-mark.
            return False
        if len(post_bounce) > self._LAST_SEEN_MAX_POST_BOUNCE:
            # Long trajectory — gap/parabolic should've handled it. Not our job.
            return False
        f_last, x_last, y_last = post_bounce[-1]
        if f_last - bounce_frame > self._LAST_SEEN_MAX_FRAMES_FROM_BOUNCE:
            return False
        if f_last == bounce_frame:
            return False  # only the bounce itself — nothing to mark

        # Where the marker lands is the ball at the bat-impact plane. Two paths:
        #   (1) The tracked trajectory already crossed the batter's stance zone
        #       (bat_line_y from calibration) somewhere between two tracked
        #       points — interpolate between them for an exact position.
        #   (2) The tracked ball stopped short of the zone — extrapolate
        #       forward along a parabolic fit until the trajectory reaches it.
        # This is dramatically more accurate than just projecting last-known
        # velocity forward, because it uses (a) the ACTUAL trajectory shape
        # for the tracked portion and (b) the physical stopping point from
        # calibration for the untracked portion.
        f_dev, x_dev, y_dev = f_last, x_last, y_last
        used_method = "last-seen"

        def _step_dir(tail_val: float, target: float) -> int:
            return 1 if target > tail_val else -1

        # --- (1) Look for a tracked-trajectory crossing of the bat zone. ---
        # The batter's stance is above the pitch surface (smaller y in screen
        # coords for a behind-the-bowler cam). Post-bounce ball rises through
        # bat_line_y from below.
        crossed = False
        if bat_line_y is not None and len(post_bounce) >= 2:
            # Find first consecutive pair (p_i, p_{i+1}) where y straddles
            # bat_line_y with the ball moving toward it.
            for i in range(len(post_bounce) - 1):
                f0, x0, y0 = post_bounce[i]
                f1, x1, y1 = post_bounce[i + 1]
                # Movement direction toward bat_line_y at this segment.
                if (y0 - bat_line_y) * (y1 - bat_line_y) <= 0 and y0 != y1:
                    # Straddles bat_line_y — linear interpolate.
                    t = (bat_line_y - y0) / (y1 - y0)
                    t = max(0.0, min(1.0, t))
                    f_dev = int(round(f0 + t * (f1 - f0)))
                    x_dev = int(round(x0 + t * (x1 - x0)))
                    y_dev = int(round(bat_line_y))
                    used_method = "bat-zone crossing (tracked)"
                    crossed = True
                    break

        # --- (2) Extrapolate forward using a parabolic fit. ---
        extrap = self._LAST_SEEN_EXTRAPOLATE_FRAMES
        if not crossed and extrap > 0 and len(post_bounce) >= 2:
            fit_pts = post_bounce[-min(6, len(post_bounce)):]
            ts = [float(p[0]) for p in fit_pts]
            xs = [float(p[1]) for p in fit_pts]
            ys = [float(p[2]) for p in fit_pts]
            can_para = len(fit_pts) >= 4 and len(set(ts)) >= 3
            if can_para:
                ts_a = np.array(ts); xs_a = np.array(xs); ys_a = np.array(ys)
                cx = np.polyfit(ts_a, xs_a, 2)
                cy = np.polyfit(ts_a, ys_a, 2)
                predict = lambda t: (float(np.polyval(cx, t)),
                                     float(np.polyval(cy, t)))
                used_method = "parabolic extrapolation"
            else:
                f0, x0, y0 = fit_pts[0]
                f1, x1, y1 = fit_pts[-1]
                df = max(f1 - f0, 1)
                vx = (x1 - x0) / df; vy = (y1 - y0) / df
                predict = lambda t: (x_last + vx * (t - f_last),
                                     y_last + vy * (t - f_last))
                used_method = "linear extrapolation"

            max_px = float(self._LAST_SEEN_EXTRAPOLATE_MAX_PX)
            best_x, best_y, best_df = x_last, y_last, 0
            tail_x_ref = fit_pts[-2][1] if len(fit_pts) >= 2 else x_last
            tail_y_ref = fit_pts[-2][2] if len(fit_pts) >= 2 else y_last
            prev_x, prev_y = float(x_last), float(y_last)
            prev_f = float(f_last)
            for step in range(1, extrap + 1):
                px, py = predict(f_last + step)
                # Check if the segment (prev -> predicted) crosses the bat
                # zone; if so, interpolate INSIDE that step so we land on
                # the zone instead of overshooting past it.
                land_at = None
                if bat_line_y is not None and py != prev_y:
                    if (prev_y - bat_line_y) * (py - bat_line_y) <= 0:
                        u = (bat_line_y - prev_y) / (py - prev_y)
                        u = max(0.0, min(1.0, u))
                        land_at = (prev_f + u * ((f_last + step) - prev_f),
                                   prev_x + u * (px - prev_x),
                                   bat_line_y)
                if land_at is None and bat_line_x is not None and px != prev_x:
                    if (prev_x - bat_line_x) * (px - bat_line_x) <= 0:
                        u = (bat_line_x - prev_x) / (px - prev_x)
                        u = max(0.0, min(1.0, u))
                        land_at = (prev_f + u * ((f_last + step) - prev_f),
                                   bat_line_x,
                                   prev_y + u * (py - prev_y))
                if land_at is not None:
                    lf, lx, ly = land_at
                    # Cap displacement even here.
                    dx = lx - x_last; dy = ly - y_last
                    mag = (dx * dx + dy * dy) ** 0.5
                    if mag > max_px:
                        s = max_px / mag
                        lx = x_last + dx * s
                        ly = y_last + dy * s
                    best_x, best_y = lx, ly
                    best_df = int(round(lf - f_last))
                    break

                # No crossing this step — accept the raw prediction (capped).
                dx = px - x_last; dy = py - y_last
                mag = (dx * dx + dy * dy) ** 0.5
                if mag > max_px:
                    s = max_px / mag
                    px = x_last + dx * s
                    py = y_last + dy * s
                best_x, best_y, best_df = px, py, step
                prev_x, prev_y = px, py
                prev_f = f_last + step
            f_dev = f_last + best_df
            x_dev = int(round(best_x))
            y_dev = int(round(best_y))

        self._dir_changes.append((f_dev, int(x_dev), int(y_dev)))
        print("\n[DEVIATION POINT — last-seen fallback]")
        print(f"  Method      : {used_method}")
        print(f"  Frame       : {f_dev}  (last-seen was F{f_last})")
        print(f"  Coordinates : x={x_dev}, y={y_dev}")
        print(f"  Post-bounce : {len(post_bounce)} points, "
              f"{f_last - bounce_frame} frames from bounce\n")
        return True

    def _check_gap_deviation(self, bounce_frame: int) -> bool:
        """
        Gap-based deviation detector.

        The bat-impact frame is masked by occlusion + motion blur, producing a
        gap in the tracked-ball timeline. The first post-gap detection is
        essentially the ball re-emerging from the bat — that IS the impact
        location.

        Scans post-bounce detections in order, finds the first gap of size
        [_GAP_MIN_FRAMES.._GAP_MAX_FRAMES] that starts within
        _GAP_MAX_BOUNCE_TO_IMPACT frames of bounce, and records the first
        post-gap point as the deviation.

        Returns True if a deviation was recorded.
        """
        ordered = sorted(self._pts, key=lambda p: p[0])
        post_bounce = [p for p in ordered if p[0] >= bounce_frame]
        if len(post_bounce) < 2:
            return False

        for i in range(len(post_bounce) - 1):
            f_prev = post_bounce[i][0]
            f_next = post_bounce[i + 1][0]
            gap = f_next - f_prev - 1  # number of missing frames between them
            # Gap must be big enough to be an impact (not tracker jitter),
            # but not so big it's unrelated tracking failure.
            if gap < self._GAP_MIN_FRAMES or gap > self._GAP_MAX_FRAMES:
                continue
            # Impact must happen within a reasonable window of bounce.
            if f_prev - bounce_frame > self._GAP_MAX_BOUNCE_TO_IMPACT:
                continue
            # The gap's leading edge must be strictly AFTER bounce — a gap
            # that starts exactly at bounce_frame is the bounce itself.
            if f_prev < bounce_frame + 1:
                continue

            # Fire: the first post-gap point is the deviation.
            f_dev, x_dev, y_dev = post_bounce[i + 1]
            self._dir_changes.append((f_dev, x_dev, y_dev))
            print("\n[DEVIATION POINT — trajectory gap]")
            print(f"  Frame       : {f_dev}  (post-gap emergence)")
            print(f"  Coordinates : x={f_dev}, y={y_dev}")
            print(f"  Gap span    : F{f_prev} -> F{f_next}  ({gap} missing frames)")
            print(f"  Bounce dist : {f_prev - bounce_frame} frames from bounce\n")
            return True
        return False

    def _check_parabolic_deviation(self, bounce_frame: int) -> bool:
        """
        Parabolic-residual deviation detector.

        Fit a parabola to the first _PARABOLA_FIT_POINTS post-bounce
        detections. Extrapolate forward. Flag the frame where actual
        position diverges from the extrapolation by > _RESIDUAL_THRESHOLD_PX
        for _RESIDUAL_CONFIRM_FRAMES consecutive detections. Returns True if
        a deviation was recorded, False otherwise.
        """
        ordered = sorted(self._pts, key=lambda p: p[0])
        post_bounce = [p for p in ordered if p[0] >= bounce_frame]
        fit_n = self._PARABOLA_FIT_POINTS
        # Need at least fit + confirm frames of data to even think about firing.
        if len(post_bounce) < fit_n + self._RESIDUAL_CONFIRM_FRAMES:
            return False

        # Fit parabola to the first fit_n stable post-bounce points.
        fit_pts = post_bounce[:fit_n]
        ts = np.array([p[0] for p in fit_pts], dtype=float)
        xs = np.array([p[1] for p in fit_pts], dtype=float)
        ys = np.array([p[2] for p in fit_pts], dtype=float)
        if len(np.unique(ts)) < 3:
            return False
        # Quadratic (parabolic) fit for both x(t) and y(t). x is typically
        # near-linear post-bounce; y is genuinely quadratic (gravity).
        try:
            cx_coef = np.polyfit(ts, xs, 2)  # [a, b, c]
            cy_coef = np.polyfit(ts, ys, 2)
        except (np.linalg.LinAlgError, ValueError):
            return False

        # Sanity check: fit quality on the fit points themselves. If the
        # parabola doesn't even fit its training data well, it's not a good
        # basis for extrapolation.
        pred_xs_fit = np.polyval(cx_coef, ts)
        pred_ys_fit = np.polyval(cy_coef, ts)
        fit_rms = float(np.sqrt(np.mean(
            (pred_xs_fit - xs) ** 2 + (pred_ys_fit - ys) ** 2
        )))
        if fit_rms > 12.0:  # px — trajectory is too noisy for a clean fit
            return False

        # Evaluate residuals on the post-fit tail — the frames we haven't seen
        # yet during the fit. This is where the bat impact would show up.
        tail = post_bounce[fit_n:]
        consecutive_over = 0
        first_over_idx = -1
        for i, (f, x, y) in enumerate(tail):
            pred_x = float(np.polyval(cx_coef, f))
            pred_y = float(np.polyval(cy_coef, f))
            residual = ((x - pred_x) ** 2 + (y - pred_y) ** 2) ** 0.5
            if residual >= self._RESIDUAL_THRESHOLD_PX:
                if consecutive_over == 0:
                    first_over_idx = i
                consecutive_over += 1
                if consecutive_over >= self._RESIDUAL_CONFIRM_FRAMES:
                    # Mark the first frame that broke the parabola — that's
                    # the impact frame, not the confirmation frame.
                    f_dev, x_dev, y_dev = tail[first_over_idx]
                    self._dir_changes.append((f_dev, x_dev, y_dev))
                    print("\n[DEVIATION POINT — parabolic residual]")
                    print(f"  Frame       : {f_dev}  "
                          f"(confirmed at F{tail[i][0]})")
                    print(f"  Coordinates : x={x_dev}, y={y_dev}")
                    print(f"  Residual    : {residual:.1f} px "
                          f"(threshold {self._RESIDUAL_THRESHOLD_PX})")
                    print(f"  Fit RMS     : {fit_rms:.2f} px "
                          f"(over {fit_n} points)\n")
                    return True
            else:
                consecutive_over = 0
                first_over_idx = -1
        return False

    def _check_direction_change(
        self,
        release_frame: Optional[int],
        bounce_frame:  Optional[int],
    ) -> None:
        """
        Flag a marker when the ball's post-bounce trajectory diverges from
        its parabolic extrapolation (primary) or, as a fallback, when the
        predicted horizontal direction (from a linear fit) disagrees with
        the actual motion (legacy path). Gated on release; the bounce frame
        itself is skipped because vy flips there, not vx.
        """
        # Only the first deviation point is recorded.
        if self._dir_changes:
            return

        # Deviation events are only meaningful AFTER bounce — that's when
        # the ball can be deflected by bat/pad/stumps. Pre-bounce path
        # changes are dominated by release wobble and gravity curvature.
        if bounce_frame is None:
            return

        # Priority 1: gap-based detector — most physically meaningful signal
        # (ball goes behind bat, reappears elsewhere = impact).
        if self._USE_GAP_DEVIATION:
            if self._check_gap_deviation(bounce_frame):
                return

        # Priority 2: parabolic-residual — for clips where ball is tracked
        # continuously through impact (no gap to detect).
        if self._USE_PARABOLIC_DEVIATION:
            if self._check_parabolic_deviation(bounce_frame):
                return
            # If parabolic fit is even POSSIBLE (>= FIT_POINTS available), we
            # trust its "not yet triggered" verdict and don't fall through to
            # the last-seen fallback or noisy legacy detector.
            post_bounce_count = sum(1 for p in self._pts if p[0] >= bounce_frame)
            if post_bounce_count >= self._PARABOLA_FIT_POINTS:
                return

        # Priority 3 (last-seen fallback) is DELIBERATELY NOT called here.
        # It has to run on the FINALISED trajectory, not per-frame — during
        # streaming updates it fires on the first post-bounce frame just
        # because len(post_bounce) is momentarily 2, even though the ball is
        # still being tracked. Instead, process.py calls finalize_deviation()
        # once after all frames have been processed; if nothing else caught
        # the deviation, the last-seen check runs there against the full data.

        # If nothing above fired, fall through to the legacy linear-velocity
        # detector below — only ever kicks in for very short bounce-to-bat
        # windows with a smooth continuous trajectory.

        ordered = sorted(self._pts, key=lambda p: p[0])
        post_bounce = [p for p in ordered if p[0] >= bounce_frame]
        if len(post_bounce) < 4:
            return
        post_release = post_bounce

        # Use points strictly before the latest as "prediction context",
        # then compare predicted vx against actual vx into the latest point.
        *prior, latest = post_release
        if len(prior) < 3:
            return

        f_prev, x_prev, y_prev = prior[-1]
        f_now,  x_now,  y_now = latest
        df = max(f_now - f_prev, 1)
        vx_actual = (x_now - x_prev) / df
        vy_actual = (y_now - y_prev) / df

        # Predicted velocity from a linear fit on the last few prior points.
        window = prior[-5:] if len(prior) >= 5 else prior
        ts = np.array([p[0] for p in window], dtype=float)
        xs = np.array([p[1] for p in window], dtype=float)
        ys = np.array([p[2] for p in window], dtype=float)
        if len(np.unique(ts)) < 2:
            return
        slope_x, _ = np.polyfit(ts, xs, 1)
        slope_y, _ = np.polyfit(ts, ys, 1)
        vx_predicted = float(slope_x)
        vy_predicted = float(slope_y)

        # Skip frames immediately after bounce — vy reverses there and the
        # post-bounce fit needs a couple of points to stabilize.
        # NOTE: was `<= 2` originally; increased to `<= 6` (2026-09-15) because
        # the tight window was letting deviation fire on post-bounce noise
        # (ball settling for 2-3 frames right after bounce) instead of the real
        # bat-impact direction change. Revert to 2 if fast-delivery clips
        # (yorkers, short-of-length) start missing their deviation.
        if abs(f_now - bounce_frame) <= 6:
            return

        speed_pred = (vx_predicted ** 2 + vy_predicted ** 2) ** 0.5
        speed_act  = (vx_actual    ** 2 + vy_actual    ** 2) ** 0.5
        if speed_pred < self._MIN_SPEED or speed_act < self._MIN_SPEED:
            return

        # Angle between predicted and actual velocity vectors.
        cos_theta = (vx_predicted * vx_actual + vy_predicted * vy_actual) \
                    / (speed_pred * speed_act)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angle_deg = float(np.degrees(np.arccos(cos_theta)))

        if angle_deg >= self._DEVIATION_ANGLE_DEG:
            # Only the FIRST deviation point post-bounce is recorded —
            # subsequent bends (secondary contact, roll, etc.) are ignored.
            if not self._dir_changes:
                # Mark the frame 1 before the confirming frame — i.e. the last
                # pre-deflection ball position (prior[-1] == f_prev).
                f_dev, x_dev, y_dev = f_prev, x_prev, y_prev
                self._dir_changes.append((f_dev, x_dev, y_dev))
                print("\n[DEVIATION POINT]")
                print(f"  Frame      : {f_dev}  (confirmed at F{f_now})")
                print(f"  Coordinates: x={x_dev}, y={y_dev}")
                print(f"  Angle      : {angle_deg:.1f} deg")
                print(f"  v_predicted: ({vx_predicted:+.2f}, {vy_predicted:+.2f}) px/frame")
                print(f"  v_actual   : ({vx_actual:+.2f}, {vy_actual:+.2f}) px/frame\n")

    def recheck_deviations(self, release_frame: Optional[int], bounce_frame: Optional[int]) -> None:
        """Re-run the deviation check across all history points with the finalized bounce frame."""
        self._dir_changes.clear()
        if bounce_frame is None or release_frame is None:
            return

        ordered = sorted(self._pts, key=lambda p: p[0])
        original_pts = list(self._pts)
        self._pts.clear()
        for f, x, y in ordered:
            self._pts.append((f, x, y))
            self._check_direction_change(release_frame, bounce_frame)
        self._pts = original_pts

    def finalize_deviation(
        self,
        bounce_frame: Optional[int],
        bat_line_x: Optional[float] = None,
        bat_line_y: Optional[float] = None,
    ) -> None:
        """Run last-chance detectors on the fully-populated trajectory.

        Called by process.py once after all frames have been processed.
        `bat_line_x` / `bat_line_y` (optional): batter's stance coordinates
        (from calibration) — when provided, the last-seen extrapolation stops
        when the projected ball meets the batter's zone instead of running
        for a fixed number of frames. This produces a much better deviation
        location for clips where tracking dies well before the actual impact.
        """
        if self._dir_changes:
            return  # something else already caught it
        if bounce_frame is None:
            return
        if self._USE_LAST_SEEN_FALLBACK:
            self._check_last_seen_fallback(
                bounce_frame, bat_line_x=bat_line_x, bat_line_y=bat_line_y,
            )

    def draw_post_bounce_path(
        self,
        frame: np.ndarray,
        bounce_frame: int,
        up_to_frame: Optional[int] = None,
        color: tuple[int, int, int] = (255, 255, 255),   # white
        thickness: int = 6,
        progress: float = 1.0,
    ) -> np.ndarray:
        """
        Fallback when no deviation is detected: draw a polyline through
        every real post-bounce ball detection, in chronological order.

        `up_to_frame` clips the path so it grows progressively during Segment B.
        `progress` (0..1) further scales how many post-bounce segments are drawn,
        matching the parabola animation feel.
        """
        if bounce_frame is None:
            return frame

        ordered = sorted(self._pts, key=lambda p: p[0])
        post = [(x, y) for (f, x, y) in ordered if f >= bounce_frame]
        if up_to_frame is not None:
            post = [(x, y) for (f, x, y) in ordered
                    if bounce_frame <= f <= up_to_frame]
        if len(post) < 2:
            return frame

        # Progress scaling
        n_full = len(post) - 1
        n_draw = max(1, int(round(n_full * max(0.0, min(1.0, progress)))))
        pts = post[: n_draw + 1]
        pts_np = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)

        overlay = frame.copy()
        cv2.polylines(overlay, [pts_np], isClosed=False,
                      color=(0, 0, 0), thickness=thickness + 3,
                      lineType=cv2.LINE_AA)
        cv2.polylines(overlay, [pts_np], isClosed=False,
                      color=color, thickness=thickness,
                      lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        return frame

    def draw_direction_change_markers(
        self,
        frame: np.ndarray,
        frame_num: int,
        size: Optional[int] = None,
    ) -> np.ndarray:
        """Overlay the cricket ball image at every detected direction-change point."""
        from utils import overlay_ball_image
        from config import BOUNCE_MARKER_SIZE
        if size is None:
            size = BOUNCE_MARKER_SIZE
        for f, x, y in self._dir_changes:
            if frame_num < f:
                continue
            overlay_ball_image(frame, x, y, size)
        return frame

    def reset(self) -> None:
        self._pts.clear()
        self._dir_changes.clear()
