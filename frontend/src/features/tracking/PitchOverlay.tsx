import { Calibration, TrackingResult } from "./types";

interface Props {
  result: TrackingResult;
  showStumps: boolean;
  showPitchingLine: boolean;
  showBallMarkers: boolean;
  showParabola: boolean;
  /** Dotted "would-have-hit" line — extends bounce→deviation past the bat
   *  toward the batter's stumps, showing where the ball would have gone if
   *  the batsman had not made contact. */
  showPrediction: boolean;
  /** Current playback time in seconds — markers / parabola animate as it advances. */
  currentTime: number;
  /** Which cricket_ball.png to use ('white' default, 'red' for red-ball jobs). */
  ballType?: "white" | "red" | "pink";
}

// Real-world stump geometry — mirrors backend pipeline/process.py
const STUMP_DIAM_M     = 0.03493;
const STUMP_GAP_M      = 0.054;
const STUMP_PITCH_M    = STUMP_DIAM_M + STUMP_GAP_M;          // 88.93 mm
const STUMP_SET_WIDTH  = 3 * STUMP_DIAM_M + 2 * STUMP_GAP_M;  // 212.79 mm
const STUMP_HEIGHT_M   = 0.65;   // trimmed from 0.71 per user — felt too tall
const RETURN_CREASE_M  = 2.64;
const STUMP_OPACITY    = 0.70;   // 30% transparent

const STUMP_LIGHT   = "#f8f5f0";
const STUMP_DARK    = "#c8ccd2";
const STUMP_OUTLINE = "#6e7882";

// SEG C bake used cricket_ball.png at 17 px. User bumped +4 for visibility.
const BALL_MARKER_SIZE_PX = 21;
const N_PARABOLA_SAMPLES = 80;

function dist(a: [number, number], b: [number, number]): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  return Math.sqrt(dx * dx + dy * dy);
}

// Bump `?v=` when the PNGs are regenerated — browsers cache images aggressively
// and won't refetch even on a hard reload without a changed URL.
const BALL_ASSET_VERSION = "4";
function ballHref(ballType: "white" | "red" | "pink"): string {
  if (ballType === "red")  return `/cricket_ball_red.png?v=${BALL_ASSET_VERSION}`;
  if (ballType === "pink") return `/cricket_ball_pink.png?v=${BALL_ASSET_VERSION}`;
  return `/cricket_ball.png?v=${BALL_ASSET_VERSION}`;
}

// Trajectory-line color per ball type — a lighter shade of the ball itself so
// the trail reads as "same ball, but the path" rather than an unrelated overlay.
// Kept in sync with backend/pipeline/bake_overlay.py _TRAJECTORY_COLOR_BGR.
const TRAJECTORY_COLOR: Record<"white" | "red" | "pink", string> = {
  white: "#e6ecf4",  // soft cool off-white / silver
  red:   "#ff8a70",  // light coral
  pink:  "#ffc9dc",  // very light pink
};

// ─────────────────────────────────────────────────────────────────────────────
// Stumps + pitching line
// ─────────────────────────────────────────────────────────────────────────────
function StumpSet({ base, creaseW }: { base: [number, number]; creaseW: number }) {
  if (creaseW <= 0) return null;
  const pxPerM = creaseW / RETURN_CREASE_M;
  const halfW  = STUMP_PITCH_M * pxPerM;
  const height = STUMP_HEIGHT_M * pxPerM;
  const diam   = Math.max(2, STUMP_DIAM_M * pxPerM);
  const [cx, cy] = base;
  const top = cy - height;
  const xs = [cx - halfW, cx, cx + halfW];
  const bailThick = Math.max(2, diam * 0.7);
  const bailY = top;  // bails seated at the very top of the stump shafts

  return (
    <g opacity={STUMP_OPACITY}>
      {xs.map((x, i) => (
        <g key={i}>
          <line x1={x} y1={cy} x2={x} y2={top}
                stroke={STUMP_OUTLINE} strokeWidth={diam + 1.5}
                strokeLinecap="round" />
          <line x1={x} y1={cy} x2={x} y2={top}
                stroke={STUMP_LIGHT} strokeWidth={diam}
                strokeLinecap="round" />
          <line x1={x + diam * 0.22} y1={cy} x2={x + diam * 0.22} y2={top}
                stroke={STUMP_DARK} strokeWidth={Math.max(1, diam * 0.32)} />
        </g>
      ))}
      <line x1={xs[0]} y1={bailY} x2={xs[1]} y2={bailY}
            stroke={STUMP_OUTLINE} strokeWidth={bailThick + 1}
            strokeLinecap="round" />
      <line x1={xs[0]} y1={bailY} x2={xs[1]} y2={bailY}
            stroke={STUMP_DARK} strokeWidth={bailThick}
            strokeLinecap="round" />
      <line x1={xs[1]} y1={bailY} x2={xs[2]} y2={bailY}
            stroke={STUMP_OUTLINE} strokeWidth={bailThick + 1}
            strokeLinecap="round" />
      <line x1={xs[1]} y1={bailY} x2={xs[2]} y2={bailY}
            stroke={STUMP_DARK} strokeWidth={bailThick}
            strokeLinecap="round" />
    </g>
  );
}

function PitchingLine({ calibration }: { calibration: Calibration }) {
  const bowl_stump = calibration.bowl_stump;
  const bat_stump  = calibration.bat_stump;
  const bowl_w = dist(calibration.bowl_L, calibration.bowl_R);
  const bat_w  = dist(calibration.bat_L,  calibration.bat_R);
  if (bowl_w <= 0 || bat_w <= 0) return null;

  const dx = bat_stump[0] - bowl_stump[0];
  const dy = bat_stump[1] - bowl_stump[1];
  const len = Math.sqrt(dx * dx + dy * dy);
  if (len < 1) return null;
  const ux = dx / len, uy = dy / len;
  const px = -uy, py = ux;

  const halfBowl = (STUMP_SET_WIDTH / 2) * (bowl_w / RETURN_CREASE_M);
  const halfBat  = (STUMP_SET_WIDTH / 2) * (bat_w  / RETURN_CREASE_M);

  const bL: [number, number] = [bowl_stump[0] - px * halfBowl, bowl_stump[1] - py * halfBowl];
  const aL: [number, number] = [bat_stump[0]  - px * halfBat,  bat_stump[1]  - py * halfBat];
  const aR: [number, number] = [bat_stump[0]  + px * halfBat,  bat_stump[1]  + py * halfBat];
  const bR: [number, number] = [bowl_stump[0] + px * halfBowl, bowl_stump[1] + py * halfBowl];
  const points = [bL, aL, aR, bR].map((p) => `${p[0]},${p[1]}`).join(" ");
  return <polygon points={points} fill="#ffffff" fillOpacity={0.30} />;
}

// ─────────────────────────────────────────────────────────────────────────────
// Cricket-ball PNG marker (alpha keyed from luminance via SVG filter in <defs>)
// ─────────────────────────────────────────────────────────────────────────────
function BallMarker({ x, y, size, href }: {
  x: number; y: number; size: number; href: string;
}) {
  const half = size / 2;
  return (
    <image
      href={href}
      x={x - half} y={y - half}
      width={size} height={size}
      preserveAspectRatio="xMidYMid meet"
      filter="url(#ball-alpha-from-luma)"
    />
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Segment routing — single source of truth for which segment `currentTime`
// sits in. Used by Parabolas + BallMarkers to gate visibility/animation.
// ─────────────────────────────────────────────────────────────────────────────
type Phase = "before" | "segA" | "segB" | "segC" | "after";

function currentPhase(t: number, segs: TrackingResult["segments"]): Phase {
  if (t < segs.seg_a_start_sec) return "before";
  if (t < segs.seg_a_end_sec) return "segA";
  if (t < segs.seg_b_end_sec) return "segB";
  if (segs.seg_c_end_sec != null && t < segs.seg_c_end_sec) return "segC";
  return "after";
}

interface Progress { rb: number; bd: number; }

function parabolaProgress(
  t: number, segs: TrackingResult["segments"],
): Progress {
  const phase = currentPhase(t, segs);
  if (phase === "before" || phase === "segA") return { rb: 0, bd: 0 };

  if (phase === "segB") {
    const r = segs.seg_b_release_sec, b = segs.seg_b_bounce_sec, d = segs.seg_b_deviation_sec;
    if (r == null || b == null) return { rb: 0, bd: 0 };
    if (t < r) return { rb: 0, bd: 0 };
    if (t < b) return { rb: (t - r) / Math.max(1e-6, b - r), bd: 0 };
    if (d == null) return { rb: 1, bd: 0 };
    if (t < d) return { rb: 1, bd: (t - b) / Math.max(1e-6, d - b) };
    return { rb: 1, bd: 1 };
  }

  if (phase === "segC") {
    const s = segs.seg_c_start_sec, rEnd = segs.seg_c_rb_end_sec, bEnd = segs.seg_c_bd_end_sec;
    if (s == null || rEnd == null || bEnd == null) return { rb: 1, bd: 1 };
    if (t < rEnd) return { rb: (t - s) / Math.max(1e-6, rEnd - s), bd: 0 };
    if (t < bEnd) return { rb: 1, bd: (t - rEnd) / Math.max(1e-6, bEnd - rEnd) };
    return { rb: 1, bd: 1 };  // hold
  }

  return { rb: 1, bd: 1 };  // after — parabola stays fully drawn
}

// ─────────────────────────────────────────────────────────────────────────────
// Parabolas + traveling ball (SEG B and SEG C)
// ─────────────────────────────────────────────────────────────────────────────
function fitConstrainedQuadratic(
  pts: { frame: number; x: number; y: number }[],
  x1: number, y1: number, x2: number, y2: number,
  f_lo: number, f_hi: number,
): { kx: number; ky: number } {
  if (pts.length < 1 || f_hi <= f_lo) return { kx: 0, ky: 0 };
  let sumAA = 0, sumABx = 0, sumABy = 0;
  for (const p of pts) {
    const t = (p.frame - f_lo) / (f_hi - f_lo);
    const a = t * (1 - t);
    sumAA  += a * a;
    sumABx += a * (p.x - ((1 - t) * x1 + t * x2));
    sumABy += a * (p.y - ((1 - t) * y1 + t * y2));
  }
  if (sumAA < 1e-9) return { kx: 0, ky: 0 };
  return { kx: sumABx / sumAA, ky: sumABy / sumAA };
}

function pointOnQuad(
  x1: number, y1: number, x2: number, y2: number,
  kx: number, ky: number, t: number,
): { x: number; y: number } {
  const a = t * (1 - t);
  return {
    x: (1 - t) * x1 + t * x2 + kx * a,
    y: (1 - t) * y1 + t * y2 + ky * a,
  };
}

function Parabolas({
  result, currentTime, ballType,
}: { result: TrackingResult; currentTime: number; ballType: "white" | "red" | "pink" }) {
  const { release, bounce, deviation } = result.events;
  if (!release || !bounce) return null;

  const phase = currentPhase(currentTime, result.segments);
  if (phase === "before" || phase === "segA") return null;  // SEG A: no parabola

  const { rb, bd } = parabolaProgress(currentTime, result.segments);

  // ── release → bounce: constrained quadratic ──────────────────────────────
  const x1 = release.x, y1 = release.y, x2 = bounce.x, y2 = bounce.y;
  const f_lo = release.frame, f_hi = bounce.frame;
  const rbPts = result.trajectory.filter(
    (p) => p.frame >= f_lo && p.frame <= f_hi,
  );
  const { kx, ky } = fitConstrainedQuadratic(rbPts, x1, y1, x2, y2, f_lo, f_hi);

  let rbPath = "";
  if (rb > 0) {
    const n = Math.max(2, Math.round(N_PARABOLA_SAMPLES * rb));
    const pts: string[] = [];
    for (let i = 0; i < n; i++) {
      const t = (i / (n - 1)) * rb;
      const p = pointOnQuad(x1, y1, x2, y2, kx, ky, t);
      pts.push(`${p.x.toFixed(2)},${p.y.toFixed(2)}`);
    }
    rbPath = pts.join(" ");
  }

  // ── bounce → deviation: straight line (legacy passes arc_ratio=0) ────────
  let bdPath = "";
  if (deviation && bd > 0) {
    const ex = bounce.x + bd * (deviation.x - bounce.x);
    const ey = bounce.y + bd * (deviation.y - bounce.y);
    bdPath = `${bounce.x},${bounce.y} ${ex.toFixed(2)},${ey.toFixed(2)}`;
  }

  // ── Traveling cricket ball leading the sweep — SEG C only per user spec ──
  let ballPos: { x: number; y: number } | null = null;
  if (phase === "segC") {
    if (rb > 0 && rb < 1) {
      ballPos = pointOnQuad(x1, y1, x2, y2, kx, ky, rb);
    } else if (deviation && rb >= 1 && bd > 0 && bd < 1) {
      ballPos = {
        x: bounce.x + bd * (deviation.x - bounce.x),
        y: bounce.y + bd * (deviation.y - bounce.y),
      };
    }
  }

  const trailColor = TRAJECTORY_COLOR[ballType];
  return (
    <g>
      {rbPath && (
        <polyline points={rbPath} fill="none"
                  stroke={trailColor} strokeWidth={16}
                  strokeOpacity={0.40}
                  strokeLinecap="round" strokeLinejoin="round" />
      )}
      {bdPath && (
        <polyline points={bdPath} fill="none"
                  stroke={trailColor} strokeWidth={13}
                  strokeOpacity={0.70}
                  strokeLinecap="round" strokeLinejoin="round" />
      )}
      {ballPos && (
        <BallMarker x={ballPos.x} y={ballPos.y}
                    size={BALL_MARKER_SIZE_PX} href={ballHref(ballType)} />
      )}
    </g>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Predicted trajectory — LBW-style "would-have-hit" projection past the bat.
//
// Geometry: the direction from bounce → deviation is the ball's actual
// post-bounce path in screen space. To predict where the ball would have
// gone, continue that same direction past deviation until it reaches the
// batter's STUMP PLANE — the horizontal screen band at bail height above
// bat_stump — rather than a fixed multiplier. The stump plane's screen
// position is derived from the calibration (same pxPerM scale used by
// StumpSet), so the endpoint is tied to the actual stumps drawn in the
// overlay, not an arbitrary length.
//
// Hit/miss verdict: if the endpoint's x-coordinate lands within the stump
// set's screen half-width around bat_stump.x, the ball would have hit — the
// endpoint indicator turns red. Otherwise green (would have missed).
//
// Fallback: if the deviation point is already above the target stump-top y
// (ball delivered high — bouncer, or hit on the way up), a short forward
// projection is drawn instead so the intent is still visible.
//
// Only revealed once the animation has drawn the bounce→deviation segment
// (bd == 1), so it appears at the moment of contact.
// ─────────────────────────────────────────────────────────────────────────────
const HIT_COLOR  = "#ff3b30";   // red — would have hit stumps
const MISS_COLOR = "#8DFF3F";   // green — would have missed

function PredictedPath({
  result, currentTime,
}: { result: TrackingResult; currentTime: number }) {
  const { bounce, deviation } = result.events;
  if (!bounce || !deviation) return null;

  const phase = currentPhase(currentTime, result.segments);
  if (phase === "before" || phase === "segA") return null;

  const { bd } = parabolaProgress(currentTime, result.segments);
  if (bd < 1) return null;

  const dx = deviation.x - bounce.x;
  const dy = deviation.y - bounce.y;
  if (Math.abs(dx) < 1e-3 && Math.abs(dy) < 1e-3) return null;

  // Stump-plane screen geometry, matching StumpSet's calculation exactly so
  // the projection lines up with the drawn stumps.
  const cal = result.calibration;
  const bat_w = dist(cal.bat_L, cal.bat_R);
  if (bat_w <= 0) return null;
  const pxPerM = bat_w / RETURN_CREASE_M;
  const stumpHeightPx     = STUMP_HEIGHT_M * pxPerM;
  const halfStumpWidthPx  = (STUMP_SET_WIDTH / 2) * pxPerM;
  const batStump = cal.bat_stump;

  // Target: top of the drawn stumps (bail height). Camera behind bowler →
  // batter's stumps at the TOP of frame → smaller screen-y is farther.
  const targetY = batStump[1] - stumpHeightPx;

  // Step 1 — locate the STUMP-PLANE CROSSING along the extended line. This is
  // the point that determines hit vs miss. Requires the direction to actually
  // carry the ball toward the target (positive t forward from deviation).
  let hitX: number | null = null, hitY: number | null = null;
  if (Math.abs(dy) > 1e-3) {
    const t = (targetY - deviation.y) / dy;
    if (t > 0.02) {
      hitX = deviation.x + dx * t;
      hitY = targetY;
    }
  }

  // Step 2 — pick a VISUAL end for the dashed line so it reads as a
  // trajectory rather than a pixel. The stump-plane crossing is often only a
  // few px past deviation (ball is near bail height at contact), so we
  // guarantee at least MIN_LINE_LEN of screen length forward. If the ball is
  // moving mostly parallel to the stump plane (hit crossing near infinity),
  // fall back to a fraction of segLen.
  const segLen = Math.sqrt(dx * dx + dy * dy);
  const ux = dx / segLen, uy = dy / segLen;
  const MIN_LINE_LEN = Math.max(60, segLen * 0.35);
  const distToStumpPlane = hitX != null && hitY != null
    ? Math.sqrt((hitX - deviation.x) ** 2 + (hitY - deviation.y) ** 2)
    : 0;
  const lineLen = Math.max(distToStumpPlane, MIN_LINE_LEN);

  // Clamp to frame edge so we never draw off-canvas.
  const { width: W, height: H } = result.video;
  const clampToEdge = (num: number, den: number): number => {
    if (Math.abs(den) < 1e-9) return Infinity;
    const t = num / den;
    return t > 0 ? t : Infinity;
  };
  const maxLen = Math.min(
    lineLen,
    clampToEdge(0 - deviation.x, ux),
    clampToEdge(W - deviation.x, ux),
    clampToEdge(0 - deviation.y, uy),
    clampToEdge(H - deviation.y, uy),
  );
  const endX = deviation.x + ux * maxLen;
  const endY = deviation.y + uy * maxLen;

  // Verdict is based on the STUMP-PLANE crossing, not the drawn line end.
  const wouldHitStumps =
    hitX != null && Math.abs(hitX - batStump[0]) <= halfStumpWidthPx;
  const indicatorFill = wouldHitStumps ? HIT_COLOR : MISS_COLOR;

  // Straight line — over the short bat→stumps distance (~1 m real-world),
  // gravity's effect on the ball is negligible, so a straight extrapolation
  // is the honest visualization. White dashes with a black border: two
  // overlaid lines share the same dasharray + round linecap, so the thicker
  // black stroke shows around each white dash as a halo.
  return (
    <g>
      <line
        x1={deviation.x} y1={deviation.y} x2={endX} y2={endY}
        stroke="#000" strokeWidth={7} strokeOpacity={0.9}
        strokeDasharray="10 8" strokeLinecap="round"
      />
      <line
        x1={deviation.x} y1={deviation.y} x2={endX} y2={endY}
        stroke="#fff" strokeWidth={4.5} strokeOpacity={1}
        strokeDasharray="10 8" strokeLinecap="round"
      />
      {hitX != null && hitY != null && (
        <circle cx={hitX} cy={hitY} r={9}
                fill={indicatorFill} stroke="#000" strokeWidth={2.5} />
      )}
    </g>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Event markers (release / bounce / deviation)
//   SEG A: progressive reveal at each event's source-frame timestamp.
//   SEG B: all three visible throughout.
//   SEG C: none during the parabola sweep; bounce re-appears once the sweep
//          completes (the "mark the bounce point again" hold).
// ─────────────────────────────────────────────────────────────────────────────
function BallMarkers({
  result, currentTime, ballType,
}: { result: TrackingResult; currentTime: number; ballType: "white" | "red" | "pink" }) {
  // For red and pink pipelines, release + bounce + deviation are baked
  // directly into output.mp4 (see backend/pipeline/process.py
  // _draw_progressive_overlays). Skip the SVG overlay markers so we don't
  // double-draw on top of the baked ones. White ball still uses SVG markers
  // (it doesn't bake them for backwards-compatibility reasons).
  if (ballType === "red" || ballType === "pink") return null;

  const { release, bounce, deviation } = result.events;
  const segs = result.segments;
  const phase = currentPhase(currentTime, segs);
  const href = ballHref(ballType);
  const size = BALL_MARKER_SIZE_PX;

  if (phase === "segA") {
    const showR = release && segs.seg_a_release_sec   != null && currentTime >= segs.seg_a_release_sec;
    const showB = bounce  && segs.seg_a_bounce_sec    != null && currentTime >= segs.seg_a_bounce_sec;
    const showD = deviation && segs.seg_a_deviation_sec != null && currentTime >= segs.seg_a_deviation_sec;
    return (
      <g>
        {showR && release   && <BallMarker x={release.x}   y={release.y}   size={size} href={href} />}
        {showB && bounce    && <BallMarker x={bounce.x}    y={bounce.y}    size={size} href={href} />}
        {showD && deviation && <BallMarker x={deviation.x} y={deviation.y} size={size} href={href} />}
      </g>
    );
  }

  if (phase === "segB") {
    return (
      <g>
        {release   && <BallMarker x={release.x}   y={release.y}   size={size} href={href} />}
        {bounce    && <BallMarker x={bounce.x}    y={bounce.y}    size={size} href={href} />}
        {deviation && <BallMarker x={deviation.x} y={deviation.y} size={size} href={href} />}
      </g>
    );
  }

  if (phase === "segC") {
    // Only the bounce mark, only during the 1 s hold after the sweep ends.
    if (segs.seg_c_bd_end_sec != null && currentTime >= segs.seg_c_bd_end_sec && bounce) {
      return <BallMarker x={bounce.x} y={bounce.y} size={size} href={href} />;
    }
    return null;
  }

  return null;
}

// ─────────────────────────────────────────────────────────────────────────────
export function PitchOverlay({
  result, showStumps, showPitchingLine, showBallMarkers, showParabola,
  showPrediction, currentTime, ballType = "white",
}: Props) {
  const { width, height } = result.video;
  const cal = result.calibration;
  if (!cal.bowl_stump || !cal.bat_stump) return null;

  const bowl_w = dist(cal.bowl_L, cal.bowl_R);
  const bat_w  = dist(cal.bat_L,  cal.bat_R);

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
      style={{
        position: "absolute", inset: 0,
        width: "100%", height: "100%",
        pointerEvents: "none",
      }}
    >
      <defs>
        {/* Strip the cricket_ball.png black background: luminance → alpha,
            then amplify so any non-near-black pixel goes fully opaque
            (matches the backend's cv2.threshold(gray, 20, 255) mask). */}
        <filter id="ball-alpha-from-luma" x="0" y="0" width="100%" height="100%">
          <feColorMatrix type="matrix" values="
            1 0 0 0 0
            0 1 0 0 0
            0 0 1 0 0
            0.2125 0.7154 0.0722 0 0" />
          <feComponentTransfer>
            <feFuncA type="linear" slope="20" intercept="0" />
          </feComponentTransfer>
        </filter>
      </defs>
      {/* Layer order (bottom → top):
          1. Batter-end stumps  (behind everything else per user)
          2. Parabola + traveling ball (above batter stumps per user)
          3. Pitching line
          4. Bowler-end stumps
          5. Event markers (top) */}
      {showStumps && <StumpSet base={cal.bat_stump} creaseW={bat_w} />}
      {showParabola && (
        <Parabolas result={result} currentTime={currentTime} ballType={ballType} />
      )}
      {showPitchingLine && <PitchingLine calibration={cal} />}
      {showStumps && <StumpSet base={cal.bowl_stump} creaseW={bowl_w} />}
      {showBallMarkers && (
        <BallMarkers result={result} currentTime={currentTime} ballType={ballType} />
      )}
      {showPrediction && (
        <PredictedPath result={result} currentTime={currentTime} />
      )}
    </svg>
  );
}
