import { useRef, useState } from "react";
import { TrackingResult, ZONE_COLORS } from "./types";
import { PitchOverlay } from "./PitchOverlay";
import { PlayerMarker, VideoPlayer, VideoPlayerHandle } from "./VideoPlayer";
import { FeedbackModal } from "./FeedbackModal";
import { getFeedback } from "./api";
import { getClientId } from "../../lib/client-id";
import { DeliveryFeedback } from "./types";

const API_BASE = (import.meta.env?.VITE_API_BASE as string) || "/api";
const FB_KEY = (jobId: string) => `tracker_feedback_done:${jobId}`;

export function ResultsView({ result }: { result: TrackingResult }) {
  const playerRef = useRef<VideoPlayerHandle>(null);
  const [showStumps, setShowStumps] = useState(true);
  const [showPitchingLine, setShowPitchingLine] = useState(true);
  const [showBallMarkers, setShowBallMarkers] = useState(true);
  const [showParabola, setShowParabola] = useState(true);
  const [showPrediction, setShowPrediction] = useState(true);
  const [currentTime, setCurrentTime] = useState(0);
  const [downloading, setDownloading] = useState(false);
  const [showFeedback, setShowFeedback] = useState(false);
  const [feedbackInitial, setFeedbackInitial] = useState<DeliveryFeedback | null>(null);
  const [feedbackDone, setFeedbackDone] = useState(
    () => typeof localStorage !== "undefined" &&
      localStorage.getItem(FB_KEY(result.job_id)) === "1",
  );

  function handleVideoEnded() {
    if (!feedbackDone && !showFeedback) {
      setFeedbackInitial(null);
      setShowFeedback(true);
    }
  }
  function handleFeedbackSubmitted() {
    localStorage.setItem(FB_KEY(result.job_id), "1");
    setFeedbackDone(true);
    setShowFeedback(false);
  }
  async function openEdit() {
    try {
      const existing = await getFeedback(result.job_id);
      setFeedbackInitial(existing);
    } catch {
      setFeedbackInitial(null);
    }
    setShowFeedback(true);
  }
  const [downloadError, setDownloadError] = useState<string | null>(null);

  async function handleDownload() {
    if (downloading) return;
    setDownloading(true);
    setDownloadError(null);
    try {
      const res = await fetch(
        `${API_BASE}/track/jobs/${result.job_id}/download_full`,
        { headers: { "X-Client-Id": getClientId() } },
      );
      if (!res.ok) {
        let msg = `Download failed (${res.status})`;
        try {
          const body = await res.json();
          if (body?.detail) msg = typeof body.detail === "string" ? body.detail : msg;
        } catch { /* ignore */ }
        throw new Error(msg);
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${result.video.filename.replace(/\.[^.]+$/, "")}_annotated.mp4`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) {
      setDownloadError(e instanceof Error ? e.message : String(e));
    } finally {
      setDownloading(false);
    }
  }

  // Timeline markers should point at where each event APPEARS in the output
  // video, not at the source-video timestamp. Use segments.seg_a_* (event
  // positions inside SEG A's live playback) and fall back to source timestamps
  // for older jobs that lack the segments field.
  const segs = result.segments;
  const releaseT  = segs?.seg_a_release_sec   ?? result.events.release?.timestamp_sec;
  const bounceT   = segs?.seg_a_bounce_sec    ?? result.events.bounce?.timestamp_sec;
  const deviationT = segs?.seg_a_deviation_sec ?? result.events.deviation?.timestamp_sec;

  const markers: PlayerMarker[] = [];
  if (result.events.release && releaseT != null) {
    markers.push({
      label: "Release",
      time: releaseT,
      color: "var(--accent)",
    });
  }
  if (result.events.bounce && bounceT != null) {
    markers.push({
      label: "Bounce",
      time: bounceT,
      color: result.events.bounce.zone
        ? ZONE_COLORS[result.events.bounce.zone]
        : "var(--good)",
    });
  }
  if (result.events.deviation && deviationT != null) {
    markers.push({
      label: "Deviation",
      time: deviationT,
      color: "var(--accent)",
    });
  }

  return (
    <div>
      <div className="panel">
        <div className="overlay-toggles" style={{
          display: "flex", gap: 8, marginBottom: 10, flexWrap: "wrap",
        }}>
          <ToggleButton
            label="Stumps"
            on={showStumps}
            onClick={() => setShowStumps((v) => !v)}
          />
          <ToggleButton
            label="Pitching line"
            on={showPitchingLine}
            onClick={() => setShowPitchingLine((v) => !v)}
          />
          <ToggleButton
            label="Ball markers"
            on={showBallMarkers}
            onClick={() => setShowBallMarkers((v) => !v)}
          />
          <ToggleButton
            label="Parabola"
            on={showParabola}
            onClick={() => setShowParabola((v) => !v)}
          />
          <ToggleButton
            label="Predicted path"
            on={showPrediction}
            onClick={() => setShowPrediction((v) => !v)}
          />
        </div>
        <VideoPlayer
          ref={playerRef}
          src={result.outputs.annotated_video_url}
          poster={result.outputs.first_frame_url}
          fps={result.video.fps}
          markers={markers}
          onTime={setCurrentTime}
          onEnded={handleVideoEnded}
          overlay={
            <PitchOverlay
              result={result}
              showStumps={showStumps}
              showPitchingLine={showPitchingLine}
              showBallMarkers={showBallMarkers}
              showParabola={showParabola}
              showPrediction={showPrediction}
              currentTime={currentTime}
              ballType={result.ball_type}
            />
          }
        />
      </div>

      <EventCards result={result} />

      <div className="panel">
        <h3 style={{ marginTop: 0 }}>Download</h3>
        <button onClick={handleDownload} disabled={downloading}>
          {downloading
            ? "Preparing download… (first time only, ~10s)"
            : "Annotated video (.mp4)"}
        </button>
        {downloadError && (
          <div className="error" style={{ marginTop: 8 }}>{downloadError}</div>
        )}
      </div>

      <div className="panel" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div className="muted">
          Processed in {result.stats.processing_time_sec.toFixed(1)} seconds
        </div>
        {feedbackDone ? (
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span className="fb-done-badge">Feedback received · Thanks!</span>
            <button className="ghost" onClick={openEdit}>Edit</button>
          </div>
        ) : (
          <button onClick={() => { setFeedbackInitial(null); setShowFeedback(true); }}>Give feedback</button>
        )}
      </div>

      {showFeedback && (
        <FeedbackModal
          jobId={result.job_id}
          initial={feedbackInitial}
          onClose={() => setShowFeedback(false)}
          onSubmitted={handleFeedbackSubmitted}
        />
      )}
    </div>
  );
}

function ToggleButton({
  label, on, onClick,
}: { label: string; on: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      aria-pressed={on}
      style={{
        padding: "6px 14px",
        borderRadius: 999,
        border: `1px solid ${on ? "var(--accent)" : "var(--border, #444)"}`,
        background: on ? "var(--accent)" : "transparent",
        color: on ? "#fff" : "var(--fg, #ddd)",
        cursor: "pointer",
        fontSize: 13,
        fontWeight: 600,
        transition: "all 0.15s",
      }}
    >
      {label}: {on ? "On" : "Off"}
    </button>
  );
}

function EventCards({ result }: { result: TrackingResult }) {
  const { release, bounce, deviation } = result.events;
  return (
    <div className="cards-row">
      <div className={`event-card ${release ? "" : "null"}`}>
        <h3>Release</h3>
        {release ? (
          <>
            <div className="value">{release.timestamp_sec.toFixed(2)} s</div>
            <div className="muted">Frame {release.frame}</div>
          </>
        ) : (
          <div className="muted">
            No release detected — the ball was never tracked for 4 consecutive
            frames near the start.
          </div>
        )}
      </div>

      <div className={`event-card ${bounce ? "" : "null"}`}>
        <h3>Bounce</h3>
        {bounce ? (
          <>
            <div className="value">{bounce.timestamp_sec.toFixed(2)} s</div>
            <div className="muted">Frame {bounce.frame}</div>
            <div style={{ marginTop: 8 }}>
              {bounce.zone ? (
                <span
                  className={`zone-badge ${bounce.zone}`}
                  style={{ background: ZONE_COLORS[bounce.zone] }}
                >
                  {bounce.zone}
                </span>
              ) : (
                <span className="muted">Bounced outside the marked pitch area.</span>
              )}
              {bounce.length_m != null && (
                <span className="muted" style={{ marginLeft: 10 }}>
                  {bounce.length_m.toFixed(2)} m from the stumps
                </span>
              )}
              {bounce.interpolated && (
                <span className="muted" style={{ marginLeft: 10 }}>(estimated)</span>
              )}
            </div>
          </>
        ) : (
          <div className="muted">No bounce point detected.</div>
        )}
      </div>

      <div className={`event-card ${deviation ? "" : "null"}`}>
        <h3>Deviation</h3>
        {deviation ? (
          <>
            <div className="value">{deviation.timestamp_sec.toFixed(2)} s</div>
            <div className="muted">Frame {deviation.frame}</div>
          </>
        ) : (
          <div className="muted">No deviation point detected.</div>
        )}
      </div>
    </div>
  );
}
