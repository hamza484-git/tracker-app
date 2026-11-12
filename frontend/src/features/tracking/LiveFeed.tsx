import { useEffect, useRef, useState } from "react";
import { getLiveFeed } from "./api";
import { LiveWicket } from "./types";

const POLL_MS = 5000;

const VERDICT_STYLE: Record<string, { bg: string; label: string }> = {
  OUT: { bg: "#ff3b30", label: "OUT" },
  "NOT OUT": { bg: "#2fa84a", label: "NOT OUT" },
};

function VerdictBadge({ w }: { w: LiveWicket }) {
  if (w.verdict) {
    const s = VERDICT_STYLE[w.verdict.decision] ?? { bg: "#6b7280", label: w.verdict.decision };
    return (
      <span style={{
        background: s.bg, color: "#fff", fontWeight: 800, fontSize: 12,
        padding: "3px 10px", borderRadius: 999, letterSpacing: 0.4,
      }}>{s.label}</span>
    );
  }
  const label = w.state === "no_delivery_found" ? "no delivery"
    : w.state === "needs_calibration" ? "needs calibration"
    : w.state === "tracking_failed" ? "tracking failed"
    : "no verdict";
  return (
    <span style={{
      background: "#374151", color: "#cbd5e1", fontWeight: 700, fontSize: 12,
      padding: "3px 10px", borderRadius: 999,
    }}>{label}</span>
  );
}

function WicketCard({ w, onPlay }: { w: LiveWicket; onPlay: (w: LiveWicket) => void }) {
  const clickable = w.has_media && !!w.video_url;
  return (
    <div
      onClick={() => clickable && onPlay(w)}
      role={clickable ? "button" : undefined}
      tabIndex={clickable ? 0 : undefined}
      onKeyDown={(e) => {
        if (clickable && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); onPlay(w); }
      }}
      style={{
        background: "var(--panel, #1a1f2b)", borderRadius: 12, overflow: "hidden",
        border: "1px solid rgba(255,255,255,0.06)", cursor: clickable ? "pointer" : "default",
        display: "flex", flexDirection: "column",
      }}
    >
      <div style={{ position: "relative", aspectRatio: "16 / 9", background: "#0a0d14" }}>
        {w.thumbnail_url ? (
          <img src={w.thumbnail_url} alt=""
               style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        ) : (
          <div style={{
            width: "100%", height: "100%", display: "flex", alignItems: "center",
            justifyContent: "center", color: "#4b5563", fontSize: 13,
          }}>no clip</div>
        )}
        {clickable && (
          <div style={{
            position: "absolute", inset: 0, display: "flex", alignItems: "center",
            justifyContent: "center",
          }}>
            <div style={{
              width: 46, height: 46, borderRadius: 999, background: "rgba(0,0,0,0.55)",
              display: "flex", alignItems: "center", justifyContent: "center",
            }}>
              <span style={{ color: "#fff", fontSize: 18, marginLeft: 3 }}>▶</span>
            </div>
          </div>
        )}
        <div style={{ position: "absolute", top: 8, right: 8 }}><VerdictBadge w={w} /></div>
      </div>
      <div style={{ padding: "10px 12px" }}>
        <div style={{ fontWeight: 700, fontSize: 14 }}>
          Over {w.over} · Innings {w.innings}
        </div>
        <div className="muted" style={{ fontSize: 12, marginTop: 2 }}>
          {w.ball_type ? `${w.ball_type} ball` : ""}
          {w.updated_at ? ` · ${new Date(w.updated_at).toLocaleTimeString()}` : ""}
        </div>
      </div>
    </div>
  );
}

function PlayerModal({ w, onClose }: { w: LiveWicket; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div onClick={onClose} style={{
      position: "fixed", inset: 0, background: "rgba(0,0,0,0.8)", zIndex: 50,
      display: "flex", alignItems: "center", justifyContent: "center", padding: 24,
    }}>
      <div onClick={(e) => e.stopPropagation()} style={{ maxWidth: 1000, width: "100%" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
          <div style={{ color: "#fff", fontWeight: 700 }}>
            Over {w.over} · Innings {w.innings} {w.verdict ? `· ${w.verdict.decision}` : ""}
          </div>
          <button onClick={onClose} style={{
            background: "transparent", color: "#fff", border: "1px solid rgba(255,255,255,0.3)",
            borderRadius: 8, padding: "4px 12px", cursor: "pointer",
          }}>Close ✕</button>
        </div>
        {w.video_url && (
          <video src={w.video_url} controls autoPlay
                 style={{ width: "100%", borderRadius: 10, background: "#000" }} />
        )}
      </div>
    </div>
  );
}

export function LiveFeed() {
  const [wickets, setWickets] = useState<LiveWicket[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState<LiveWicket | null>(null);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    let alive = true;
    const ctrl = new AbortController();
    const load = () => {
      getLiveFeed(50, ctrl.signal)
        .then((r) => { if (alive) { setWickets(r.wickets); setError(null); } })
        .catch((e) => { if (alive && e.name !== "AbortError") setError(e.message || "feed failed"); });
    };
    load();
    timer.current = window.setInterval(load, POLL_MS);
    return () => { alive = false; ctrl.abort(); if (timer.current) window.clearInterval(timer.current); };
  }, []);

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "24px 16px" }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 16 }}>
        <h1 style={{ margin: 0, fontSize: 22 }}>Live Wickets</h1>
        <span className="muted" style={{ fontSize: 13 }}>
          updates every {POLL_MS / 1000}s
        </span>
      </div>

      {error && (
        <div className="panel" style={{ borderColor: "#7f1d1d" }}>
          Couldn’t load the feed: {error}
        </div>
      )}

      {wickets === null && !error && <div className="muted">Loading…</div>}

      {wickets && wickets.length === 0 && (
        <div className="muted">No wickets yet. They’ll appear here as they’re tracked.</div>
      )}

      {wickets && wickets.length > 0 && (
        <div style={{
          display: "grid", gap: 16,
          gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))",
        }}>
          {wickets.map((w) => (
            <WicketCard key={w.prefix} w={w} onPlay={setPlaying} />
          ))}
        </div>
      )}

      {playing && <PlayerModal w={playing} onClose={() => setPlaying(null)} />}
    </div>
  );
}
