import { useEffect, useRef, useState } from "react";
import { getLiveFeed } from "./api";
import { LiveWicket } from "./types";

const POLL_MS = 5000;

const VERDICT_STYLE: Record<string, { bg: string; label: string }> = {
  OUT: { bg: "#ff3b30", label: "OUT" },
  "NOT OUT": { bg: "#2fa84a", label: "NOT OUT" },
};

function relTime(iso: string | null): string {
  if (!iso) return "";
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return "just now";
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} min ago`;
  const h = Math.floor(m / 60);
  return `${h} h ago`;
}

function VerdictBadge({ w, big }: { w: LiveWicket; big?: boolean }) {
  const pad = big ? "5px 14px" : "3px 10px";
  const fs = big ? 14 : 12;
  if (w.verdict) {
    const s = VERDICT_STYLE[w.verdict.decision] ?? { bg: "#6b7280", label: w.verdict.decision };
    return (
      <span style={{
        background: s.bg, color: "#fff", fontWeight: 800, fontSize: fs,
        padding: pad, borderRadius: 999, letterSpacing: 0.5,
        boxShadow: "0 2px 8px rgba(0,0,0,0.35)",
      }}>{s.label}</span>
    );
  }
  const label = w.state === "no_delivery_found" ? "no delivery"
    : w.state === "needs_calibration" ? "needs calibration"
    : w.state === "tracking_failed" ? "tracking failed"
    : "no verdict";
  return (
    <span style={{
      background: "#374151", color: "#cbd5e1", fontWeight: 700, fontSize: fs, padding: pad, borderRadius: 999,
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
      onKeyDown={(e) => { if (clickable && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); onPlay(w); } }}
      style={{
        background: "#141922", borderRadius: 14, overflow: "hidden",
        border: "1px solid rgba(255,255,255,0.07)", cursor: clickable ? "pointer" : "default",
        display: "flex", flexDirection: "column",
        boxShadow: "0 6px 20px rgba(0,0,0,0.25)", transition: "transform .12s ease",
      }}
      onMouseEnter={(e) => { if (clickable) e.currentTarget.style.transform = "translateY(-3px)"; }}
      onMouseLeave={(e) => { e.currentTarget.style.transform = "translateY(0)"; }}
    >
      <div style={{ position: "relative", aspectRatio: "16 / 9", background: "#0a0d14" }}>
        {w.thumbnail_url ? (
          <img src={w.thumbnail_url} alt="" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        ) : (
          <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: "#4b5563", fontSize: 13 }}>no clip</div>
        )}
        {clickable && (
          <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <div style={{ width: 48, height: 48, borderRadius: 999, background: "rgba(0,0,0,0.55)", border: "2px solid rgba(255,255,255,0.85)", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <span style={{ color: "#fff", fontSize: 18, marginLeft: 3 }}>▶</span>
            </div>
          </div>
        )}
        <div style={{ position: "absolute", top: 8, left: 8, background: "rgba(0,0,0,0.6)", color: "#fff", fontSize: 11, fontWeight: 700, padding: "2px 8px", borderRadius: 6 }}>
          Over {w.over}
        </div>
        <div style={{ position: "absolute", top: 8, right: 8 }}><VerdictBadge w={w} /></div>
      </div>
      <div style={{ padding: "11px 13px" }}>
        <div style={{ fontWeight: 700, fontSize: 15 }}>
          {w.batter ? w.batter : `Delivery ${w.delivery_id.slice(0, 6)}`}
          {w.bowler && <span style={{ color: "#8b94a3", fontWeight: 500 }}> &nbsp;b {w.bowler}</span>}
        </div>
        <div style={{ fontSize: 12, marginTop: 3, color: "#8b94a3" }}>
          {w.ball_type ? `${w.ball_type} ball` : ""}{w.ball_type && w.updated_at ? " · " : ""}{relTime(w.updated_at)}
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
  const hit = w.verdict?.would_hit_stumps;
  return (
    <div onClick={onClose} style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.82)", zIndex: 50, display: "flex", alignItems: "center", justifyContent: "center", padding: 24 }}>
      <div onClick={(e) => e.stopPropagation()} style={{ maxWidth: 1040, width: "100%" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10, gap: 12 }}>
          <div style={{ color: "#fff" }}>
            <div style={{ fontWeight: 800, fontSize: 17 }}>
              {w.batter || "Delivery"}{w.bowler ? ` b ${w.bowler}` : ""}
            </div>
            <div style={{ color: "#9aa3b2", fontSize: 13 }}>
              {w.match_label || ""}{w.match_label ? " · " : ""}Over {w.over} · Innings {w.innings}
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <VerdictBadge w={w} big />
            <button onClick={onClose} style={{ background: "transparent", color: "#fff", border: "1px solid rgba(255,255,255,0.3)", borderRadius: 8, padding: "6px 14px", cursor: "pointer", fontWeight: 600 }}>Close ✕</button>
          </div>
        </div>
        {w.video_url && (
          <video src={w.video_url} controls autoPlay style={{ width: "100%", borderRadius: 12, background: "#000", boxShadow: "0 10px 40px rgba(0,0,0,0.5)" }} />
        )}
        {w.verdict && (
          <div style={{ marginTop: 10, color: hit ? "#ff6b61" : "#5fd27a", fontSize: 14, fontWeight: 600 }}>
            {hit ? "● Projected to go on and hit the stumps." : "● Projected to miss the stumps."}
            <span style={{ color: "#6b7280", fontWeight: 400 }}> &nbsp;Tracked from release → bounce → bat contact.</span>
          </div>
        )}
      </div>
    </div>
  );
}

interface MatchInfo { id: string; label: string; venue: string | null; count: number; }

export function LiveFeed() {
  const [wickets, setWickets] = useState<LiveWicket[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState<LiveWicket | null>(null);
  const [selected, setSelected] = useState<string>("all");
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

  // Distinct matches (newest activity first), derived from the feed itself.
  const matches: MatchInfo[] = [];
  for (const w of wickets ?? []) {
    let m = matches.find((x) => x.id === w.match_id);
    if (!m) { m = { id: w.match_id, label: w.match_label || w.match_id, venue: w.venue ?? null, count: 0 }; matches.push(m); }
    if (w.has_media) m.count += 1;
  }

  const shown = (wickets ?? []).filter((w) => selected === "all" || w.match_id === selected);
  const sel = matches.find((m) => m.id === selected);
  const headerTitle = sel ? sel.label : matches.length > 1 ? "All Matches" : (matches[0]?.label || "Live Wickets");
  const headerSub = sel?.venue ? `${sel.venue} · auto-tracked wicket feed`
    : matches.length > 1 ? `${matches.length} matches live · auto-tracked wicket feed`
    : "auto-tracked wicket feed";
  const headerCount = shown.filter((w) => w.has_media).length;

  const Pill = ({ id, label, count }: { id: string; label: string; count: number }) => (
    <button onClick={() => setSelected(id)} style={{
      background: selected === id ? "#2563eb" : "#141922",
      color: selected === id ? "#fff" : "#9aa3b2",
      border: `1px solid ${selected === id ? "#2563eb" : "rgba(255,255,255,0.1)"}`,
      borderRadius: 999, padding: "7px 15px", fontSize: 13, fontWeight: 700, cursor: "pointer",
    }}>{label}<span style={{ opacity: 0.7, fontWeight: 500 }}> · {count}</span></button>
  );

  return (
    <div style={{ maxWidth: 1240, margin: "0 auto", padding: "22px 18px 48px" }}>
      <style>{`@keyframes livepulse {0%{opacity:1;transform:scale(1)}50%{opacity:.35;transform:scale(1.5)}100%{opacity:1;transform:scale(1)}}`}</style>

      {/* Match banner */}
      <div style={{
        background: "linear-gradient(135deg,#1c2433,#10151e)", borderRadius: 16, padding: "18px 22px",
        border: "1px solid rgba(255,255,255,0.07)", marginBottom: 16,
        display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12,
      }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ width: 9, height: 9, borderRadius: 999, background: "#ff3b30", display: "inline-block", animation: "livepulse 1.6s infinite" }} />
            <span style={{ color: "#ff6b61", fontWeight: 800, fontSize: 12, letterSpacing: 1.5 }}>LIVE</span>
          </div>
          <div style={{ fontSize: 26, fontWeight: 800, marginTop: 6 }}>{headerTitle}</div>
          <div style={{ color: "#8b94a3", fontSize: 13, marginTop: 2 }}>{headerSub}</div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div style={{ fontSize: 34, fontWeight: 800, lineHeight: 1 }}>{headerCount}</div>
          <div style={{ color: "#8b94a3", fontSize: 12, marginTop: 4 }}>wickets tracked · updates every {POLL_MS / 1000}s</div>
        </div>
      </div>

      {/* Match selector — only when more than one match is live */}
      {matches.length > 1 && (
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 20 }}>
          <Pill id="all" label="All matches" count={matches.reduce((s, m) => s + m.count, 0)} />
          {matches.map((m) => <Pill key={m.id} id={m.id} label={m.label} count={m.count} />)}
        </div>
      )}

      {error && (
        <div style={{ background: "#2a1416", border: "1px solid #7f1d1d", borderRadius: 10, padding: 14, color: "#fca5a5" }}>
          Couldn’t load the feed: {error}
        </div>
      )}
      {wickets === null && !error && <div style={{ color: "#8b94a3" }}>Loading…</div>}
      {wickets && wickets.length === 0 && (
        <div style={{ color: "#8b94a3" }}>No wickets yet. They’ll appear here automatically as they’re tracked.</div>
      )}

      {shown.length > 0 && (
        <div style={{ display: "grid", gap: 18, gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))" }}>
          {shown.map((w) => <WicketCard key={w.prefix} w={w} onPlay={setPlaying} />)}
        </div>
      )}

      {playing && <PlayerModal w={playing} onClose={() => setPlaying(null)} />}
    </div>
  );
}
