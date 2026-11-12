"""Wicket consumer — connects the stream clipper to the ball tracker.

The clipper (scorecard pipeline) drops a ~35 s window of the stream for every
wicket into the INCOMING folder:

    incoming/{match_id}/innings_{n}/over_{k}/{delivery_id}/window.mp4
    incoming/{match_id}/innings_{n}/over_{k}/{delivery_id}/meta.json   (written last)

For each one this service:
  1. finds the delivery inside the window          (pipeline/find_delivery.py)
  2. cuts the short raw delivery clip
  3. runs the existing tracker on it               (pipeline/worker.py — unchanged)
  4. stores the results under the same path in two places:

    auto-wicket-clips/{...}/delivery.mp4, meta.json            raw delivery
    auto-wicket-tracked/{...}/annotated.mp4, results.json,
                             thumbnail.jpg, status.json        tracker output

  5. deletes the window. It is never kept, except for a day under
     incoming/_unmatched/ when no delivery could be found in it.

Per-match inputs live next to the raw clips:

    auto-wicket-clips/{match_id}/match.json                 {"ball_type": "red"}
    auto-wicket-clips/{match_id}/calibration/{W}x{H}.json   the 4 crease corners

Locally the two "buckets" are plain folders; pointing them at the real GCS
buckets is the deployment step.

Run:
    python wicket_consumer.py --once          # process what is waiting, then exit
    python wicket_consumer.py                 # keep watching
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2

import gcs_sync
from ball_pipelines import BALL_PIPELINES
from pipeline.find_delivery import cut_clip, find_delivery

BACKEND_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("WICKET_WORK_DIR", BACKEND_DIR / "_tracker_work" / "wickets"))
INCOMING = Path(os.environ.get("WICKET_INCOMING_DIR", ROOT / "incoming"))
CLIPS = Path(os.environ.get("WICKET_CLIPS_DIR", ROOT / "auto-wicket-clips"))
TRACKED = Path(os.environ.get("WICKET_TRACKED_DIR", ROOT / "auto-wicket-tracked"))
JOBS = ROOT / "_jobs"

DEFAULT_BALL_TYPE = os.environ.get("WICKET_DEFAULT_BALL_TYPE", "white").strip().lower()
# How many delivery candidates to try through the tracker before giving up.
MAX_ATTEMPTS = int(os.environ.get("WICKET_MAX_ATTEMPTS", "2"))
UNMATCHED_KEEP_HOURS = float(os.environ.get("WICKET_UNMATCHED_KEEP_HOURS", "24"))

# Limits, so one bad or oversized clip can't tie the machine up.
MAX_WINDOW_SEC = float(os.environ.get("WICKET_MAX_WINDOW_SEC", "180"))
MAX_WINDOW_BYTES = int(float(os.environ.get("WICKET_MAX_WINDOW_MB", "1024")) * (1 << 20))
MAX_FRAME_SIDE = 4096
TRACKER_TIMEOUT_SEC = float(os.environ.get("WICKET_TRACKER_TIMEOUT_SEC", "1800"))

# Folder names become paths in both stores, so they are restricted to
# letters, digits, dot, dash and underscore (same rule as the clipper).
_SAFE_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


def _safe_parts(parts) -> bool:
    return len(parts) == 4 and all(_SAFE_PART.match(x) and ".." not in x for x in parts)


class NotReady(Exception):
    """The clip can't be processed yet (still being written); try again later."""


class Rejected(Exception):
    """The clip must not be processed at all; it is set aside."""


def log(msg: str, **kv) -> None:
    extra = " ".join(f"{k}={v}" for k, v in kv.items())
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg} {extra}".rstrip(), flush=True)


# Things already reported once (so a clip waiting on calibration doesn't log every pass).
_announced: set[str] = set()


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _status(prefix: str, state: str, **extra) -> None:
    _write_json(TRACKED / prefix / "status.json", {
        "state": state,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    })


# ── Per-match inputs ──────────────────────────────────────────────────────────
def _ball_type(match_id: str) -> str:
    cfg = CLIPS / match_id / "match.json"
    bt = DEFAULT_BALL_TYPE
    if cfg.exists():
        bt = str(json.loads(cfg.read_text(encoding="utf-8")).get("ball_type", bt)).lower()
    if bt not in BALL_PIPELINES:
        raise ValueError(f"unknown ball_type {bt!r} (expected one of {sorted(BALL_PIPELINES)})")
    return bt


def _calibration(match_id: str, window: Path) -> tuple[dict | None, Path]:
    """Calibration saved for this match at this video's resolution, or None.
    When missing, a frame is saved beside where the file belongs so the
    corners can be marked on it."""
    cap = cv2.VideoCapture(str(window))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, frame = cap.read()
    cap.release()
    if not ok or w <= 0 or h <= 0:
        raise ValueError(f"cannot read {window}")
    path = CLIPS / match_id / "calibration" / f"{w}x{h}.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            calib = data.get("calibration", data)
            _check_calibration(calib, w, h)
            return calib, path
        except (ValueError, KeyError, TypeError, AttributeError, IndexError) as e:
            # A broken calibration file is the operator's to fix — treat it
            # like a missing one rather than failing the clip.
            if str(path) not in _announced:
                _announced.add(str(path))
                log("calibration file is invalid", path=path, error=e)
            return None, path
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_jpg = path.with_name(f"{w}x{h}_frame.jpg")
    if not frame_jpg.exists():
        cv2.imwrite(str(frame_jpg), frame)
    return None, path


def _check_calibration(calib: dict, w: int, h: int) -> None:
    """Raise unless the four crease corners are present and inside the frame."""
    for key in ("bat_L", "bat_R", "bowl_L", "bowl_R"):
        x, y = float(calib[key][0]), float(calib[key][1])
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError(f"{key} is outside the {w}x{h} frame")
    if calib["bowl_L"][0] >= calib["bowl_R"][0] or calib["bat_L"][0] >= calib["bat_R"][0]:
        raise ValueError("left corner must be left of right corner")


# ── Running the existing tracker ──────────────────────────────────────────────
def _run_tracker(clip: Path, job_dir: Path, calibration: dict, ball_type: str,
                 filename: str) -> tuple[dict | None, dict | None]:
    """Run pipeline/worker.py on one clip, exactly as the web app does
    (jobs.run_job). Returns (results, error)."""
    cfg = BALL_PIPELINES[ball_type]
    job_dir.mkdir(parents=True, exist_ok=True)
    args_file = job_dir / "job_args.json"
    _write_json(args_file, {
        "video_path": str(clip),
        "output_dir": str(job_dir),
        "calibration": calibration,
        "weights_path": str(cfg["weights"]),
        "interp_method": "parabola",
        "job_id": job_dir.name,
        "video_filename": filename,
    })
    env = {
        **os.environ,
        "TRACKER_LEGACY_DIR": str(cfg["legacy_dir"]),
        "TRACKER_BALL_TYPE": ball_type,
        "TRACKER_BALL_ASSETS_DIR": os.environ.get(
            "TRACKER_BALL_ASSETS_DIR", str(BACKEND_DIR.parent / "frontend" / "public")),
    }
    try:
        proc = subprocess.run(
            [sys.executable, "-u", "-m", "pipeline.worker", str(args_file)],
            cwd=str(BACKEND_DIR), env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=TRACKER_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:        # run() has already killed it
        return None, {"code": "TIMEOUT",
                      "message": f"tracker did not finish within {TRACKER_TIMEOUT_SEC:.0f}s"}
    finally:
        args_file.unlink(missing_ok=True)
    if proc.returncode == 0 and (job_dir / "results.json").exists():
        return json.loads((job_dir / "results.json").read_text(encoding="utf-8")), None
    err_file = job_dir / "error.json"
    if err_file.exists():
        return None, json.loads(err_file.read_text(encoding="utf-8"))
    tail = "\n".join((proc.stdout or "").splitlines()[-5:])
    return None, {"code": "INTERNAL", "message": tail or f"worker exited with {proc.returncode}"}


def _tracked_fully(results: dict) -> bool:
    ev = results.get("events", {})
    return ev.get("release") is not None and ev.get("bounce") is not None


# ── One wicket ────────────────────────────────────────────────────────────────
def process_wicket(wicket_dir: Path) -> str:
    """Handle one incoming wicket folder. Returns the final state."""
    prefix = wicket_dir.relative_to(INCOMING).as_posix()
    parts = prefix.split("/")
    # The path is the identity. IDs written inside meta.json are never used to
    # build a path, so a crafted meta.json can't redirect where results go.
    if not _safe_parts(parts):
        raise Rejected(f"unexpected folder layout or characters: {prefix!r}")
    if INCOMING.resolve() not in wicket_dir.resolve().parents:
        raise Rejected("folder resolves outside the incoming directory")
    match_id = parts[0]

    meta_path = wicket_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            raise ValueError("meta.json is not an object")
    except (ValueError, OSError) as e:
        if time.time() - meta_path.stat().st_mtime < 60:
            raise NotReady(str(e))           # probably still being written
        raise Rejected(f"unreadable meta.json: {e}")

    window = wicket_dir / "window.mp4"
    if not window.is_file():
        raise Rejected("window.mp4 missing")
    if window.stat().st_size > MAX_WINDOW_BYTES:
        raise Rejected(f"window.mp4 is larger than {MAX_WINDOW_BYTES >> 20} MB")
    cap = cv2.VideoCapture(str(window))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0 or n_frames <= 0 or w <= 0 or h <= 0:
        raise Rejected("window.mp4 is not a readable video")
    if max(w, h) > MAX_FRAME_SIDE or n_frames / fps > MAX_WINDOW_SEC:
        raise Rejected(f"window is too large ({w}x{h}, {n_frames / fps:.0f}s)")
    started = time.time()

    ball_type = _ball_type(match_id)
    calibration, calib_path = _calibration(match_id, window)
    if calibration is None:
        waiting_h = (time.time() - meta_path.stat().st_mtime) / 3600
        if waiting_h > UNMATCHED_KEEP_HOURS:
            raise Rejected(f"no calibration was provided within {UNMATCHED_KEEP_HOURS:.0f} h")
        if prefix not in _announced:         # retried every pass; say it once
            _announced.add(prefix)
            rel = calib_path.relative_to(CLIPS).as_posix()
            _status(prefix, "needs_calibration",
                    message=f"No valid calibration for this match at this resolution. "
                            f"Create {rel} in the clips store (a frame to mark is saved next to it).")
            log("needs calibration", prefix=prefix, expected=calib_path)
        return "needs_calibration"
    _announced.discard(prefix)

    # 1. Find the delivery in the window.
    log("scanning window", prefix=prefix, ball=ball_type)
    candidates, info = find_delivery(window, BALL_PIPELINES[ball_type]["weights"], calibration)
    scan_sec = round(time.time() - started, 1)
    if not candidates:
        _set_aside(wicket_dir)
        _status(prefix, "no_delivery_found", scan_sec=scan_sec,
                frames_with_ball=info["frames_with_ball"],
                window_kept_hours=UNMATCHED_KEEP_HOURS)
        log("no delivery found", prefix=prefix, scan_sec=scan_sec)
        return "no_delivery_found"

    # 2 + 3. Cut each candidate and let the tracker judge it: the first one
    # where it finds both a release and a bounce is the delivery.
    job_dirs: list[Path] = []
    chosen = None           # (candidate, clip_path, job_dir, results)
    fallback = None         # best attempt so far, if none fully tracks
    last_error = None
    for n, cand in enumerate(candidates[:MAX_ATTEMPTS], start=1):
        job_dir = JOBS / uuid.uuid4().hex[:8]
        job_dirs.append(job_dir)
        clip = job_dir / "delivery.mp4"
        cut_clip(window, clip, cand.clip_start, cand.clip_end(info["duration_sec"]))
        log("tracking candidate", prefix=prefix, attempt=n,
            release_sec=round(cand.release_sec, 2), score=cand.score)
        results, error = _run_tracker(clip, job_dir, calibration, ball_type,
                                      f"{meta.get('delivery_id', 'delivery')}.mp4")
        attempt = (cand, clip, job_dir, results)
        if results is not None and _tracked_fully(results):
            chosen = attempt
            break
        if results is not None and (fallback is None or fallback[3] is None):
            fallback = attempt
        elif fallback is None:
            fallback = attempt
        last_error = error or last_error
    if chosen is None:
        chosen = fallback
    cand, clip, job_dir, results = chosen

    # 4. Store. Raw delivery first...
    (CLIPS / prefix).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(clip, CLIPS / prefix / "delivery.mp4")
    _write_json(CLIPS / prefix / "meta.json", {
        **meta,
        "ball_type": ball_type,
        "delivery": {
            "file": "delivery.mp4",
            "window_start_sec": round(cand.clip_start, 3),
            "window_end_sec": round(cand.clip_end(info["duration_sec"]), 3),
            "release_sec_in_window": round(cand.release_sec, 3),
            "finder_score": cand.score,
            "candidates_found": len(candidates),
        },
    })

    # ...then the tracker's output.
    out = TRACKED / prefix
    out.mkdir(parents=True, exist_ok=True)
    if results is not None:
        annotated = job_dir / "output.mp4"
        try:
            # Same fully-drawn video the web app's "download" button produces.
            from pipeline.bake_overlay import bake_full_overlay
            annotated = bake_full_overlay(job_dir.name, job_dir, ball_type)
        except Exception as e:  # noqa: BLE001 — fall back to the basic render
            log("overlay bake failed, using basic render", error=e)
        shutil.copyfile(annotated, out / "annotated.mp4")
        shutil.copyfile(job_dir / "results.json", out / "results.json")
        if (job_dir / "first_frame.jpg").exists():
            shutil.copyfile(job_dir / "first_frame.jpg", out / "thumbnail.jpg")
        state = "done" if _tracked_fully(results) else "tracked_partially"
        ev = results.get("events", {})
        _status(prefix, state, ball_type=ball_type, scan_sec=scan_sec,
                total_sec=round(time.time() - started, 1),
                events_found={k: ev.get(k) is not None for k in ("release", "bounce", "deviation")},
                verdict=results.get("verdict"))
    else:
        state = "tracking_failed"
        _status(prefix, state, ball_type=ball_type, error=last_error,
                total_sec=round(time.time() - started, 1))

    # 5. The window has done its job.
    shutil.rmtree(wicket_dir, ignore_errors=True)
    _prune_empty(wicket_dir.parent)
    for d in job_dirs:
        shutil.rmtree(d, ignore_errors=True)
    log(state, prefix=prefix, total_sec=round(time.time() - started, 1))
    return state


def _set_aside(wicket_dir: Path) -> None:
    """Move a clip out of the queue into incoming/_unmatched/ (kept for
    UNMATCHED_KEEP_HOURS so it can be looked at, then deleted). Folders that
    don't have the expected safe layout are deleted outright."""
    rel = wicket_dir.relative_to(INCOMING)
    if not _safe_parts(rel.parts):
        shutil.rmtree(wicket_dir, ignore_errors=True)
        _prune_empty(wicket_dir.parent)
        return
    dst = INCOMING / "_unmatched" / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(dst, ignore_errors=True)
    shutil.move(str(wicket_dir), str(dst))
    # Age is counted from now, not from when the clip was cut.
    if (dst / "meta.json").exists():
        os.utime(dst / "meta.json")
    _prune_empty(wicket_dir.parent)


def _prune_empty(d: Path) -> None:
    """Remove now-empty parent folders up to (not including) INCOMING."""
    while d != INCOMING and INCOMING in d.parents:
        try:
            d.rmdir()
        except OSError:
            return
        d = d.parent


def _prune_unmatched() -> None:
    root = INCOMING / "_unmatched"
    if not root.exists():
        return
    cutoff = time.time() - UNMATCHED_KEEP_HOURS * 3600
    for meta in root.rglob("meta.json"):
        if meta.stat().st_mtime < cutoff:
            shutil.rmtree(meta.parent, ignore_errors=True)


def pending() -> list[Path]:
    """Wicket folders ready to process (meta.json present = upload finished)."""
    if not INCOMING.exists():
        return []
    out = []
    for meta in INCOMING.rglob("meta.json"):
        rel = meta.relative_to(INCOMING)
        if any(part.startswith("_") for part in rel.parts):
            continue
        out.append(meta.parent)
    return sorted(out, key=lambda p: (p / "meta.json").stat().st_mtime)


def run_once() -> dict[str, str]:
    """Process everything waiting, one at a time (single GPU/CPU: no parallel jobs)."""
    results = {}
    # In bucket mode, pull new windows and refresh each pending match's inputs
    # (an operator may have just uploaded a calibration) before processing.
    if gcs_sync.enabled():
        gcs_sync.pull_new_windows(INCOMING)
    dirs = pending()
    if gcs_sync.enabled():
        for match_id in sorted({d.relative_to(INCOMING).parts[0] for d in dirs}):
            gcs_sync.refresh_match_inputs(match_id, CLIPS)
    for wicket_dir in dirs:
        prefix = wicket_dir.relative_to(INCOMING).as_posix()
        try:
            state = process_wicket(wicket_dir)
        except NotReady:
            state = "not_ready"
        except Exception as e:  # noqa: BLE001 — one bad clip must not stop the queue
            # Take it out of the queue, otherwise it would fail again on
            # every pass forever.
            kind = "rejected" if isinstance(e, Rejected) else "error"
            log(kind, prefix=prefix, error=e)
            try:
                _set_aside(wicket_dir)
            except OSError as move_err:
                log("could not set aside", prefix=prefix, error=move_err)
            if _safe_parts(prefix.split("/")):
                _status(prefix, kind,
                        error={"code": type(e).__name__, "message": str(e)[:500]})
            state = kind
        results[prefix] = state
        # Push outputs to the buckets and drop the _incoming/ window once done.
        if gcs_sync.enabled():
            try:
                gcs_sync.publish(prefix, state, CLIPS, TRACKED)
            except Exception as e:  # noqa: BLE001 — never let a GCS hiccup stop the queue
                log("gcs publish error", prefix=prefix, error=e)
    _prune_unmatched()
    return results


def _single_instance_lock():
    """Hold an OS-level lock for the life of the process so two consumers
    can't process the same clips. Released automatically if we crash."""
    ROOT.mkdir(parents=True, exist_ok=True)
    fh = open(ROOT / ".consumer.lock", "a+")
    try:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Wicket clip -> delivery -> tracker")
    p.add_argument("--once", action="store_true", help="process what is waiting, then exit")
    p.add_argument("--interval", type=float, default=3.0, help="seconds between checks")
    args = p.parse_args(argv)

    for d in (INCOMING, CLIPS, TRACKED):
        d.mkdir(parents=True, exist_ok=True)
    lock = _single_instance_lock()
    if lock is None:
        print("another wicket consumer is already running on this folder - exiting",
              file=sys.stderr)
        return 3
    shutil.rmtree(JOBS, ignore_errors=True)      # leftovers from a crashed run
    if gcs_sync.enabled():
        log("wicket consumer started (bucket mode)",
            clips_bucket=os.environ.get("WICKET_CLIPS_BUCKET"),
            tracked_bucket=os.environ.get("WICKET_TRACKED_BUCKET"))
    else:
        log("wicket consumer started (local folders)",
            incoming=INCOMING, clips=CLIPS, tracked=TRACKED)
    if args.once:
        for prefix, state in run_once().items():
            print(f"{state:20s} {prefix}")
        return 0
    # Wickets waiting on a calibration are retried each pass; only log once.
    while True:
        run_once()
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
