"""FastAPI app exposing the cricket tracker.

Routes (all under /api/track):
  POST /first-frame      upload video → return first frame URL + upload_id
  POST /analyze          create job from upload_id + calibration
  GET  /jobs             recent jobs (for the job list UI)
  GET  /jobs/{job_id}    job status (poll target)
  POST /jobs/{job_id}/cancel
  GET  /jobs/{job_id}/result  full Section-4 JSON with fresh signed URLs
  GET  /files/{job_id}/{filename}  local-storage file serving (LocalStorage only)
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
from fastapi import (
    BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from jobs import JobStore, run_job
from pipeline.errors import InvalidVideoError
from storage import LocalStorage, make_storage


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "500"))
MAX_CLIP_SECONDS = float(os.environ.get("MAX_CLIP_SECONDS", "30"))
WORK_DIR = Path(os.environ.get("TRACKER_WORK_DIR", "./_tracker_work"))
# Cloud Run's own scale-to-zero idle window isn't configurable via gcloud;
# this self-shutdown makes the L4 instance give itself up sooner. 0 disables.
IDLE_SHUTDOWN_SECONDS = float(os.environ.get("IDLE_SHUTDOWN_SECONDS", "300"))

# Per-ball-type pipeline: legacy tracker folder + ONNX weights. The folder is
# bound inside each job's worker subprocess via TRACKER_LEGACY_DIR.
_LEGACY_V8M = Path(__file__).resolve().parent / "worker" / "legacy" / "Zone-div_V8m"
_MODELS_DIR = Path(__file__).resolve().parent / "models"
BALL_PIPELINES: dict[str, dict[str, Path]] = {
    "white": {
        "legacy_dir": Path(os.environ.get("TRACKER_LEGACY_DIR_WHITE", _LEGACY_V8M)),
        "weights": Path(os.environ.get(
            "TRACKER_WEIGHTS_WHITE", str(_MODELS_DIR / "whiteball.onnx"))),
    },
    "red": {
        "legacy_dir": Path(os.environ.get("TRACKER_LEGACY_DIR_RED", _LEGACY_V8M)),
        # Using redsepv2.onnx (YOLOv8m-P2 trained on reddatav2/merged_dataset,
        # 2026-09-27). Previous models still on disk (rollback by swapping
        # filename below): redkaggle.onnx, redsep.onnx, redball.onnx.
        "weights": Path(os.environ.get(
            "TRACKER_WEIGHTS_RED", str(_MODELS_DIR / "redsepv2.onnx"))),
    },
    "pink": {
        "legacy_dir": Path(os.environ.get("TRACKER_LEGACY_DIR_PINK", _LEGACY_V8M)),
        # Using pinkoct.onnx (YOLO exported 2026-10-21 from pinkoct.pt,
        # imgsz=1280, opset=12). Previous model still on disk (rollback by
        # swapping filename below or setting TRACKER_WEIGHTS_PINK): pinkball.onnx.
        "weights": Path(os.environ.get(
            "TRACKER_WEIGHTS_PINK", str(_MODELS_DIR / "pinkoct.onnx"))),
    },
}

ACCEPTED_MIME = {
    "video/mp4", "video/quicktime", "video/x-msvideo", "video/avi",
}


# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Cricket Ball Tracker")
# Same-origin in prod (frontend bundle served below). CORS_ORIGINS overrides
# for split-deploy scenarios; comma-separated list.
_cors_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

store = JobStore()
storage = make_storage(WORK_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# Idle self-shutdown: exit the process after IDLE_SHUTDOWN_SECONDS with no
# requests and no job running, so Cloud Run (--min-instances=0) reclaims the
# L4 instance promptly instead of waiting on its internal idle window.
# ─────────────────────────────────────────────────────────────────────────────
_last_activity = time.time()


@app.middleware("http")
async def _touch_activity(request: Request, call_next):
    global _last_activity
    _last_activity = time.time()
    return await call_next(request)


def _idle_watchdog() -> None:
    while True:
        time.sleep(15)
        idle_for = time.time() - _last_activity
        if idle_for >= IDLE_SHUTDOWN_SECONDS and not store.exec_lock.locked():
            os._exit(0)


if IDLE_SHUTDOWN_SECONDS > 0:
    threading.Thread(target=_idle_watchdog, daemon=True).start()

# Upload area for first-frame extraction, awaiting /analyze.
_uploads_root: Path = (
    storage.upload_dir() if isinstance(storage, LocalStorage)
    else WORK_DIR / "uploads"
)
_uploads_root.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _save_upload_streaming(src: UploadFile, dst: Path, max_bytes: int) -> int:
    """Save UploadFile to dst, enforcing max_bytes. Returns bytes written."""
    written = 0
    chunk = 1 << 20  # 1 MiB
    with open(dst, "wb") as f:
        while True:
            data = src.file.read(chunk)
            if not data:
                break
            written += len(data)
            if written > max_bytes:
                f.close()
                dst.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail={
                        "code": "VIDEO_TOO_LARGE",
                        "message": f"max {max_bytes // (1<<20)} MiB",
                    },
                )
            f.write(data)
    return written


def _extract_first_frame(video_path: Path, out_jpg: Path) -> tuple[int, int, float]:
    """Returns (width, height, duration_sec)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise InvalidVideoError("cannot open")
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise InvalidVideoError("no first frame")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    cv2.imwrite(str(out_jpg), frame)
    duration = frames / fps if fps > 0 else 0.0
    return w, h, duration


def _result_with_urls(rec) -> dict:
    """Stamp fresh signed URLs onto the cached result."""
    if rec.result is None:
        return {}
    out = json.loads(json.dumps(rec.result))  # deep copy
    job_id = rec.job_id
    out["outputs"] = {
        "annotated_video_url": storage.signed_url(job_id, "output.mp4"),
        "csv_url": storage.signed_url(
            job_id, rec.result["outputs"].get("csv_url", "detections.csv")
        ),
        "first_frame_url": storage.signed_url(job_id, "first_frame.jpg"),
    }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/track/first-frame")
def first_frame(video: UploadFile = File(...)):
    """Upload a video, return its first frame URL + dimensions + upload_id."""
    if video.content_type not in ACCEPTED_MIME:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_VIDEO",
                "message": f"unsupported content-type {video.content_type}",
            },
        )
    upload_id = uuid.uuid4().hex[:12]
    upload_dir = _uploads_root / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    src = upload_dir / "input.mp4"
    _save_upload_streaming(video, src, MAX_UPLOAD_MB * (1 << 20))

    try:
        w, h, duration = _extract_first_frame(src, upload_dir / "first_frame.jpg")
    except InvalidVideoError as e:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_VIDEO", "message": str(e)},
        )

    # 0.5 s of slack absorbs fps metadata rounding on legitimate 30 s clips.
    if duration > MAX_CLIP_SECONDS + 0.5:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(
            status_code=422,
            detail={
                "code": "VIDEO_TOO_LONG",
                "message": (
                    f"clip is {duration:.1f} s — max {MAX_CLIP_SECONDS:.0f} s"
                ),
            },
        )

    return {
        "upload_id": upload_id,
        "filename": video.filename or "upload.mp4",
        "width": w,
        "height": h,
        "duration_sec": round(duration, 2),
        "first_frame_url": f"/api/track/uploads/{upload_id}/first_frame.jpg",
    }


@app.get("/api/track/uploads/{upload_id}/suggest")
def suggest_calibration(upload_id: str):
    """Rough auto-placement of the 4 stump-base markers so the user starts with
    a plausible layout and only needs to drag-tune. Default assumes the camera
    is behind the bowler: bowler's stumps near the BOTTOM of the frame
    (closer = wider) and batter's stumps near the TOP (farther = narrower).
    """
    p = _uploads_root / upload_id / "first_frame.jpg"
    if not p.exists():
        raise HTTPException(status_code=404, detail="upload not found")
    img = cv2.imread(str(p))
    if img is None:
        raise HTTPException(status_code=422, detail="cannot read first frame")
    h, w = img.shape[:2]
    cx = w / 2
    # Camera behind bowler: batter is FAR (top of frame, narrower).
    bat_y = int(h * 0.30)
    bowl_y = int(h * 0.82)
    bat_half = w * 0.10   # batter's end narrower (farther from camera)
    bowl_half = w * 0.22  # bowler's end wider (closer to camera)
    # Seed middle-stump bases slightly behind their crease at each end.
    # 1.22 m stump-to-crease over 15.24 m pitch length ⇒ ~8% of the on-screen
    # bat→bowl vector. This matches the offset the pipeline used to synthesize.
    stump_off = (bowl_y - bat_y) * (1.22 / 15.24)
    bat_stump_y = int(bat_y - stump_off)
    bowl_stump_y = int(bowl_y + stump_off)
    return {
        "calibration": {
            "bat_L":  [int(cx - bat_half),  bat_y],
            "bat_R":  [int(cx + bat_half),  bat_y],
            "bowl_L": [int(cx - bowl_half), bowl_y],
            "bowl_R": [int(cx + bowl_half), bowl_y],
            "bat_stump":  [int(cx), bat_stump_y],
            "bowl_stump": [int(cx), bowl_stump_y],
        },
        "video": {"width": w, "height": h},
    }


@app.get("/api/track/uploads/{upload_id}/first_frame.jpg")
def get_upload_first_frame(upload_id: str):
    p = _uploads_root / upload_id / "first_frame.jpg"
    if not p.exists():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(p, media_type="image/jpeg")


@app.post("/api/track/analyze")
def analyze(
    background_tasks: BackgroundTasks,
    upload_id: str = Form(...),
    calibration: str = Form(...),
    interp_method: str = Form("parabola"),
    ball_type: str = Form("white"),
    filename: Optional[str] = Form(None),
):
    """Create a job from an existing upload + calibration JSON string."""
    upload_dir = _uploads_root / upload_id
    src_video = upload_dir / "input.mp4"
    if not src_video.exists():
        raise HTTPException(
            status_code=404,
            detail={"code": "INVALID_VIDEO", "message": "upload_id not found"},
        )

    if ball_type not in BALL_PIPELINES:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_BALL_TYPE",
                "message": f"ball_type must be one of {sorted(BALL_PIPELINES)}",
            },
        )
    pipeline_cfg = BALL_PIPELINES[ball_type]

    try:
        calib_data = json.loads(calibration)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_CALIBRATION", "message": str(e)},
        )

    rec = store.create(
        upload_id=upload_id,
        calibration=calib_data,
        filename=filename or "video.mp4",
        interp_method=interp_method,
        ball_type=ball_type,
    )
    job_dir = storage.job_dir(rec.job_id)
    job_video = job_dir / "input.mp4"
    shutil.copy2(src_video, job_video)
    # Cleanup the upload area now that we own a copy.
    shutil.rmtree(upload_dir, ignore_errors=True)

    def _on_done(jid: str):
        # Upload to remote storage (no-op for LocalStorage)
        try:
            storage.upload_job_artifacts(jid, job_dir)
        except Exception:  # noqa: BLE001
            pass

    background_tasks.add_task(
        run_job, store, rec.job_id, job_video, job_dir,
        calib_data, pipeline_cfg["weights"], interp_method,
        pipeline_cfg["legacy_dir"], _on_done,
    )
    return JSONResponse(
        status_code=202,
        content={"job_id": rec.job_id, "status": "queued"},
    )


@app.get("/api/track/jobs")
def list_jobs(limit: int = 20):
    items = []
    for rec in store.list(limit=limit):
        bounce = None
        zone = None
        if rec.result:
            bounce = rec.result.get("events", {}).get("bounce")
            zone = bounce.get("zone") if bounce else None
        items.append({
            "job_id": rec.job_id,
            "filename": rec.filename,
            "status": rec.status,
            "ball_type": rec.ball_type,
            "created_at": rec.created_at,
            "zone": zone,
            "first_frame_url": (
                storage.signed_url(rec.job_id, "first_frame.jpg")
                if rec.status == "done" else None
            ),
            "has_feedback": (storage.job_dir(rec.job_id) / "feedback.json").exists(),
        })
    return {"jobs": items}


@app.get("/api/track/jobs/{job_id}")
def get_job(job_id: str):
    rec = store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="unknown job_id")
    return store.to_public(rec)


@app.post("/api/track/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if not store.request_cancel(job_id):
        raise HTTPException(status_code=409, detail="cannot cancel")
    return {"job_id": job_id, "status": "cancelling"}


@app.get("/api/track/jobs/{job_id}/result")
def get_result(job_id: str):
    rec = store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="unknown job_id")
    if rec.status != "done":
        raise HTTPException(
            status_code=409,
            detail={"status": rec.status, "message": "job not done"},
        )
    return _result_with_urls(rec)


@app.get("/api/track/jobs/{job_id}/download_full")
def download_full(job_id: str):
    """Lazy-bake and serve `output_full.mp4` — original annotated video with
    all frontend SVG overlays (stumps, pitching line, parabolas, event ball
    markers) baked in. First call per job pays the render cost (~5–15s for a
    30s clip); subsequent calls serve the cached file instantly."""
    rec = store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="unknown job_id")
    if rec.status != "done":
        raise HTTPException(
            status_code=409,
            detail={"status": rec.status, "message": "job not done"},
        )
    from pipeline.bake_overlay import bake_full_overlay
    job_dir = storage.job_dir(job_id)
    try:
        out = bake_full_overlay(job_id, job_dir, rec.ball_type)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:  # noqa: BLE001 — surface render failures as 500
        raise HTTPException(status_code=500, detail=f"bake failed: {e}")
    return FileResponse(out, media_type="video/mp4", filename="annotated.mp4")


@app.post("/api/track/jobs/{job_id}/feedback")
async def submit_feedback(job_id: str, request: Request):
    rec = store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="unknown job_id")
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="expected object")
    from datetime import datetime, timezone
    payload["job_id"] = job_id
    payload["submitted_at"] = datetime.now(timezone.utc).isoformat()
    path = storage.job_dir(job_id) / "feedback.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"ok": True}


@app.get("/api/track/jobs/{job_id}/feedback")
def get_feedback(job_id: str):
    path = storage.job_dir(job_id) / "feedback.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="no feedback")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/track/files/{job_id}/{filename}")
def serve_local_file(job_id: str, filename: str):
    """Local-storage signed-URL endpoint (no auth — dev only)."""
    # Path traversal guard
    safe = Path(filename).name
    if safe != filename:
        raise HTTPException(status_code=400, detail="bad filename")
    p = storage.job_dir(job_id) / safe
    if not p.exists():
        raise HTTPException(status_code=404, detail="not found")
    media = "video/mp4" if safe.endswith(".mp4") else (
        "image/jpeg" if safe.endswith(".jpg") else (
            "text/csv" if safe.endswith(".csv") else (
                "application/json" if safe.endswith(".json") else "application/octet-stream"
            )
        )
    )
    return FileResponse(p, media_type=media, filename=safe)


@app.get("/api/track/health")
def health():
    return {
        "ok": True,
        "pipelines": {
            name: {
                "weights_present": cfg["weights"].exists(),
                "legacy_dir_present": cfg["legacy_dir"].exists(),
            }
            for name, cfg in BALL_PIPELINES.items()
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Frontend bundle (Cloud Run: FRONTEND_DIST=/app/frontend_dist)
# Declared LAST so the /api/* routes above take precedence over the SPA
# catch-all. The fallback serves index.html for unknown paths so React
# Router history routes work on deep links / refresh.
# ─────────────────────────────────────────────────────────────────────────────
_frontend_dist = os.environ.get("FRONTEND_DIST")
if _frontend_dist and Path(_frontend_dist).is_dir():
    _dist = Path(_frontend_dist)

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404)
        candidate = _dist / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_dist / "index.html")
