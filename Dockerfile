# syntax=docker/dockerfile:1.6
#
# Cricket Ball Tracker — Cloud Run + L4 GPU image (single service).
#
# Build context: repo root. All sources live inside the repo — no external
# staging step. Frontend is built in stage 1 and served by the backend.

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — frontend build
# ─────────────────────────────────────────────────────────────────────────────
FROM node:20-slim AS frontend
WORKDIR /frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
ENV VITE_API_BASE=/api
RUN npm run build


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — runtime (CUDA + Python + backend)
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        ffmpeg libgl1 libglib2.0-0 \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3

WORKDIR /app

# Swap the CPU onnxruntime pin for the GPU build so the L4 is actually used.
COPY backend/requirements.txt /tmp/requirements.txt
RUN sed -i 's/^onnxruntime>=.*/onnxruntime-gpu>=1.16/' /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Backend source — includes worker/legacy/{Zone-div(final),Zone-div_V8m}
# and models/{whiteball,redball,pinkball}.onnx (see .dockerignore for excludes).
COPY backend/ /app/

# Frontend bundle from stage 1
COPY --from=frontend /frontend/dist /app/frontend_dist

# Sanity: fail the build early if any required artifact is missing.
RUN test -f /app/worker/legacy/Zone-div_V8m/tracker.py \
 && test -f /app/models/whiteball.onnx \
 && test -f /app/models/redball.onnx \
 && test -f /app/models/pinkball.onnx \
 && test -f /app/frontend_dist/index.html

ENV TRACKER_WORK_DIR=/tmp/tracker_work \
    WICKET_WORK_DIR=/tmp/wickets \
    TRACKER_BALL_ASSETS_DIR=/app/frontend_dist \
    FRONTEND_DIST=/app/frontend_dist \
    PORT=8080 \
    SIGNED_URL_TTL_SECONDS=3600 \
    MAX_UPLOAD_MB=500 \
    MAX_CLIP_SECONDS=30

EXPOSE 8080

# Single worker — JobStore + exec_lock are in-process and must not be sharded.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
