# Live wicket tracking — deployment runbook & security review

Automatic flow: a wicket falls → the scorecard pipeline's clipper cuts a ~35 s
window and uploads it to GCS → the wicket consumer (this repo) pulls it, finds
the delivery, runs the ball tracker on a GPU, and writes the annotated clip +
result back to GCS.

```
 scorecard stack                         GCS                        tracker (GPU box)
 ┌───────────┐  window.mp4+meta   ┌──────────────────┐  pull   ┌────────────────┐
 │  clipper  │ ─────────────────▶ │ auto-wicket-clips│ ──────▶ │ wicket_consumer│
 └───────────┘                    │   _incoming/…    │         │  find delivery │
      ▲ Redis "wicket"            │   {match}/…      │ ◀─────  │  run tracker   │
      │                          └──────────────────┘  raw    │  (GPU)         │
                                  ┌──────────────────┐  push   │                │
                                  │auto-wicket-tracked│ ◀────── │  annotated+json│
                                  └──────────────────┘         └────────────────┘
```

The two sides share **no network** — only the buckets. They can be deployed,
restarted and scaled independently.

---

## 0. Prerequisites (the things that decide whether today is possible)

- **A GPU on the box that runs the consumer.** Without it the tracker is far
  too slow (~1 s/frame on CPU; a 5 s delivery = minutes). This is the single
  hard gate. Verify with `nvidia-smi` and the NVIDIA container runtime.
- **A service account (or the VM's default SA) with object read/write/delete on
  both buckets.** Grant (you, not automated — IAM is yours):
  ```bash
  SA=<your-service-account-email>
  gcloud storage buckets add-iam-policy-binding gs://auto-wicket-clips \
      --member="serviceAccount:$SA" --role="roles/storage.objectAdmin"
  gcloud storage buckets add-iam-policy-binding gs://auto-wicket-tracked \
      --member="serviceAccount:$SA" --role="roles/storage.objectAdmin"
  ```
  The clipper side needs the same on `auto-wicket-clips` only (it just uploads).
  On a GCP VM, prefer the VM's own service account over a key file.
- Buckets already exist and are private: `auto-wicket-clips` (raw, 30-day
  delete + 1-day on `_incoming/`) and `auto-wicket-tracked` (processed).
  Public access prevention is **enforced** on both.

---

## 1. Deploy the clipper change (scorecard stack)

The clipper already builds the window and uploads it. Point it at the bucket:

In the scorecard pipeline's `.env`:
```
HANDOFF_BUCKET_NAME=auto-wicket-clips
# CLIP_SECONDS_BEFORE / CLIP_SECONDS_AFTER default to 30 / 5 — widen BEFORE if a
# delivery is ever missed because the manual score delay is longer than ~1 min.
```
Give the clipper container GCS credentials (the VM's SA, or mount a key and set
`GOOGLE_APPLICATION_CREDENTIALS`). Then, from the scorecard repo:
```bash
docker compose build clipper mediamtx
docker compose up -d
```
> `mediamtx.yml` now keeps ~3 minutes of stream (was 14 s) so the clipper can
> reach back far enough. `CLIPS_LOCAL_DIR` must be **unset** for bucket mode.

## 2. Deploy the wicket consumer (GPU box)

From this repo's root:
```bash
export WICKET_CLIPS_BUCKET=auto-wicket-clips
export WICKET_TRACKED_BUCKET=auto-wicket-tracked
docker compose -f docker-compose.wicket.yml up -d --build
```
The image bakes in the ONNX models and legacy tracker folders; first build is
large (CUDA + models). Logs: `docker compose -f docker-compose.wicket.yml logs -f`.

## 3. Calibrate each match (once)

The camera is fixed, so calibration is per match, per resolution. On the first
wicket of a match the consumer can't calibrate itself — it writes a frame and
marks the wicket `needs_calibration`. To calibrate:

1. Download the saved frame:
   `gs://auto-wicket-clips/{match_id}/calibration/{W}x{H}_frame.jpg`
2. Mark the four crease corners and save a JSON next to it:
   `gs://auto-wicket-clips/{match_id}/calibration/{W}x{H}.json`
   ```json
   {"bat_L":[x,y],"bat_R":[x,y],"bowl_L":[x,y],"bowl_R":[x,y]}
   ```
3. Also set the ball colour: `gs://auto-wicket-clips/{match_id}/match.json`
   ```json
   {"ball_type":"red"}
   ```
The consumer re-checks every pass and processes the held wicket automatically
once the calibration appears. (A proper calibration screen is a later task; the
existing web UI's calibration canvas can produce these coordinates today.)

## 4. Verify it works

- Drop a test window (or wait for a real wicket) and watch the consumer logs.
- Output lands at `gs://auto-wicket-tracked/{match}/innings_n/over_k/{delivery}/`
  as `annotated.mp4`, `results.json`, `thumbnail.jpg`, `status.json`.
- `status.json.state` is `done` (release+bounce found), `tracked_partially`,
  `no_delivery_found`, or `needs_calibration`.

---

## Security review

### Hardened in the new code (clipper + consumer + gcs_sync)
- **Untrusted score input.** The wicket trigger arrives over Redis, which other
  processes can publish to. Every ID (match/innings/over/delivery) is validated
  against `^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$` and `..` is rejected **before** it
  becomes a path or a stream URL, in the clipper, the consumer, and gcs_sync.
- **Flood control.** One clip per match per 20 s, ≤2 running / ≤10 queued, so a
  burst of fake wickets can't exhaust disk or GPU.
- **Path-identity.** The folder path is the identity; IDs inside `meta.json` are
  never used to build a path, so a crafted `meta.json` can't redirect output.
- **Resource limits.** Window size/length/resolution capped; a hung tracker run
  is killed after 30 min; one bad clip is set aside, not retried forever.
- **Single instance.** An OS lock stops two consumers racing the same clips.
- **ffmpeg** is restricted to `http,https,tcp,tls,crypto` so a playlist can't
  point it at local files.
- **Buckets.** Private, public-access-prevention enforced, uniform access; the
  consumer writes only under validated prefixes; `_incoming/` windows auto-expire
  in 1 day.

### Residual risks — NOT fixed here (mostly the older scorecard stack)
These are the real exposure and should be closed before/with go-live:
1. **Redis is published on :6379 with no password.** Anyone who can reach it can
   inject fake wickets or rewrite the on-air score. Bind it to localhost / the
   internal Docker network, or set a password. *Highest priority.*
2. **Postgres on :5432 with password `scorecard`.** Restrict and rotate.
3. **Orchestrator has no auth and controls Docker** (`/var/run/docker.sock`),
   and is reachable via Caddy `/matches/*`. Add an auth token.
4. **mediamtx control API (:9997) and ingest are open.** Restrict to the box /
   add a publish token.
5. **`.env` holds the production DB password + API login** and sits in a
   OneDrive-synced folder. Move it out of cloud sync.
6. **Credentials for the buckets:** prefer the VM service account / Workload
   Identity. If you must use a key file, mount it read-only — never bake it into
   an image or commit it.

Whether 1/2/4 are reachable from the internet depends on the GCP firewall —
check that the VM only exposes 80/443 (+ the stream ports) externally.

---

## Rollback
- Clipper: set `HANDOFF_BUCKET_NAME=` (empty) → it stops uploading; or redeploy
  the previous image. The `mediamtx.yml` buffer change is safe to keep.
- Consumer: `docker compose -f docker-compose.wicket.yml down`. Nothing it does
  touches the scorecard stack.
- Buckets: `_incoming/` self-cleans; delivered clips expire on the 30-day rule.
