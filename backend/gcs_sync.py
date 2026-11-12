"""GCS hand-off for the wicket consumer.

When the clipper and the tracker run on different machines, the window is
passed through Google Cloud Storage instead of a shared folder:

    gs://{clips_bucket}/_incoming/{prefix}/window.mp4   uploaded by the clipper
    gs://{clips_bucket}/_incoming/{prefix}/meta.json     (uploaded last)

The consumer still does all its work on local disk. This module only syncs:

  pull_new_windows()    _incoming/ in the bucket   -> local incoming/
  refresh_match_inputs()match.json + calibration/   -> local clips/
  publish()             local clips/ + tracked/     -> the two buckets,
                        and deletes the _incoming/ window once the wicket
                        reaches a terminal state.

Persistent per-match inputs and the raw delivery live in the clips bucket;
the tracker output lives in the tracked bucket. Both use the same path as
the local folders, so nothing about the layout changes.

Bucket mode is ON when WICKET_CLIPS_BUCKET is set; otherwise every call here
is a no-op and the consumer runs purely on local folders (dev / co-located).

Env:
    WICKET_CLIPS_BUCKET      e.g. auto-wicket-clips   (also the default
                             source of _incoming windows)
    WICKET_TRACKED_BUCKET    e.g. auto-wicket-tracked
    WICKET_INCOMING_BUCKET   default = WICKET_CLIPS_BUCKET
    WICKET_INCOMING_PREFIX   default "_incoming"
    WICKET_PULL_MAX          most windows pulled per pass (default 10)
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

_CLIPS_BUCKET = os.environ.get("WICKET_CLIPS_BUCKET", "").strip()
_TRACKED_BUCKET = os.environ.get("WICKET_TRACKED_BUCKET", "").strip()
_INCOMING_BUCKET = os.environ.get("WICKET_INCOMING_BUCKET", "").strip() or _CLIPS_BUCKET
_INCOMING_PREFIX = os.environ.get("WICKET_INCOMING_PREFIX", "_incoming").strip("/")
_PULL_MAX = int(os.environ.get("WICKET_PULL_MAX", "10"))

# States that mean "done with this window" — the _incoming/ copy is deleted.
# needs_calibration / not_ready keep the window so a later pass can retry.
_TERMINAL = {"done", "tracked_partially", "tracking_failed",
             "no_delivery_found", "rejected", "error"}

_client = None


def _log(msg: str, **kv) -> None:
    extra = " ".join(f"{k}={v}" for k, v in kv.items())
    print(f"[{datetime.now().strftime('%H:%M:%S')}] gcs {msg} {extra}".rstrip(), flush=True)


def enabled() -> bool:
    return bool(_CLIPS_BUCKET)


def client():
    global _client
    if _client is None:
        from google.cloud import storage
        _client = storage.Client()
    return _client


def _bucket(name: str):
    return client().bucket(name)


def _safe_prefix(prefix: str) -> bool:
    parts = prefix.split("/")
    import re
    pat = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
    return len(parts) == 4 and all(pat.match(p) and ".." not in p for p in parts)


# ── Pull ──────────────────────────────────────────────────────────────────────
def pull_new_windows(incoming_dir: Path) -> list[str]:
    """Download windows that landed in _incoming/ and aren't local yet.
    Returns the prefixes pulled. A window is ready only once its meta.json
    object exists (the clipper uploads it last)."""
    if not enabled():
        return []
    pulled: list[str] = []
    bucket = _bucket(_INCOMING_BUCKET)
    base = f"{_INCOMING_PREFIX}/"
    try:
        metas = [b for b in client().list_blobs(_INCOMING_BUCKET, prefix=base)
                 if b.name.endswith("/meta.json")]
    except Exception as e:  # noqa: BLE001 — a listing error must not crash the loop
        _log("list_failed", bucket=_INCOMING_BUCKET, error=e)
        return []
    for meta_blob in metas:
        if len(pulled) >= _PULL_MAX:
            break
        prefix = meta_blob.name[len(base):-len("/meta.json")]
        if not _safe_prefix(prefix):
            _log("skip_unsafe_incoming", key=meta_blob.name[:120])
            continue
        local = incoming_dir / prefix
        # Already have it locally, or already set aside — don't re-pull.
        if (local / "meta.json").exists() or (incoming_dir / "_unmatched" / prefix).exists():
            continue
        try:
            local.mkdir(parents=True, exist_ok=True)
            win = bucket.blob(f"{base}{prefix}/window.mp4")
            if not win.exists():
                continue                      # meta without window — ignore
            win.download_to_filename(str(local / "window.mp4.part"))
            os.replace(local / "window.mp4.part", local / "window.mp4")
            meta_blob.download_to_filename(str(local / "meta.json.part"))
            os.replace(local / "meta.json.part", local / "meta.json")
            pulled.append(prefix)
            _log("pulled_window", prefix=prefix)
        except Exception as e:  # noqa: BLE001
            _log("pull_failed", prefix=prefix, error=e)
            # Leave no half-written files behind.
            for p in (local / "window.mp4.part", local / "meta.json.part"):
                try:
                    p.unlink()
                except OSError:
                    pass
    return pulled


def refresh_match_inputs(match_id: str, clips_dir: Path) -> None:
    """Pull match.json and calibration/*.json for one match so the consumer
    sees calibration an operator has just uploaded to the bucket."""
    if not enabled():
        return
    try:
        for blob in client().list_blobs(_CLIPS_BUCKET, prefix=f"{match_id}/"):
            rel = blob.name[len(match_id) + 1:]
            if rel == "match.json" or (
                rel.startswith("calibration/") and rel.endswith(".json")
            ):
                dst = clips_dir / match_id / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                blob.download_to_filename(str(dst.with_suffix(dst.suffix + ".part")))
                os.replace(dst.with_suffix(dst.suffix + ".part"), dst)
    except Exception as e:  # noqa: BLE001
        _log("refresh_failed", match=match_id, error=e)


# ── Push ──────────────────────────────────────────────────────────────────────
def _upload(bucket_name: str, local: Path, key: str) -> None:
    _bucket(bucket_name).blob(key).upload_from_filename(str(local))


def publish(prefix: str, state: str, clips_dir: Path, tracked_dir: Path) -> None:
    """Push this wicket's outputs to the buckets and, once it is finished with,
    delete its _incoming/ window. Best-effort: a GCS hiccup is logged, not
    fatal — the next pass tries again (local files are the source of truth)."""
    if not enabled() or not _safe_prefix(prefix):
        return
    match_id = prefix.split("/")[0]
    try:
        # Raw delivery + its meta (only exist once a delivery was produced).
        for name in ("delivery.mp4", "meta.json"):
            f = clips_dir / prefix / name
            if f.is_file():
                _upload(_CLIPS_BUCKET, f, f"{prefix}/{name}")
        # Calibration frame(s) saved for this match, so an operator can mark them.
        calib_dir = clips_dir / match_id / "calibration"
        if calib_dir.is_dir():
            for frame in calib_dir.glob("*_frame.jpg"):
                _upload(_CLIPS_BUCKET, frame, f"{match_id}/calibration/{frame.name}")
        # Tracker output + status.json (status also carries needs_calibration).
        out_dir = tracked_dir / prefix
        if out_dir.is_dir() and _TRACKED_BUCKET:
            for f in out_dir.iterdir():
                if f.is_file():
                    _upload(_TRACKED_BUCKET, f, f"{prefix}/{f.name}")
    except Exception as e:  # noqa: BLE001
        _log("publish_failed", prefix=prefix, error=e)
        return

    if state in _TERMINAL:
        _delete_incoming(prefix)


def _delete_incoming(prefix: str) -> None:
    base = f"{_INCOMING_PREFIX}/{prefix}/"
    try:
        for blob in client().list_blobs(_INCOMING_BUCKET, prefix=base):
            blob.delete()
    except Exception as e:  # noqa: BLE001
        _log("delete_incoming_failed", prefix=prefix, error=e)
