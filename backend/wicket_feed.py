"""Read side of the live wicket feed.

Lists recent processed wickets from the tracked store (the GCS bucket in
production, the local folder in dev) and resolves their media. The wicket
consumer writes, per wicket:

    {match}/innings_n/over_k/{delivery}/status.json   state + verdict
                                        annotated.mp4  the tracked clip
                                        results.json
                                        thumbnail.jpg

The feed returns one item per status.json, newest first, with same-origin
media URLs that point back at the `/api/track/live/media/...` route — so the
frontend never sees the bucket and the private videos stay behind the app.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

BACKEND_DIR = Path(__file__).resolve().parent
_ROOT = Path(os.environ.get("WICKET_WORK_DIR", BACKEND_DIR / "_tracker_work" / "wickets"))
TRACKED_DIR = Path(os.environ.get("WICKET_TRACKED_DIR", _ROOT / "auto-wicket-tracked"))
TRACKED_BUCKET = os.environ.get("WICKET_TRACKED_BUCKET", "").strip()

_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_MEDIA = {
    "annotated.mp4": "video/mp4",
    "thumbnail.jpg": "image/jpeg",
    "results.json": "application/json",
    "status.json": "application/json",
}
# States that have a playable clip + thumbnail.
_WITH_MEDIA = {"done", "tracked_partially"}

_client = None


def _gcs():
    global _client
    if _client is None:
        from google.cloud import storage
        _client = storage.Client()
    return _client


def using_bucket() -> bool:
    return bool(TRACKED_BUCKET)


def _safe_prefix(prefix: str) -> bool:
    parts = prefix.split("/")
    return len(parts) == 4 and all(_SAFE.match(p) and ".." not in p for p in parts)


def _item(prefix: str, status: dict) -> dict:
    parts = prefix.split("/")
    v = status.get("verdict") if isinstance(status.get("verdict"), dict) else None
    state = status.get("state")
    has_media = state in _WITH_MEDIA
    return {
        "prefix": prefix,
        "match_id": parts[0],
        "innings": parts[1].replace("innings_", ""),
        "over": parts[2].replace("over_", ""),
        "delivery_id": parts[3],
        "state": state,
        "ball_type": status.get("ball_type"),
        "updated_at": status.get("updated_at"),
        "verdict": None if not v else {
            "decision": v.get("decision"),
            "would_hit_stumps": v.get("would_hit_stumps"),
        },
        "has_media": has_media,
        "thumbnail_url": f"/api/track/live/media/{prefix}/thumbnail.jpg" if has_media else None,
        "video_url": f"/api/track/live/media/{prefix}/annotated.mp4" if has_media else None,
    }


def list_recent(limit: int = 30) -> list[dict]:
    """Recent wickets, newest first. Best-effort: unreadable entries are skipped."""
    items: list[dict] = []
    if TRACKED_BUCKET:
        try:
            blobs = [b for b in _gcs().list_blobs(TRACKED_BUCKET)
                     if b.name.endswith("/status.json")]
        except Exception:  # noqa: BLE001 — a listing failure returns an empty feed
            return []
        for b in blobs:
            prefix = b.name[:-len("/status.json")]
            if not _safe_prefix(prefix):
                continue
            try:
                items.append(_item(prefix, json.loads(b.download_as_text())))
            except Exception:  # noqa: BLE001
                continue
    elif TRACKED_DIR.exists():
        for sp in TRACKED_DIR.rglob("status.json"):
            prefix = sp.parent.relative_to(TRACKED_DIR).as_posix()
            if not _safe_prefix(prefix):
                continue
            try:
                items.append(_item(prefix, json.loads(sp.read_text(encoding="utf-8"))))
            except Exception:  # noqa: BLE001
                continue
    items.sort(key=lambda i: i.get("updated_at") or "", reverse=True)
    return items[:limit]


def resolve_media(prefix: str, name: str) -> Optional[dict]:
    """Validate and locate one media file. Returns {"kind","ctype",...} or None."""
    if not _safe_prefix(prefix) or name not in _MEDIA:
        return None
    ctype = _MEDIA[name]
    if TRACKED_BUCKET:
        blob = _gcs().bucket(TRACKED_BUCKET).blob(f"{prefix}/{name}")
        try:
            if not blob.exists():
                return None
            return {"kind": "gcs", "ctype": ctype, "bytes": blob.download_as_bytes()}
        except Exception:  # noqa: BLE001
            return None
    p = TRACKED_DIR / prefix / name
    if not p.is_file():
        return None
    return {"kind": "local", "ctype": ctype, "path": p}
