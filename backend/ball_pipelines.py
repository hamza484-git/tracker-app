"""Per-ball-type pipeline config: legacy tracker folder + ONNX weights.

Shared by the web app (app.py) and the wicket consumer (wicket_consumer.py)
so both always use the same model for a given ball colour. The legacy folder
is bound inside each job's worker subprocess via TRACKER_LEGACY_DIR.
"""
from __future__ import annotations

import os
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent
_LEGACY_V8M = _BACKEND_DIR / "worker" / "legacy" / "Zone-div_V8m"
_MODELS_DIR = _BACKEND_DIR / "models"

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
