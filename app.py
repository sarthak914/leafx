"""Flask REST API for leaf disease detection (Milestone 2 build).

Endpoints:
- GET  /api/health   server status + which trained weights are loaded
- POST /api/predict  run YOLOv8 detection on a base64 leaf image
- GET  /api/history  last 25 predictions (SQLite)
The built React frontend (DTI/dist) is served from "/".
"""

from __future__ import annotations

import base64
import io
import math
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=True, encoding="utf-8-sig")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.predict import DEFAULT_WEIGHTS_REL, find_weights, summarize_detections  # noqa: E402

DB_PATH = ROOT / "predictions.db"
FRONTEND_DIST = ROOT / "DTI" / "dist"
DEFAULT_CONF = 0.25


def json_safe_payload(obj: Any) -> Any:
    """Normalize nested structures for strict JSON (no NaN/Inf; numpy scalars → Python)."""
    if isinstance(obj, dict):
        return {str(k): json_safe_payload(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe_payload(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        if math.isnan(x) or math.isinf(x):
            return 0.0
        return x
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


class StrictJSONProvider(DefaultJSONProvider):
    """Browsers reject JSON with NaN/Infinity; forbid them app-wide."""

    ensure_ascii = False

    def dumps(self, obj: Any, **kwargs: Any) -> str:
        kwargs.setdefault("allow_nan", False)
        if isinstance(obj, (dict, list)):
            obj = json_safe_payload(obj)
        return super().dumps(obj, **kwargs)


app = Flask(__name__)
CORS(app)
app.json = StrictJSONProvider(app)


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            plant TEXT NOT NULL,
            disease TEXT NOT NULL,
            confidence_pct REAL NOT NULL,
            severity_pct REAL NOT NULL,
            severity_level TEXT NOT NULL,
            health_pct REAL NOT NULL,
            recommendation TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def save_prediction_history(payload: dict[str, Any]) -> None:
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO predictions
            (created_at, plant, disease, confidence_pct, severity_pct, severity_level, health_pct, recommendation)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            payload["plant"],
            payload["disease"],
            payload["confidence"],
            payload["severity"],
            payload["severityLevel"],
            payload["plantHealth"],
            payload["suggestion"],
        ),
    )
    conn.commit()
    conn.close()


WEIGHTS_PATH = find_weights(ROOT)
MODEL = YOLO(str(WEIGHTS_PATH)) if WEIGHTS_PATH else None
if MODEL is None:
    print(
        f"[WARN] No trained weights found. Train with `python train_yolo.py` (creates {DEFAULT_WEIGHTS_REL}) "
        "or set PLANTDOC_WEIGHTS in .env. /api/predict will return 503 until then."
    )


def decode_data_url(data_url: str) -> Image.Image:
    if "," not in data_url:
        raise ValueError("Invalid image payload.")
    encoded = data_url.split(",", 1)[1]
    image_bytes = base64.b64decode(encoded)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def encode_image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


@app.get("/api/health")
def health() -> Any:
    return jsonify(
        {
            "status": "ok",
            "modelLoaded": MODEL is not None,
            "weights": WEIGHTS_PATH.name if WEIGHTS_PATH else None,
            "weightsPath": str(WEIGHTS_PATH) if WEIGHTS_PATH else None,
            "classes": len(MODEL.names) if MODEL is not None else 0,
        }
    )


@app.post("/api/predict")
def api_predict() -> Any:
    if MODEL is None:
        return jsonify(
            {
                "error": (
                    "No trained model loaded. Run `python train_yolo.py` first "
                    "(or set PLANTDOC_WEIGHTS in .env) and restart the server."
                )
            }
        ), 503

    body = request.get_json(silent=True) or {}
    image_data_url = str(body.get("imageDataUrl", "")).strip()
    confidence_threshold = float(body.get("conf", DEFAULT_CONF))
    if not image_data_url:
        return jsonify({"error": "imageDataUrl is required"}), 400

    try:
        image = decode_data_url(image_data_url)
    except Exception:
        return jsonify({"error": "Could not read the uploaded file as an image."}), 400

    try:
        started = time.perf_counter()
        # Pass the PIL image (RGB) directly; raw numpy arrays are treated as BGR by Ultralytics.
        results = MODEL.predict(source=image, conf=confidence_threshold, verbose=False)
        inference_ms = (time.perf_counter() - started) * 1000.0
        if not results:
            return jsonify({"error": "Model did not return any result"}), 500

        result = results[0]
        annotated = Image.fromarray(result.plot()[:, :, ::-1])
        detections = [
            {
                "plant": s.plant,
                "disease": s.disease,
                "isHealthy": s.is_healthy,
                "confidence": round(s.confidence_pct, 2),
                "severity": round(s.severity_pct, 2),
                "severityLevel": s.severity,
                "plantHealth": round(s.health_pct, 2),
                "suggestion": s.recommendation,
                "bbox": [round(v, 1) for v in s.bbox_xyxy],
            }
            for s in summarize_detections(result, MODEL)
        ]
        detections.sort(key=lambda d: d["confidence"], reverse=True)

        if not detections:
            return jsonify(
                {
                    "status": "no_detection",
                    "message": (
                        f"No leaf condition detected above {confidence_threshold:.0%} confidence. "
                        "Try a closer, well-lit photo of a single leaf."
                    ),
                    "annotatedImage": encode_image_to_data_url(annotated),
                    "detections": [],
                    "inferenceMs": round(inference_ms, 1),
                }
            )

        payload = {"status": "detected", **detections[0]}
        payload["annotatedImage"] = encode_image_to_data_url(annotated)
        payload["detections"] = detections
        payload["inferenceMs"] = round(inference_ms, 1)
        save_prediction_history(payload)
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/history")
def history() -> Any:
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT id, created_at, plant, disease, confidence_pct, severity_pct, severity_level, health_pct, recommendation
        FROM predictions
        ORDER BY id DESC
        LIMIT 25
        """
    ).fetchall()
    conn.close()
    return jsonify(
        [
            {
                "id": row["id"],
                "createdAt": row["created_at"],
                "plant": row["plant"],
                "disease": row["disease"],
                "confidence": row["confidence_pct"],
                "severity": row["severity_pct"],
                "severityLevel": row["severity_level"],
                "plantHealth": row["health_pct"],
                "suggestion": row["recommendation"],
            }
            for row in rows
        ]
    )


@app.get("/")
def index() -> Any:
    if (FRONTEND_DIST / "index.html").exists():
        return send_from_directory(FRONTEND_DIST, "index.html")
    return jsonify(
        {
            "message": "Frontend not built. Run: cd DTI && npm install && npm run build",
            "health": "/api/health",
        }
    )


@app.get("/<path:path>")
def frontend_assets(path: str) -> Any:
    """Must be registered last so /api/* routes are not treated as static files."""
    if path.startswith("api/"):
        return jsonify({"error": "Not found"}), 404

    target = FRONTEND_DIST / path
    if target.exists() and target.is_file():
        return send_from_directory(FRONTEND_DIST, path)
    # Client-side routes (/scan, /results, /history) fall back to the SPA entry.
    if (FRONTEND_DIST / "index.html").exists():
        return send_from_directory(FRONTEND_DIST, "index.html")
    return jsonify({"error": "Not found"}), 404


init_db()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
