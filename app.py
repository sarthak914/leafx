"""Flask REST API for leaf disease detection (Milestone 3 build).

Endpoints:
- GET  /api/health          server status + which models are loaded
- POST /api/predict         diagnose a base64 leaf image (classifier + detector + Grad-CAM)
- GET  /api/history         the logged-in user's last 25 scans
- POST /api/auth/signup     create an account (and log in)
- POST /api/auth/login      log in
- POST /api/auth/logout     log out
- GET  /api/auth/me         current user (or null)
- POST /api/chat            Gemini farming assistant (needs GEMINI_API_KEY in .env)
The built React frontend (DTI/dist) is served from "/".
"""

from __future__ import annotations

import base64
import io
import math
import os
import re
import secrets
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Never let Ultralytics pip-install packages in the middle of a request (slow, needs internet).
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory, session
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO
from werkzeug.security import check_password_hash, generate_password_hash

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=True, encoding="utf-8-sig")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.classifier import DEFAULT_CLS_WEIGHTS_REL, diagnose, find_classifier_weights  # noqa: E402
from model.predict import DEFAULT_WEIGHTS_REL, find_weights, summarize_detections  # noqa: E402

DB_PATH = ROOT / "predictions.db"
SECRET_KEY_PATH = ROOT / ".flask_secret"
FRONTEND_DIST = ROOT / "DTI" / "dist"
DEFAULT_CONF = 0.25
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


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


def load_secret_key() -> str:
    """Session-signing key: FLASK_SECRET_KEY in .env, else a random key kept in .flask_secret (not committed)."""
    env_key = os.environ.get("FLASK_SECRET_KEY", "").strip()
    if env_key:
        return env_key
    if SECRET_KEY_PATH.is_file():
        return SECRET_KEY_PATH.read_text(encoding="utf-8").strip()
    key = secrets.token_hex(32)
    SECRET_KEY_PATH.write_text(key, encoding="utf-8")
    return key


app = Flask(__name__)
CORS(app)
app.json = StrictJSONProvider(app)
app.secret_key = load_secret_key()
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL
        )
        """
    )
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
            recommendation TEXT NOT NULL,
            user_id INTEGER REFERENCES users (id)
        )
        """
    )
    # Databases created by the Milestone 2 build have no user_id column yet.
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(predictions)")}
    if "user_id" not in columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN user_id INTEGER REFERENCES users (id)")
    conn.commit()
    conn.close()


def current_user() -> dict[str, Any] | None:
    user_id = session.get("user_id")
    if not user_id:
        return None
    conn = get_db_connection()
    row = conn.execute("SELECT id, name, email FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    if row is None:
        session.clear()
        return None
    return {"id": row["id"], "name": row["name"], "email": row["email"]}


def save_prediction_history(payload: dict[str, Any], user_id: int) -> None:
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO predictions
            (created_at, plant, disease, confidence_pct, severity_pct, severity_level, health_pct,
             recommendation, user_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            user_id,
        ),
    )
    conn.commit()
    conn.close()


WEIGHTS_PATH = find_weights(ROOT)
MODEL = YOLO(str(WEIGHTS_PATH)) if WEIGHTS_PATH else None
CLS_WEIGHTS_PATH = find_classifier_weights(ROOT)
CLASSIFIER = YOLO(str(CLS_WEIGHTS_PATH)) if CLS_WEIGHTS_PATH else None
if CLASSIFIER is None:
    print(f"[INFO] No classifier at {DEFAULT_CLS_WEIGHTS_REL}; using the detector only.")
if MODEL is None and CLASSIFIER is None:
    print(
        f"[WARN] No trained weights found. Train with `python train_classifier.py` ({DEFAULT_CLS_WEIGHTS_REL}) "
        f"or `python train_yolo.py` ({DEFAULT_WEIGHTS_REL}). /api/predict will return 503 until then."
    )


def parse_conf(raw: Any) -> float:
    """Confidence threshold from the request: a number in [0, 1], else the default."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_CONF
    if math.isnan(value):
        return DEFAULT_CONF
    return min(1.0, max(0.0, value))


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
            "modelLoaded": MODEL is not None or CLASSIFIER is not None,
            "classifier": CLS_WEIGHTS_PATH.name if CLS_WEIGHTS_PATH else None,
            "detector": WEIGHTS_PATH.name if WEIGHTS_PATH else None,
            "classes": len((CLASSIFIER or MODEL).names) if (CLASSIFIER or MODEL) is not None else 0,
            "chatbot": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
        }
    )


def detector_only_prediction(image: Image.Image, confidence_threshold: float) -> dict[str, Any]:
    """Milestone 2 path, used when no classifier has been trained."""
    result = MODEL.predict(source=image, conf=confidence_threshold, verbose=False)[0]
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
        return {
            "status": "no_detection",
            "message": (
                f"No leaf condition detected above {confidence_threshold:.0%} confidence. "
                "Try a closer, well-lit photo of a single leaf."
            ),
            "annotated": annotated,
            "heatmap": None,
        }
    return {"status": "detected", **detections[0], "top3": [], "annotated": annotated, "heatmap": None}


@app.post("/api/predict")
def api_predict() -> Any:
    if MODEL is None and CLASSIFIER is None:
        return jsonify(
            {
                "error": (
                    "No trained model loaded. Run `python train_classifier.py` first "
                    "(or set PLANTDOC_CLS_WEIGHTS in .env) and restart the server."
                )
            }
        ), 503

    body = request.get_json(silent=True) or {}
    image_data_url = str(body.get("imageDataUrl", "")).strip()
    confidence_threshold = parse_conf(body.get("conf"))
    if not image_data_url:
        return jsonify({"error": "imageDataUrl is required"}), 400

    try:
        image = decode_data_url(image_data_url)
    except Exception:
        return jsonify({"error": "Could not read the uploaded file as an image."}), 400

    try:
        started = time.perf_counter()
        if CLASSIFIER is not None:
            result = diagnose(image, CLASSIFIER, MODEL, conf=confidence_threshold)
        else:
            result = detector_only_prediction(image, confidence_threshold)
        inference_ms = (time.perf_counter() - started) * 1000.0

        payload = {k: v for k, v in result.items() if k not in ("annotated", "heatmap")}
        payload["annotatedImage"] = encode_image_to_data_url(result["annotated"])
        payload["heatmapImage"] = encode_image_to_data_url(result["heatmap"]) if result["heatmap"] else None
        payload["inferenceMs"] = round(inference_ms, 1)

        user = current_user()
        payload["saved"] = bool(user and payload["status"] == "detected")
        if payload["saved"]:
            save_prediction_history(payload, user["id"])
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/history")
def history() -> Any:
    user = current_user()
    if user is None:
        return jsonify({"error": "Log in to see your scan history."}), 401
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT id, created_at, plant, disease, confidence_pct, severity_pct, severity_level, health_pct, recommendation
        FROM predictions
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 25
        """,
        (user["id"],),
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


@app.post("/api/auth/signup")
def signup() -> Any:
    body = request.get_json(silent=True) or {}
    name = str(body.get("name", "")).strip()
    email = str(body.get("email", "")).strip().lower()
    password = str(body.get("password", ""))

    if not name or not email or not password:
        return jsonify({"error": "Name, email and password are required."}), 400
    if not EMAIL_RE.match(email):
        return jsonify({"error": "Please enter a valid email address."}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters."}), 400

    conn = get_db_connection()
    try:
        cur = conn.execute(
            "INSERT INTO users (created_at, name, email, password_hash) VALUES (?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), name[:80], email, generate_password_hash(password)),
        )
        conn.commit()
        user_id = cur.lastrowid
    except sqlite3.IntegrityError:
        return jsonify({"error": "An account with this email already exists."}), 409
    finally:
        conn.close()

    session.clear()
    session["user_id"] = user_id
    return jsonify({"user": {"id": user_id, "name": name[:80], "email": email}}), 201


@app.post("/api/auth/login")
def login() -> Any:
    body = request.get_json(silent=True) or {}
    email = str(body.get("email", "")).strip().lower()
    password = str(body.get("password", ""))
    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    conn = get_db_connection()
    row = conn.execute("SELECT id, name, email, password_hash FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    if row is None or not check_password_hash(row["password_hash"], password):
        return jsonify({"error": "Wrong email or password."}), 401

    session.clear()
    session["user_id"] = row["id"]
    return jsonify({"user": {"id": row["id"], "name": row["name"], "email": row["email"]}})


@app.post("/api/auth/logout")
def logout() -> Any:
    session.clear()
    return jsonify({"user": None})


@app.get("/api/auth/me")
def me() -> Any:
    return jsonify({"user": current_user()})


CHAT_SYSTEM = """You are LeafX Assistant, a friendly plant-disease and farming advisor inside a leaf-scanning app.

Rules:
- Only answer questions about plants, crops, plant diseases, pests, soil, fertilizers, irrigation, weather effects on crops and gardening. For anything else, reply: "I can only help with plant and farming questions."
- Be practical and beginner-friendly. Keep answers short: a few sentences or a short bulleted list.
- For diseases, cover cause, symptoms, prevention and treatment when relevant. Prefer integrated pest management; mention organic options before chemical ones.
- When you name a pesticide or fungicide, remind the user to follow the label and check with a local agronomist.
- If the user writes in Hindi or another language, reply in that language.
- If scan context is provided, it comes from an AI model that can be wrong; say so if the user relies on it heavily."""

# Tried in this order. "lite" models answer in 1-2 s and are rarely overloaded; Google retires
# models over time, so any other available flash model is tried after these.
DEFAULT_GEMINI_MODELS = ("gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.1-flash-lite")
FLASH_MODEL_RE = re.compile(r"^gemini-(\d+(?:\.\d+)?)-flash(-lite)?$")
GEMINI_TIMEOUT_MS = 25_000
MAX_MODEL_ATTEMPTS = 4
_gemini_model_cache: dict[str, str] = {}


def _available_flash_models(client: Any) -> list[str]:
    """'gemini-X.Y-flash' / '-flash-lite' models this key can use, newest first (no image/preview variants)."""
    try:
        names = [
            m.name.removeprefix("models/")
            for m in client.models.list()
            if "generateContent" in (getattr(m, "supported_actions", None) or [])
        ]
    except Exception:
        return []
    versioned = [(float(match.group(1)), n) for n in names if (match := FLASH_MODEL_RE.match(n))]
    return [n for _, n in sorted(versioned, reverse=True)]


def _gemini_model_order() -> list[str]:
    """Last model that worked, then GEMINI_MODEL from .env, then the defaults (duplicates removed)."""
    order = [_gemini_model_cache.get("name", ""), os.environ.get("GEMINI_MODEL", "").strip(), *DEFAULT_GEMINI_MODELS]
    return list(dict.fromkeys(m for m in order if m))


def _should_try_next_model(error: Exception) -> bool:
    """Retired (404), overloaded (503), rate-limited (429) or timed-out models: try another one."""
    text = f"{type(error).__name__} {error}"
    markers = ("404", "NOT_FOUND", "no longer available", "503", "UNAVAILABLE", "504", "DEADLINE",
               "429", "RESOURCE_EXHAUSTED", "Timeout", "ReadError")
    return any(m in text for m in markers)


@app.post("/api/chat")
def chat() -> Any:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return jsonify(
            {"error": "The assistant is not configured yet: add GEMINI_API_KEY to the .env file and restart."}
        ), 503

    body = request.get_json(silent=True) or {}
    message = str(body.get("message", "")).strip()
    if not message:
        return jsonify({"error": "message is required"}), 400
    if len(message) > 1000:
        return jsonify({"error": "Please keep your question under 1000 characters."}), 400

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return jsonify({"error": "google-genai is not installed. Run: pip install -r requirements.txt"}), 500

    contents = []
    context = body.get("context") or {}
    if isinstance(context, dict) and context.get("disease"):
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=(
                            "Context from my latest leaf scan (AI estimate): "
                            f"plant {context.get('plant', 'unknown')}, condition {context.get('disease')}, "
                            f"confidence {context.get('confidence', '?')}%, severity {context.get('severityLevel', '?')}."
                        )
                    )
                ],
            )
        )
        contents.append(types.Content(role="model", parts=[types.Part.from_text(text="Got it. How can I help?")]))
    history = body.get("history")
    for turn in (history if isinstance(history, list) else [])[-10:]:
        if not isinstance(turn, dict):
            continue
        role = "model" if turn.get("role") == "assistant" else "user"
        text = str(turn.get("text", ""))[:2000]
        if text:
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=message)]))

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS))
    config = types.GenerateContentConfig(system_instruction=CHAT_SYSTEM, temperature=0.4, max_output_tokens=700)

    response = None
    last_error: Exception | None = None
    tried: list[str] = []
    candidates = _gemini_model_order()
    discovered = False
    while len(tried) < MAX_MODEL_ATTEMPTS:
        if len(tried) == len(candidates):
            if discovered:
                break
            # Our list is used up: ask Google which flash models this key can use right now.
            discovered = True
            candidates += [m for m in _available_flash_models(client) if m not in candidates]
            continue
        model_name = candidates[len(tried)]
        tried.append(model_name)
        try:
            response = client.models.generate_content(model=model_name, contents=contents, config=config)
            _gemini_model_cache["name"] = model_name
            break
        except Exception as e:
            last_error = e
            if "API_KEY_INVALID" in str(e) or "API key not valid" in str(e):
                return jsonify({"error": "Gemini rejected the API key. Check GEMINI_API_KEY in .env and restart."}), 502
            if not _should_try_next_model(e):
                break

    if response is None:
        print(f"[chat] all Gemini attempts failed ({', '.join(tried)}): {type(last_error).__name__}")
        return jsonify(
            {"error": "The AI assistant is busy right now (Google's servers are overloaded). Please try again in a minute."}
        ), 503

    reply = (getattr(response, "text", None) or "").strip()
    return jsonify({"reply": reply or "Sorry, I couldn't come up with an answer. Try rephrasing your question."})


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
    # Client-side routes (/scan, /results, /history, ...) fall back to the SPA entry.
    if (FRONTEND_DIST / "index.html").exists():
        return send_from_directory(FRONTEND_DIST, "index.html")
    return jsonify({"error": "Not found"}), 404


init_db()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
