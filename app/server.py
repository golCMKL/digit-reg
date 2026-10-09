"""
Serve the digit CNN behind POST /predict (same API as the original Flask app).

Usage:
    python server.py --backend torch     --port 5000   # uses model.load_model()
    python server.py --backend onnx      --port 5001   # digit_cnn.onnx (+ .onnx.data)
    python server.py --backend onnx-int8 --port 5001   # digit_cnn_int8.onnx
"""
import argparse
import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import psutil
import psycopg
import redis
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError

IMG_SIZE = 28  # preprocessing matches the original app: grayscale -> 28x28 -> /255 (no mean/std)

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"

PROJECT_ROOT = APP_DIR.parent

IMAGE_DIR = Path(
    os.getenv(
        "IMAGE_DIR",
        str(PROJECT_ROOT / "data" / "collected_images")
    )
)

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
predict_fn = None

cache = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=6379,
    decode_responses=True
)

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "digit_db"),
    "user": os.getenv("DB_USER", "digit_user"),
    "password": os.getenv("DB_PASSWORD", "digit_password"),
}

def init_db():
    with psycopg.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    request_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    prediction INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    cached BOOLEAN NOT NULL
                )
            """)
        conn.commit()


def preprocess(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("L").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr[None, None, :, :]  # (1, 1, 28, 28)

def save_uploaded_image(file_bytes: bytes, image_hash: str) -> str:
    """Save an uploaded image once and return its project-relative path."""
    with Image.open(io.BytesIO(file_bytes)) as image:
        image_format = (image.format or "PNG").upper()

    extensions = {
        "PNG": ".png",
        "JPEG": ".jpg",
        "WEBP": ".webp",
        "BMP": ".bmp",
        "GIF": ".gif",
        "TIFF": ".tiff",
    }

    extension = extensions.get(image_format, ".img")
    filename = f"{image_hash}{extension}"
    destination = IMAGE_DIR / filename

    # Identical uploads share one stored image.
    if not destination.exists():
        destination.write_bytes(file_bytes)

    return f"data/collected_images/{filename}"


def softmax_np(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def build_torch():
    import torch
    import torch.nn.functional as F
    from model import load_model  # only needed for the PyTorch backend

    model = load_model()
    model.eval()

    def run(x: np.ndarray):
        with torch.no_grad():
            probs = F.softmax(model(torch.from_numpy(x)), dim=1)
        pred = int(probs.argmax(dim=1).item())
        return pred, float(probs[0, pred].item())

    return run


def build_onnx(path):
    import onnxruntime as ort

    # digit_cnn.onnx.data must sit in the same folder as digit_cnn.onnx
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    def run(x: np.ndarray):
        probs = softmax_np(sess.run(None, {input_name: x})[0])
        pred = int(probs.argmax(axis=1)[0])
        return pred, float(probs[0, pred])

    return run

def initialize_model(backend):
    global predict_fn

    if backend == "torch":
        predict_fn = build_torch()

    elif backend == "onnx":
        predict_fn = build_onnx(str(MODEL_DIR / "digit_cnn.onnx"))

    elif backend == "onnx-int8":
        
        predict_fn = build_onnx(str(MODEL_DIR / "digit_cnn_int8.onnx"))
    else:
        raise ValueError(f"Unknown backend: {backend}")

    predict_fn(
        np.zeros(
            (1, 1, IMG_SIZE, IMG_SIZE),
            dtype=np.float32
        )
    )

def log_prediction(
    prediction,
    confidence,
    cached,
    latency_ms,
    error=None,
    input_mean=None,
    input_std=None,
    input_min=None,
    input_max=None,
    nonzero_ratio=None,
    image_path=None,
    image_hash=None,
):
    with psycopg.connect(**DB_CONFIG) as conn, conn.cursor() as cur:
        cur.execute("""
                INSERT INTO predictions (
                    prediction,
                    confidence,
                    cached,
                    latency_ms,
                    error,
                    input_mean,
                    input_std,
                    input_min,
                    input_max,
                    nonzero_ratio,
                    image_path,
                    image_hash
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s
                )
            """, (
            prediction,
            confidence,
            cached,
            latency_ms,
            error,
            input_mean,
            input_std,
            input_min,
            input_max,
            nonzero_ratio,
            image_path,
            image_hash,
        ))


@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.perf_counter()

    if "image" not in request.files:
        log_prediction(
            prediction=None,
            confidence=None,
            cached=False,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            error="No image provided",
        )
        return jsonify({"error": "No image provided"}), 400

    file_bytes = request.files["image"].read()

    if not file_bytes:
        log_prediction(
            prediction=None,
            confidence=None,
            cached=False,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            error="Empty image file",
        )
        return jsonify({"error": "Empty image file"}), 400

    image_hash = hashlib.sha256(file_bytes).hexdigest()

    # Validate and preprocess the image before saving it.
    try:
        x = preprocess(file_bytes)
        image_path = save_uploaded_image(file_bytes, image_hash)
    except (UnidentifiedImageError, ValueError, OSError):
        log_prediction(
            prediction=None,
            confidence=None,
            cached=False,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            error="Invalid image",
            image_hash=image_hash,
        )
        return jsonify({"error": "Invalid image"}), 400

    # Collect statistics for cache hits as well as cache misses.
    input_mean = float(x.mean())
    input_std = float(x.std())
    input_min = float(x.min())
    input_max = float(x.max())
    nonzero_ratio = float((x > 0).mean())

    cache_key = f"prediction:{image_hash}"

    try:
        cached_result = cache.get(cache_key)
    except redis.RedisError:
        # If Redis is unavailable, inference can still proceed.
        cached_result = None

    if cached_result is not None:
        result = json.loads(cached_result)

        log_prediction(
            prediction=result["prediction"],
            confidence=result["confidence"],
            cached=True,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            input_mean=input_mean,
            input_std=input_std,
            input_min=input_min,
            input_max=input_max,
            nonzero_ratio=nonzero_ratio,
            image_path=image_path,
            image_hash=image_hash,
        )

        result["cached"] = True
        return jsonify(result)

    try:
        pred, conf = predict_fn(x)
    except (ValueError, RuntimeError):
        log_prediction(
            prediction=None,
            confidence=None,
            cached=False,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            error="Inference failed",
            input_mean=input_mean,
            input_std=input_std,
            input_min=input_min,
            input_max=input_max,
            nonzero_ratio=nonzero_ratio,
            image_path=image_path,
            image_hash=image_hash,
        )
        return jsonify({"error": "Inference failed"}), 500

    result = {
        "prediction": pred,
        "confidence": conf,
        "cached": False,
    }

    try:
        cache.setex(cache_key, 3600, json.dumps(result))
    except redis.RedisError:
        pass

    log_prediction(
        prediction=pred,
        confidence=conf,
        cached=False,
        latency_ms=(time.perf_counter() - start_time) * 1000,
        input_mean=input_mean,
        input_std=input_std,
        input_min=input_min,
        input_max=input_max,
        nonzero_ratio=nonzero_ratio,
        image_path=image_path,
        image_hash=image_hash,
    )

    return jsonify(result)

@app.route("/cache-health", methods=["GET"])
def cache_health():
    return jsonify({
        "cache_available": cache.ping()
    })

@app.route("/memory", methods=["GET"])
def memory():
    rss = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    return jsonify({"rss_mb": rss})


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Prediction API is running"})


backend = os.getenv("MODEL_BACKEND")

if backend:
    initialize_model(backend)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backend",
        choices=["torch", "onnx", "onnx-int8"],
        required=True
    )

    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")

    parser.add_argument("--init-db", action="store_true")

    args = parser.parse_args()

    if args.init_db:
        init_db()
        print("Database initialized")
    else:
        initialize_model(args.backend)
        print(f"Loaded {args.backend} model successfully")
        app.run(
            host=args.host,
            port=args.port,
            debug=False,
            threaded=False
        )

    initialize_model(args.backend)

    print(f"Loaded {args.backend} model successfully")

    app.run(
        host=args.host,
        port=args.port,
        debug=False,
        threaded=False
    )