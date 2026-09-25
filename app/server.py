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

import numpy as np
import psutil
import redis
from flask import Flask, jsonify, request
from PIL import Image

IMG_SIZE = 28  # preprocessing matches the original app: grayscale -> 28x28 -> /255 (no mean/std)

app = Flask(__name__)
predict_fn = None

cache = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=6379,
    decode_responses=True
)


def preprocess(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("L").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr[None, None, :, :]  # (1, 1, 28, 28)


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
        model_path = os.path.join(
            os.path.dirname(__file__),
            "digit_cnn.onnx"
        )
        predict_fn = build_onnx(model_path)

    elif backend == "onnx-int8":
        model_path = os.path.join(
        os.path.dirname(__file__),
        "digit_cnn_int8.onnx"
    )
        predict_fn = build_onnx(model_path)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    predict_fn(
        np.zeros(
            (1, 1, IMG_SIZE, IMG_SIZE),
            dtype=np.float32
        )
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file_bytes = request.files["image"].read()

    # Create a unique key based on the image
    image_hash = hashlib.sha256(file_bytes).hexdigest()
    cache_key = f"prediction:{image_hash}"

    cached = cache.get(cache_key)

    if cached is not None:
        result = json.loads(cached)
        result["cached"] = True
        return jsonify(result)

    # Cache miss -> actually run the model
    x = preprocess(file_bytes)
    pred, conf = predict_fn(x)

    result = {
        "prediction": pred,
        "confidence": conf
    }

    # Store result for 1 hour
    cache.setex(
        cache_key,
       3600,
        json.dumps(result)
    )

    result["cached"] = False

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

    args = parser.parse_args()

    initialize_model(args.backend)

    print(f"Loaded {args.backend} model successfully")

    app.run(
        host=args.host,
        port=args.port,
        debug=False,
        threaded=False
    )