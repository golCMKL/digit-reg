from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "mlp_model.pkl")

print("MLP model loaded successfully")


@app.route('/predict', methods=["POST"])
def predict(): 
    if "image" not in request.files:
        return jsonify({
            "error: No Image provided"
        }), 400
    file = request.files["image"]

    image = Image.open(file).convert("L")

    image = image.resize((28, 28))

    image = np.array(image).astype("float32")

    image = image /255.0

    image = image.reshape(1,784)

    prediction = model.predict(image)[0]

    probabilities = model.predict_proba(image)[0]

    return jsonify({
        "prediction": int(prediction),
        "confidence": float(probabilities[prediction])
    })

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "MLP prediction API is running"
    })

if __name__ == '__main__':
    app.run()