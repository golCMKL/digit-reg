
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from model import load_model
from PIL import Image

app = Flask(__name__)

model = load_model()


print("Loaded model successfully")

#BASE_DIR = Path(__file__).resolve().parent
#model = joblib.load(BASE_DIR / "mlp_model.pkl")



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

    image = torch.tensor(image).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(image)

    probabilities = F.softmax(output, dim=1)

    prediction = probabilities.argmax(dim=1).item()

    confidence = probabilities[0, prediction].item()

    return jsonify({
        "prediction": int(prediction),
        "confidence": float(confidence)
    })

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Prediction API is running"
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)