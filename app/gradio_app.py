import io

import gradio as gr
import numpy as np
import requests
from PIL import Image

FLASK_URL = "http://127.0.0.1:5000/predict"

def predict_digit(image):
    if image is None:
        return "Please draw a digit."

    if isinstance(image, dict):
        image = image["composite"]

    image = image.convert("L")

    image = np.array(image).astype("float32")

    image = np.where(image < 200, 255, 0).astype("uint8")

    image = Image.fromarray(image)

    # Resize to 28x28
    image = image.resize((28, 28))

    processed_image = image.copy()
    
    # Send the image to Flask
    image_bytes = io.BytesIO()
    processed_image.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    response = requests.post(
        FLASK_URL,
        files={"image": ("digit.png", image_bytes, "image/png")}
    )

    if response.status_code != 200:
        return f"Error: {response.text}"

    result = response.json()

    result_text = (
        f"Prediction: {result['prediction']}\n"
        f"Confidence: {result['confidence']:.2%}"
    )

    return result_text, processed_image

demo = gr.Interface(
    fn=predict_digit,

    inputs=gr.ImageEditor(
        type="pil",
        image_mode="L",
        sources=[],
        layers=False,
        label="Draw a digit",
        canvas_size=(800, 800),
    ),

    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Image(
            type="pil",
            image_mode="L",
            label="Image sent to Flask",
            width=280,
            height=280
        )
    ],


    title="Handwritten Digit Classifier"
)

demo.launch()