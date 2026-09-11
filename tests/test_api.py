import io

from PIL import Image

from app.main import app


def test_home():
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200


def test_predict():
    client = app.test_client()

    image = Image.new("L", (28, 28), 0)

    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    response = client.post(
        "/predict",
        data={
            "image": (
                image_bytes,
                "digit.png"
            )
        },
        content_type="multipart/form-data"
    )

    assert response.status_code == 200

    data = response.json

    assert "prediction" in data
    assert "confidence" in data