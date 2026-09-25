import io

from PIL import Image

from app.server import app, initialize_model


def setup_module():
    initialize_model("onnx-int8")


def test_home():
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200


def test_predict(monkeypatch):
    client = app.test_client()

    # Prevent the test from depending on a real Redis server
    class MockCache:
        def get(self, key):
            return None

        def setex(self, key, timeout, value):
            pass

    monkeypatch.setattr("app.server.cache", MockCache())

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