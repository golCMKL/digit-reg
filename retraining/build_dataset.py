import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psycopg
from PIL import Image, UnidentifiedImageError

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "retraining" / "datasets"

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5433")),
    "dbname": os.getenv("DB_NAME", "digit_db"),
    "user": os.getenv("DB_USER", "digit_user"),
    "password": os.getenv("DB_PASSWORD", "digit_password"),
}

IMAGE_SIZE = (28, 28)


def load_labeled_samples():
    """Load verified labels and their associated prediction metadata."""
    query = """
        SELECT
            l.prediction_id,
            l.image_path,
            l.true_label,
            p.prediction,
            p.confidence,
            p.image_hash
        FROM labeled_samples AS l
        JOIN predictions AS p
            ON p.id = l.prediction_id
        ORDER BY l.created_at, l.id
    """

    with psycopg.connect(**DB_CONFIG) as conn, conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()


def resolve_image_path(image_path):
    """Resolve a database image path safely within the project directory."""
    relative_path = Path(image_path)

    if relative_path.is_absolute():
        raise ValueError("Expected a relative image path")

    full_path = (PROJECT_ROOT / relative_path).resolve()

    if not full_path.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("Image path points outside the project directory")

    return full_path


def preprocess_image(image_path):
    """Match the API's grayscale, 28x28, [0, 1] preprocessing."""
    with Image.open(image_path) as image:
        image = image.convert("L")
        image = image.resize(IMAGE_SIZE)
        pixels = np.asarray(image, dtype=np.float32) / 255.0

    # CNN input layout: (channels, height, width)
    return pixels[np.newaxis, :, :]


def build_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    samples = load_labeled_samples()

    if not samples:
        raise RuntimeError(
            "No labeled samples found. Label some images first."
        )

    images = []
    labels = []
    metadata = []
    seen_images = {}
    skipped = 0

    for row in samples:
        prediction_id, image_path, true_label, prediction, confidence, image_hash = row

        if true_label is None or not 0 <= true_label <= 9:
            print(f"Skipping prediction {prediction_id}: invalid label")
            skipped += 1
            continue

        try:
            full_path = resolve_image_path(image_path)

            if not full_path.is_file():
                raise FileNotFoundError(full_path)

            pixels = preprocess_image(full_path)

        except (
            OSError,
            ValueError,
            UnidentifiedImageError,
        ) as exc:
            print(f"Skipping prediction {prediction_id}: {exc}")
            skipped += 1
            continue

        # Avoid counting the same image file more than once.
        # Repeated predictions may refer to the same cached image.
        image_key = image_hash or str(full_path)

        if image_key in seen_images:
            existing_label = seen_images[image_key]

            if existing_label != true_label:
                raise ValueError(
                    f"Conflicting labels for image {image_key}: "
                    f"{existing_label} and {true_label}"
                )

            continue

        seen_images[image_key] = true_label

        images.append(pixels)
        labels.append(int(true_label))

        metadata.append({
            "prediction_id": prediction_id,
            "image_path": image_path,
            "image_hash": image_hash or "",
            "true_label": int(true_label),
            "model_prediction": prediction,
            "confidence": confidence,
        })

    if not images:
        raise RuntimeError(
            "No usable images remain. Check the image paths and labels."
        )

    # Shapes: X = (N, 1, 28, 28), y = (N,)
    X = np.stack(images).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dataset_path = OUTPUT_DIR / f"labeled_dataset_{timestamp}.npz"
    metadata_path = OUTPUT_DIR / f"labeled_metadata_{timestamp}.csv"
    summary_path = OUTPUT_DIR / f"dataset_summary_{timestamp}.json"

    # Save the training arrays
    np.savez_compressed(dataset_path, X=X, y=y)

    # Save provenance for every included image
    with metadata_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)

    class_counts = {
        str(digit): int(np.sum(y == digit))
        for digit in range(10)
    }

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sample_count": len(y),
        "skipped_count": skipped,
        "image_shape": list(X.shape),
        "image_dtype": str(X.dtype),
        "label_dtype": str(y.dtype),
        "pixel_min": float(X.min()),
        "pixel_max": float(X.max()),
        "class_counts": class_counts,
        "dataset_file": dataset_path.name,
        "metadata_file": metadata_path.name,
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nDataset built successfully!")
    print(f"Usable samples : {len(y)}")
    print(f"Skipped samples: {skipped}")
    print(f"X shape        : {X.shape}")
    print(f"y shape        : {y.shape}")
    print(f"Pixel range    : {X.min():.3f} to {X.max():.3f}")
    print(f"Dataset        : {dataset_path}")
    print(f"Metadata       : {metadata_path}")
    print(f"Summary        : {summary_path}")
    print(f"Class counts   : {class_counts}")


if __name__ == "__main__":
    build_dataset()