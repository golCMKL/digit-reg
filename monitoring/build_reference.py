import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "app" / "benchmark_images"
OUTPUT_FILE = PROJECT_ROOT / "monitoring" / "reference_stats.json"

IMAGE_SIZE = 28


def extract_label(filename):
    """Extract label from filenames such as 000_label7.png."""
    match = re.search(r"_label(\d+)", filename)
    return int(match.group(1)) if match else None


def process_image(path):
    """Match the API's grayscale, resize, and normalization steps."""
    with Image.open(path) as image:
        image = image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
        pixels = np.asarray(image, dtype=np.float32) / 255.0

    return {
        "input_mean": float(pixels.mean()),
        "input_std": float(pixels.std()),
        "input_min": float(pixels.min()),
        "input_max": float(pixels.max()),
        "nonzero_ratio": float((pixels > 0).mean()),
        "pixel_values": pixels.ravel(),
    }


def main():
    image_paths = sorted(IMAGE_DIR.glob("*.png"))

    if not image_paths:
        raise FileNotFoundError(
            f"No PNG images found in {IMAGE_DIR}"
        )

    features = {
        "input_mean": [],
        "input_std": [],
        "input_min": [],
        "input_max": [],
        "nonzero_ratio": [],
    }
    labels = []
    pixel_values = []
    skipped = 0

    for path in image_paths:
        try:
            result = process_image(path)
        except (OSError, ValueError) as exc:
            print(f"Skipping {path.name}: {exc}")
            skipped += 1
            continue

        for key, values in features.items():
            values.append(result[key])

        pixel_values.extend(result["pixel_values"].tolist())

        label = extract_label(path.name)
        if label is not None:
            labels.append(label)

    if not pixel_values:
        raise RuntimeError("No images could be processed.")

    # Fixed intensity bins shared across all images.
    pixel_counts, pixel_edges = np.histogram(
        pixel_values,
        bins=20,
        range=(0.0, 1.0),
    )

    reference = {
        "image_count": len(features["input_mean"]),
        "skipped_images": skipped,
        "image_size": IMAGE_SIZE,
        "preprocessing": "grayscale, resize to 28x28, divide by 255",
        "features": features,
        "pixel_histogram": {
            "counts": pixel_counts.tolist(),
            "bin_edges": pixel_edges.tolist(),
        },
        "label_counts": {
            str(label): int(count)
            for label, count in zip(
                *np.unique(labels, return_counts=True)
            )
        },
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(reference, indent=2),
        encoding="utf-8",
    )

    print(f"Processed images: {reference['image_count']}")
    print(f"Skipped images:   {skipped}")
    print(f"Reference saved:  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()