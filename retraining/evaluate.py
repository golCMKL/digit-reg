
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader, TensorDataset
from train import DigitCNN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "retraining" / "runs"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a candidate CNN on a held-out test set."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")

    dataset_path = args.dataset
    checkpoint_path = args.checkpoint

    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with np.load(dataset_path, allow_pickle=False) as data:
        if "X_test" not in data or "y_test" not in data:
            raise ValueError("Dataset must contain X_test and y_test.")

        X_test = data["X_test"]
        y_test = data["y_test"]

    if len(X_test) == 0 or len(y_test) == 0:
        raise ValueError(
            "Test set is empty. Collect sufficient labeled data and "
            "prepare a real held-out test split before evaluating."
        )

    if len(X_test) != len(y_test):
        raise ValueError("Test image and label counts differ.")

    if X_test.ndim != 4 or X_test.shape[1:] != (1, 28, 28):
        raise ValueError(
            f"Expected X_test shape (N, 1, 28, 28), got {X_test.shape}."
        )

    if (
        not np.isfinite(X_test).all()
        or X_test.min() < 0
        or X_test.max() > 1
    ):
        raise ValueError("Test pixels must be finite and in [0, 1].")

    if not np.isin(y_test, np.arange(10)).all():
        raise ValueError("Test labels must be digits from 0 through 9.")

    X_test = X_test.astype(np.float32, copy=False)
    y_test = y_test.astype(np.int64, copy=False)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    if checkpoint.get("architecture") != "DigitCNN":
        raise ValueError("Checkpoint architecture is not DigitCNN.")

    model = DigitCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_test),
            torch.from_numpy(y_test),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )

    all_predictions = []

    with torch.inference_mode():
        for images, _ in loader:
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1)
            all_predictions.extend(predictions.cpu().tolist())

    y_pred = np.asarray(all_predictions, dtype=np.int64)

    labels = list(range(10))
    accuracy = float(accuracy_score(y_test, y_pred))

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=labels,
    )

    run_dir = checkpoint_path.parent
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = run_dir / f"evaluation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    results = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(dataset_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "test_samples": len(y_test),
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix_labels": labels,
        "confusion_matrix": matrix.tolist(),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "warning": (
            "This candidate uses the newly defined DigitCNN architecture. "
            "Compare against the production model only after confirming "
            "compatible architecture and preprocessing."
        ),
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    with (output_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["actual\\predicted", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])

    with (output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["actual_label", "predicted_label", "correct"])
        for actual, predicted in zip(y_test, y_pred):
            writer.writerow([
                int(actual),
                int(predicted),
                bool(actual == predicted),
            ])

    print(f"Test samples: {len(y_test)}")
    print(f"Accuracy:     {accuracy:.4f}")
    print("\nPer-digit metrics:")

    for digit in labels:
        item = report[str(digit)]
        print(
            f"Digit {digit}: "
            f"precision={item['precision']:.3f}, "
            f"recall={item['recall']:.3f}, "
            f"F1={item['f1-score']:.3f}, "
            f"support={int(item['support'])}"
        )

    print(f"\nEvaluation results: {output_dir}")


if __name__ == "__main__":
    main()