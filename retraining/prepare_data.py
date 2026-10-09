from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "retraining" / "datasets"


def class_counts(y: np.ndarray) -> dict[str, int]:
    """Return the number of samples for each digit."""
    labels, counts = np.unique(y, return_counts=True)
    return {
        str(int(label)): int(count)
        for label, count in zip(labels, counts)
    }


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate a dataset produced by build_dataset.py."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        if "X" not in data or "y" not in data:
            raise ValueError("Dataset must contain arrays named 'X' and 'y'.")

        X = data["X"]
        y = data["y"]

    if len(X) != len(y):
        raise ValueError(
            f"Number of images ({len(X)}) does not match "
            f"number of labels ({len(y)})."
        )

    if len(y) == 0:
        raise ValueError("Dataset is empty.")

    if X.ndim != 4 or X.shape[1:] != (1, 28, 28):
        raise ValueError(
            f"Expected X shape (N, 1, 28, 28), got {X.shape}."
        )

    if not np.isfinite(X).all():
        raise ValueError("Images contain NaN or infinite values.")

    if not np.isfinite(y).all() or not np.isin(y, np.arange(10)).all():
        raise ValueError("Labels must be digits from 0 through 9.")

    X = X.astype(np.float32, copy=False)
    y = y.astype(np.int64, copy=False)

    if X.min() < 0 or X.max() > 1:
        raise ValueError("Expected normalized image pixels in [0, 1].")

    return X, y


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    allow_small_data: bool,
) -> dict[str, np.ndarray]:
    """Create reproducible data splits."""
    counts = class_counts(y)
    smallest_class = min(counts.values())

    if smallest_class < 3:
        if not allow_small_data:
            raise ValueError(
                "Some classes have fewer than 3 samples: "
                f"{counts}. Collect more labeled examples, or use "
                "--allow-small-data for a training-only pipeline test."
            )

        print(
            "\nWARNING: Small-data mode enabled. All samples will be "
            "assigned to training; validation and test sets will be empty."
        )

        return {
            "X_train": X,
            "y_train": y,
            "X_val": X[:0],
            "y_val": y[:0],
            "X_test": X[:0],
            "y_test": y[:0],
        }

    # Reserve 20% for the final test set.
    X_remaining, X_test, y_remaining, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y,
    )

    # Validation is 20% of the remaining 80% (about 16% overall).
    X_train, X_val, y_train, y_val = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.20,
        random_state=seed,
        stratify=y_remaining,
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare reproducible digit-recognition data splits."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the .npz dataset produced by build_dataset.py",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for splitting (default: 42)",
    )
    parser.add_argument(
        "--allow-small-data",
        action="store_true",
        help="Use training-only mode if stratified splitting is not possible.",
    )
    args = parser.parse_args()

    dataset_path = args.dataset
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    X, y = load_dataset(dataset_path)

    print(f"Loaded {len(y)} samples")
    print(f"Class distribution: {class_counts(y)}")

    splits = split_dataset(
        X=X,
        y=y,
        seed=args.seed,
        allow_small_data=args.allow_small_data,
    )

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    output_path = DATASET_DIR / f"prepared_split_seed{args.seed}.npz"
    summary_path = DATASET_DIR / f"prepared_split_seed{args.seed}_summary.json"

    np.savez_compressed(output_path, **splits)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_dataset": str(dataset_path.resolve()),
        "seed": args.seed,
        "mode": (
            "training_only"
            if len(splits["y_val"]) == 0
            else "stratified_train_validation_test"
        ),
        "split_sizes": {
            name.removeprefix("y_"): len(splits[name])
            for name in ("y_train", "y_val", "y_test")
        },
        "class_counts": {
            name.removeprefix("y_"): class_counts(splits[name])
            for name in ("y_train", "y_val", "y_test")
        },
        "warning": (
            "No validation or test evaluation is possible in training-only mode."
            if len(splits["y_val"]) == 0
            else None
        ),
        "split_file": output_path.name,
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nPrepared splits:")
    for name in ("train", "val", "test"):
        labels = splits[f"y_{name}"]
        print(
            f"  {name:5}: {len(labels):4} samples | "
            f"{class_counts(labels)}"
        )

    print(f"\nSplit dataset: {output_path}")
    print(f"Summary:       {summary_path}")


if __name__ == "__main__":
    main()