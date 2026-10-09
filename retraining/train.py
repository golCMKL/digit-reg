
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "retraining" / "runs"


class DigitCNN(nn.Module):
    """CNN for 28x28 grayscale digit classification."""

    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_splits(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Prepared dataset not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        required = {
            "X_train", "y_train",
            "X_val", "y_val",
            "X_test", "y_test",
        }

        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Prepared dataset is missing: {sorted(missing)}")

        splits = {name: data[name] for name in required}

    for split in ("train", "val", "test"):
        X = splits[f"X_{split}"]
        y = splits[f"y_{split}"]

        if len(X) != len(y):
            raise ValueError(f"{split}: image and label counts differ.")

        if X.size and (
            X.ndim != 4
            or X.shape[1:] != (1, 28, 28)
        ):
            raise ValueError(
                f"{split}: expected (N, 1, 28, 28), got {X.shape}."
            )

        if y.size and not np.isin(y, np.arange(10)).all():
            raise ValueError(f"{split}: labels must be digits 0 through 9.")

        if X.size and (
            not np.isfinite(X).all()
            or X.min() < 0
            or X.max() > 1
        ):
            raise ValueError(
                f"{split}: image values must be finite and in [0, 1]."
            )

        splits[f"X_{split}"] = X.astype(np.float32, copy=False)
        splits[f"y_{split}"] = y.astype(np.int64, copy=False)

    if len(splits["y_train"]) == 0:
        raise ValueError("Training set is empty.")

    return splits


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            count = labels.size(0)
            total_loss += loss.item() * count
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += count

    return (
        total_loss / total_samples,
        total_correct / total_samples,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a candidate CNN for digit recognition."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive.")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")

    dataset_path = args.dataset
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path

    set_seed(args.seed)
    splits = load_splits(dataset_path)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print(f"Training samples: {len(splits['y_train'])}")
    print(f"Validation samples: {len(splits['y_val'])}")
    print(f"Test samples: {len(splits['y_test'])}")

    train_loader = make_loader(
        splits["X_train"],
        splits["y_train"],
        args.batch_size,
        shuffle=True,
    )

    has_validation = len(splits["y_val"]) > 0
    val_loader = (
        make_loader(
            splits["X_val"],
            splits["y_val"],
            args.batch_size,
            shuffle=False,
        )
        if has_validation
        else None
    )

    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    checkpoint_path = run_dir / "candidate_model.pth"
    history = []
    best_metric = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
        }

        if val_loader is not None:
            val_loss, val_accuracy = run_epoch(
                model,
                val_loader,
                criterion,
                device,
            )
            record["val_loss"] = val_loss
            record["val_accuracy"] = val_accuracy

            # Select the checkpoint by validation loss.
            metric = val_loss
        else:
            # Training-only mode: no validation-based model selection.
            metric = train_loss

        history.append(record)

        print(
            f"Epoch {epoch:03}/{args.epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"train accuracy: {train_accuracy:.4f}"
            + (
                f" | val loss: {record['val_loss']:.4f}"
                f" | val accuracy: {record['val_accuracy']:.4f}"
                if val_loader is not None
                else " | no validation"
            )
        )

        if metric < best_metric:
            best_metric = metric
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": "DigitCNN",
                    "epoch": epoch,
                    "selection_metric": (
                        "validation_loss"
                        if has_validation
                        else "training_loss"
                    ),
                    "selection_metric_value": metric,
                    "seed": args.seed,
                },
                checkpoint_path,
            )

    training_only = not has_validation

    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_dataset": str(dataset_path.resolve()),
        "architecture": "DigitCNN",
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": str(device),
        "training_only": training_only,
        "checkpoint_selection": (
            "lowest validation loss"
            if has_validation
            else "lowest training loss; not a generalization metric"
        ),
        "test_evaluated": False,
        "history": history,
        "checkpoint": checkpoint_path.name,
    }

    (run_dir / "training_summary.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"\nCandidate checkpoint: {checkpoint_path}")
    print(f"Training summary:     {run_dir / 'training_summary.json'}")

    if training_only:
        print(
            "\nWARNING: This run had no validation data. "
            "The saved checkpoint is for pipeline testing only; "
            "it has not been selected using validation performance."
        )


if __name__ == "__main__":
    main()