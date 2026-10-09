
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from train import DigitCNN

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch and ONNX model outputs."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional prepared dataset containing X_train.",
    )
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()

    if args.atol < 0 or args.rtol < 0:
        parser.error("--atol and --rtol must be non-negative.")

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path

    checkpoint_path = resolve(args.checkpoint)
    onnx_path = resolve(args.onnx)

    for path in (checkpoint_path, onnx_path):
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint.get("architecture") != "DigitCNN":
        raise ValueError("Expected a DigitCNN checkpoint.")

    model = DigitCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]

    if input_info.name != "images":
        raise ValueError(
            f"Unexpected ONNX input name: {input_info.name}"
        )

    if output_info.name != "logits":
        raise ValueError(
            f"Unexpected ONNX output name: {output_info.name}"
        )

    # Include edge cases and deterministic synthetic inputs.
    rng = np.random.default_rng(42)
    batches = [
        np.zeros((1, 1, 28, 28), dtype=np.float32),
        np.ones((1, 1, 28, 28), dtype=np.float32),
        rng.random((4, 1, 28, 28), dtype=np.float32),
    ]
    batch_names = ["all_zero", "all_one", "random"]

    # Optionally test real preprocessed images as well.
    if args.dataset is not None:
        dataset_path = resolve(args.dataset)

        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}"
            )

        with np.load(dataset_path, allow_pickle=False) as data:
            if "X_train" not in data:
                raise ValueError("Dataset does not contain X_train.")
            X_train = data["X_train"]

        if X_train.ndim != 4 or X_train.shape[1:] != (1, 28, 28):
            raise ValueError(
                f"Expected X_train shape (N, 1, 28, 28), got {X_train.shape}."
            )

        if len(X_train) > 0:
            if (
                not np.isfinite(X_train).all()
                or X_train.min() < 0
                or X_train.max() > 1
            ):
                raise ValueError(
                    "Dataset images must be finite and normalized to [0, 1]."
                )

            batches.append(
                X_train[: min(32, len(X_train))].astype(
                    np.float32, copy=False
                )
            )
            batch_names.append("dataset_samples")

    results = []
    all_passed = True

    for name, images in zip(batch_names, batches):
        with torch.inference_mode():
            torch_logits = model(torch.from_numpy(images)).numpy()

        onnx_logits = session.run(
            ["logits"],
            {"images": images},
        )[0]

        if torch_logits.shape != onnx_logits.shape:
            raise ValueError(
                f"{name}: output shapes differ: "
                f"{torch_logits.shape} vs {onnx_logits.shape}"
            )

        max_abs_diff = float(
            np.max(np.abs(torch_logits - onnx_logits))
        )
        mean_abs_diff = float(
            np.mean(np.abs(torch_logits - onnx_logits))
        )

        torch_predictions = torch_logits.argmax(axis=1)
        onnx_predictions = onnx_logits.argmax(axis=1)
        predictions_match = bool(
            np.array_equal(torch_predictions, onnx_predictions)
        )
        logits_close = bool(
            np.allclose(
                torch_logits,
                onnx_logits,
                atol=args.atol,
                rtol=args.rtol,
            )
        )

        passed = logits_close and predictions_match
        all_passed = all_passed and passed

        results.append({
            "batch": name,
            "sample_count": len(images),
            "max_absolute_logit_difference": max_abs_diff,
            "mean_absolute_logit_difference": mean_abs_diff,
            "predictions_match": predictions_match,
            "logits_within_tolerance": logits_close,
            "passed": passed,
            "pytorch_predictions": torch_predictions.tolist(),
            "onnx_predictions": onnx_predictions.tolist(),
        })

        print(
            f"{name}: "
            f"max diff={max_abs_diff:.8g}, "
            f"mean diff={mean_abs_diff:.8g}, "
            f"predictions match={predictions_match}, "
            f"PASS={passed}"
        )

    report = {
        "validated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint_path.resolve()),
        "onnx_model": str(onnx_path.resolve()),
        "onnxruntime_version": ort.__version__,
        "providers": session.get_providers(),
        "atol": args.atol,
        "rtol": args.rtol,
        "overall_passed": all_passed,
        "results": results,
    }

    report_path = onnx_path.with_name("validation_report.json")
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport: {report_path}")
    print(f"Overall: {'PASS' if all_passed else 'FAIL'}")

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()