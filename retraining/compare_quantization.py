
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def benchmark(
    session: ort.InferenceSession,
    input_name: str,
    image: np.ndarray,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    """Measure single-image CPU inference latency."""
    inputs = {input_name: image}

    for _ in range(warmup):
        session.run(None, inputs)

    timings = []

    for _ in range(iterations):
        start = time.perf_counter()
        session.run(None, inputs)
        timings.append((time.perf_counter() - start) * 1000)

    values = np.asarray(timings)

    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(np.min(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare FP32 and INT8 ONNX models."
    )
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--int8", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    if args.warmup < 0 or args.iterations < 1:
        parser.error("--warmup must be non-negative and --iterations positive.")

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path

    fp32_path = resolve(args.fp32)
    int8_path = resolve(args.int8)
    dataset_path = resolve(args.dataset)

    for path in (fp32_path, int8_path, dataset_path):
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

    with np.load(dataset_path, allow_pickle=False) as data:
        if "X_train" not in data or "y_train" not in data:
            raise ValueError("Dataset must contain X_train and y_train.")

        X = data["X_train"].astype(np.float32, copy=False)
        y = data["y_train"].astype(np.int64, copy=False)

    if len(X) == 0:
        raise ValueError("No training images available for comparison.")
    if len(X) != len(y):
        raise ValueError("Image and label counts differ.")
    if X.ndim != 4 or X.shape[1:] != (1, 28, 28):
        raise ValueError(f"Unexpected image shape: {X.shape}")
    if (
        not np.isfinite(X).all()
        or X.min() < 0
        or X.max() > 1
    ):
        raise ValueError("Images must be finite and normalized to [0, 1].")

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1

    fp32_session = ort.InferenceSession(
        str(fp32_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    int8_session = ort.InferenceSession(
        str(int8_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    fp32_input = fp32_session.get_inputs()[0].name
    int8_input = int8_session.get_inputs()[0].name
    fp32_output = fp32_session.get_outputs()[0].name
    int8_output = int8_session.get_outputs()[0].name

    fp32_logits = fp32_session.run(
        [fp32_output], {fp32_input: X}
    )[0]
    int8_logits = int8_session.run(
        [int8_output], {int8_input: X}
    )[0]

    if fp32_logits.shape != int8_logits.shape:
        raise ValueError(
            f"Output shapes differ: {fp32_logits.shape} vs "
            f"{int8_logits.shape}"
        )

    fp32_predictions = fp32_logits.argmax(axis=1)
    int8_predictions = int8_logits.argmax(axis=1)

    agreement = float(np.mean(fp32_predictions == int8_predictions))
    fp32_accuracy = float(np.mean(fp32_predictions == y))
    int8_accuracy = float(np.mean(int8_predictions == y))

    absolute_differences = np.abs(fp32_logits - int8_logits)

    # Benchmark the same representative image, batch size 1.
    benchmark_image = X[:1]

    fp32_latency = benchmark(
        fp32_session,
        fp32_input,
        benchmark_image,
        args.warmup,
        args.iterations,
    )
    int8_latency = benchmark(
        int8_session,
        int8_input,
        benchmark_image,
        args.warmup,
        args.iterations,
    )

    fp32_size = fp32_path.stat().st_size
    int8_size = int8_path.stat().st_size
    size_reduction = (1 - int8_size / fp32_size) * 100

    latency_reduction = (
        1 - int8_latency["median_ms"] / fp32_latency["median_ms"]
    ) * 100

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fp32_model": str(fp32_path.resolve()),
        "int8_model": str(int8_path.resolve()),
        "dataset": str(dataset_path.resolve()),
        "sample_count": len(X),
        "dataset_role": (
            "Training samples only; agreement and accuracy here are not "
            "a held-out generalization evaluation."
        ),
        "prediction_agreement": agreement,
        "fp32_training_sample_accuracy": fp32_accuracy,
        "int8_training_sample_accuracy": int8_accuracy,
        "max_absolute_logit_difference": float(
            np.max(absolute_differences)
        ),
        "mean_absolute_logit_difference": float(
            np.mean(absolute_differences)
        ),
        "model_size_bytes": {
            "fp32": int(fp32_size),
            "int8": int(int8_size),
        },
        "model_size_reduction_percent": float(size_reduction),
        "latency_batch_size": 1,
        "latency_iterations": args.iterations,
        "latency_warmup": args.warmup,
        "fp32_latency_ms": fp32_latency,
        "int8_latency_ms": int8_latency,
        "median_latency_reduction_percent": float(latency_reduction),
        "providers": {
            "fp32": fp32_session.get_providers(),
            "int8": int8_session.get_providers(),
        },
    }

    report_path = int8_path.with_name("quantization_comparison.json")
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Compared samples: {len(X)}")
    print(f"Prediction agreement: {agreement:.2%}")
    print(f"FP32 sample accuracy: {fp32_accuracy:.2%}")
    print(f"INT8 sample accuracy: {int8_accuracy:.2%}")
    print(
        "Max absolute logit difference: "
        f"{report['max_absolute_logit_difference']:.8g}"
    )

    print("\nModel size:")
    print(f"  FP32: {fp32_size / 1024:.2f} KB")
    print(f"  INT8: {int8_size / 1024:.2f} KB")
    print(f"  Reduction: {size_reduction:.2f}%")

    print("\nSingle-image CPU latency:")
    print(
        f"  FP32 median: {fp32_latency['median_ms']:.4f} ms "
        f"(P95 {fp32_latency['p95_ms']:.4f} ms)"
    )
    print(
        f"  INT8 median: {int8_latency['median_ms']:.4f} ms "
        f"(P95 {int8_latency['p95_ms']:.4f} ms)"
    )
    print(f"  Median latency reduction: {latency_reduction:.2f}%")
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()