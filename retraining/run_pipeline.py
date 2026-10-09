from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RETRAINING = ROOT / "retraining"
DATASETS = RETRAINING / "datasets"
RUNS = RETRAINING / "runs"


def run_step(name: str, command: list[str], stages: dict) -> None:
    """Run a pipeline stage and stop if the command fails."""
    print(f"\n{'=' * 65}")
    print(f"STAGE: {name}")
    print(f"{'=' * 65}")
    print("Command:", subprocess.list2cmdline(command))

    result = subprocess.run(command, cwd=ROOT, check=False)

    stages[name] = {
        "status": "passed" if result.returncode == 0 else "failed",
        "return_code": result.returncode,
        "command": subprocess.list2cmdline(command),
    }

    if result.returncode != 0:
        raise RuntimeError(
            f"Stage '{name}' failed with exit code {result.returncode}."
        )


def newest_file(directory: Path, pattern: str) -> Path:
    """Find the newest file matching a pattern."""
    files = [
        path for path in directory.glob(pattern)
        if path.is_file()
    ]

    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {directory}"
        )

    return max(files, key=lambda path: path.stat().st_mtime)


def resolve_path(path: Path) -> Path:
    """Resolve relative user paths from the project root."""
    if path.is_absolute():
        return path.resolve()
    return (ROOT / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the digit-recognition retraining workflow."
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-small-data",
        action="store_true",
        help=(
            "Allow training-only mode when a valid stratified split "
            "cannot be made. Evaluation and promotion will be blocked."
        ),
    )
    parser.add_argument(
        "--skip-quantization",
        action="store_true",
        help="Skip INT8 quantization and its comparison benchmark.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be positive.")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative.")
    if args.iterations < 1:
        parser.error("--iterations must be positive.")

    RUNS.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary_dir = RUNS / f"pipeline_{run_id}"
    summary_dir.mkdir(parents=True, exist_ok=False)
    summary_path = summary_dir / "pipeline_summary.json"

    stages: dict[str, dict] = {}
    summary = {
        "pipeline_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "running",
        "stages": stages,
        "artifacts": {},
        "notes": [],
    }

    python = sys.executable

    def save_summary() -> None:
        summary["updated_at"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        summary_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    try:
        # 1. Build a fresh labeled dataset from the database.
        run_step(
            "build_dataset",
            [python, str(RETRAINING / "build_dataset.py")],
            stages,
        )

        dataset = newest_file(DATASETS, "labeled_dataset_*.npz")
        summary["artifacts"]["labeled_dataset"] = str(dataset)

        # 2. Prepare reproducible train/validation/test splits.
        prepare_command = [
            python,
            str(RETRAINING / "prepare_data.py"),
            "--dataset", str(dataset),
            "--seed", str(args.seed),
        ]

        if args.allow_small_data:
            prepare_command.append("--allow-small-data")

        run_step("prepare_data", prepare_command, stages)

        prepared = DATASETS / f"prepared_split_seed{args.seed}.npz"
        if not prepared.is_file():
            raise FileNotFoundError(
                f"Prepared dataset not found: {prepared}"
            )

        summary["artifacts"]["prepared_dataset"] = str(prepared)

        with np.load(prepared, allow_pickle=False) as data:
            required = {"X_train", "y_train", "X_test", "y_test"}
            missing = required.difference(data.files)

            if missing:
                raise ValueError(
                    f"Prepared dataset is missing arrays: {sorted(missing)}"
                )

            train_count = len(data["y_train"])
            test_count = len(data["y_test"])

        summary["dataset_counts"] = {
            "train": train_count,
            "test": test_count,
        }

        if train_count == 0:
            raise ValueError("Training split is empty.")

        # Find newly created checkpoints so an older run isn't reused.
        old_checkpoints = set(RUNS.glob("*/candidate_model.pth"))

        # 3. Train a new candidate.
        run_step(
            "train",
            [
                python,
                str(RETRAINING / "train.py"),
                "--dataset", str(prepared),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--learning-rate", str(args.learning_rate),
                "--seed", str(args.seed),
            ],
            stages,
        )

        new_checkpoints = [
            path for path in RUNS.glob("*/candidate_model.pth")
            if path not in old_checkpoints and path.is_file()
        ]

        if not new_checkpoints:
            raise FileNotFoundError(
                "Training finished, but no new candidate_model.pth "
                "was found."
            )

        checkpoint = max(
            new_checkpoints,
            key=lambda path: path.stat().st_mtime,
        )
        candidate_dir = checkpoint.parent
        fp32_onnx = candidate_dir / "candidate_model.onnx"

        summary["artifacts"]["checkpoint"] = str(checkpoint)

        # 4. Evaluate only when there is a held-out test set.
        if test_count > 0:
            run_step(
                "evaluate",
                [
                    python,
                    str(RETRAINING / "evaluate.py"),
                    "--dataset", str(prepared),
                    "--checkpoint", str(checkpoint),
                    "--batch-size", str(args.batch_size),
                ],
                stages,
            )

            evaluation_dirs = [
                path for path in candidate_dir.glob("evaluation_*")
                if path.is_dir()
            ]
            if not evaluation_dirs:
                raise FileNotFoundError(
                    "Evaluation completed but no evaluation directory "
                    "was created."
                )

            metrics_path = max(
                evaluation_dirs,
                key=lambda path: path.stat().st_mtime,
            ) / "metrics.json"

            summary["artifacts"]["evaluation"] = str(metrics_path)
        else:
            stages["evaluate"] = {
                "status": "skipped",
                "reason": "No held-out test samples are available.",
            }
            summary["notes"].append(
                "Evaluation skipped: there are no held-out test samples."
            )

        # 5. Export the candidate to FP32 ONNX.
        run_step(
            "export_onnx",
            [
                python,
                str(RETRAINING / "export_model.py"),
                "--checkpoint", str(checkpoint),
                "--output", str(fp32_onnx),
            ],
            stages,
        )
        summary["artifacts"]["fp32_onnx"] = str(fp32_onnx)

        # 6. Check PyTorch and ONNX output consistency.
        run_step(
            "validate_onnx",
            [
                python,
                str(RETRAINING / "validate_model.py"),
                "--checkpoint", str(checkpoint),
                "--onnx", str(fp32_onnx),
                "--dataset", str(prepared),
            ],
            stages,
        )

        validation_reports = list(
            candidate_dir.glob("validation_report.json")
        )
        if validation_reports:
            summary["artifacts"]["validation_report"] = str(
                max(
                    validation_reports,
                    key=lambda path: path.stat().st_mtime,
                )
            )

        # 7. Quantize and compare the candidate if requested.
        if not args.skip_quantization:
            run_step(
                "quantize_int8",
                [
                    python,
                    str(RETRAINING / "quantize_model.py"),
                    "--onnx", str(fp32_onnx),
                    "--dataset", str(prepared),
                    "--batch-size", str(args.batch_size),
                ],
                stages,
            )

            int8_onnx = candidate_dir / "candidate_model_int8.onnx"
            if not int8_onnx.is_file():
                raise FileNotFoundError(
                    f"Quantized model not found: {int8_onnx}"
                )

            run_step(
                "compare_quantization",
                [
                    python,
                    str(RETRAINING / "compare_quantization.py"),
                    "--fp32", str(fp32_onnx),
                    "--int8", str(int8_onnx),
                    "--dataset", str(prepared),
                    "--warmup", str(args.warmup),
                    "--iterations", str(args.iterations),
                ],
                stages,
            )

            summary["artifacts"]["int8_onnx"] = str(int8_onnx)
            comparison_path = candidate_dir / "quantization_comparison.json"
            if comparison_path.is_file():
                summary["artifacts"]["quantization_comparison"] = str(
                    comparison_path
                )
        else:
            stages["quantize_int8"] = {"status": "skipped"}
            stages["compare_quantization"] = {"status": "skipped"}

        # Promotion is deliberately never automatic.
        if test_count == 0:
            summary["promotion"] = {
                "status": "blocked",
                "reason": "No held-out test samples are available.",
            }
        else:
            summary["promotion"] = {
                "status": "not_attempted",
                "reason": (
                    "Promotion requires separate approval and additional "
                    "production compatibility checks."
                ),
            }

        summary["status"] = "completed"
        return_code = 0

    except (subprocess.CalledProcessError, OSError, RuntimeError, ValueError) as error:
        summary["status"] = "failed"
        summary["error"] = str(error)
        print(f"\nPIPELINE FAILED: {error}", file=sys.stderr)
        return_code = 1

    finally:
        summary["finished_at"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        save_summary()
        print(f"\nPipeline status: {summary['status']}")
        print(f"Summary: {summary_path}")

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())

