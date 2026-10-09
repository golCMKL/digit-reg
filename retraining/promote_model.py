from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Report not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object in {path}")

    return data


def get_test_sample_count(report: dict[str, Any]) -> int:
    """Read a test sample count or derive it from a confusion matrix."""
    for key in ("test_samples", "n_test", "num_test_samples", "n_samples"):
        value = report.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value

    matrix = report.get("confusion_matrix")
    if isinstance(matrix, list) and matrix:
        try:
            return sum(sum(int(value) for value in row) for row in matrix)
        except (TypeError, ValueError):
            pass

    raise ValueError(
        "Could not determine the number of test samples from the evaluation "
        "report. Add a 'test_samples' field to metrics.json."
    )


def get_accuracy(report: dict[str, Any]) -> float:
    value = report.get("accuracy")

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(
            "Evaluation report must contain numeric 'accuracy', "
            "expressed as a fraction from 0.0 to 1.0."
        )

    accuracy = float(value)
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError(
            f"Expected accuracy between 0 and 1, got {accuracy}."
        )

    return accuracy


def check_validation(report: dict[str, Any]) -> None:
    # Accept the explicit pass field used by the validation script.
    passed = report.get("overall_pass", report.get("passed"))

    if passed is not True:
        raise ValueError(
            "ONNX validation has not explicitly passed. Check "
            "validation_report.json and confirm its pass field."
        )


def promote(
    candidate: Path,
    target: Path,
    evaluation_path: Path,
    validation_path: Path,
    min_accuracy: float,
    min_test_samples: int,
) -> None:
    if not candidate.is_file():
        raise FileNotFoundError(f"Candidate model not found: {candidate}")

    if candidate.resolve() == target.resolve():
        raise ValueError("Candidate and production target must be different files.")

    evaluation = read_json(evaluation_path)
    validation = read_json(validation_path)

    check_validation(validation)

    accuracy = get_accuracy(evaluation)
    test_samples = get_test_sample_count(evaluation)

    if test_samples < min_test_samples:
        raise ValueError(
            f"Only {test_samples} test samples found; "
            f"at least {min_test_samples} are required."
        )

    if accuracy < min_accuracy:
        raise ValueError(
            f"Test accuracy {accuracy:.2%} is below the required "
            f"{min_accuracy:.2%}."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Preserve the current model before replacing it.
    backup = None
    if target.exists():
        backup = target.with_name(
            f"{target.name}.backup_{timestamp}"
        )
        shutil.copy2(target, backup)
        print(f"Backup created: {backup}")

    # Copy to a temporary file in the same directory, then replace atomically.
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

        shutil.copy2(candidate, temp_path)
        temp_path.replace(target)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise

    deployment_record = {
        "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "candidate": str(candidate.resolve()),
        "production_model": str(target.resolve()),
        "evaluation_report": str(evaluation_path.resolve()),
        "validation_report": str(validation_path.resolve()),
        "test_samples": test_samples,
        "test_accuracy": accuracy,
        "minimum_required_accuracy": min_accuracy,
        "backup": str(backup.resolve()) if backup else None,
    }

    record_path = target.with_name("model_deployment.json")
    record_path.write_text(
        json.dumps(deployment_record, indent=2),
        encoding="utf-8",
    )

    print("Promotion successful.")
    print(f"Production model: {target}")
    print(f"Test samples: {test_samples}")
    print(f"Test accuracy: {accuracy:.2%}")
    print(f"Deployment record: {record_path}")


def rollback(target: Path) -> None:
    if not target.parent.is_dir():
        raise FileNotFoundError(f"Production directory not found: {target.parent}")

    backups = sorted(
        (
            path
            for path in target.parent.glob(f"{target.name}.backup_*")
            if path.is_file()
            and "_before_rollback" not in path.name
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not backups:
        raise FileNotFoundError(f"No backups found for {target}")

    backup = backups[0]
    if not backup.is_file():
        raise FileNotFoundError(f"Backup is not a file: {backup}")

    # Save the current model too, so rollback is reversible.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    current_copy = None

    if target.is_file():
        current_copy = target.with_name(
            f"{target.name}.backup_{timestamp}_before_rollback"
        )
        shutil.copy2(target, current_copy)

    with tempfile.NamedTemporaryFile(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        shutil.copy2(backup, temp_path)
        temp_path.replace(target)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    print("Rollback successful.")
    print(f"Restored: {target}")
    print(f"From backup: {backup}")
    if current_copy:
        print(f"Previous production model preserved: {current_copy}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Safely promote or roll back a digit-recognition model."
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("app/digit_cnn_int8.onnx"),
        help="Production model path (default: app/digit_cnn_int8.onnx).",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Restore the newest backup instead of promoting a candidate.",
    )
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.90,
        help="Minimum held-out test accuracy as a fraction (default: 0.90).",
    )
    parser.add_argument(
        "--min-test-samples",
        type=int,
        default=100,
        help="Minimum number of held-out test samples (default: 100).",
    )

    args = parser.parse_args()

    try:
        if args.rollback:
            rollback(args.target)
            return 0

        if args.candidate is None or args.evaluation is None or args.validation is None:
            parser.error(
                "Promotion requires --candidate, --evaluation, and --validation."
            )

        if not 0.0 <= args.min_accuracy <= 1.0:
            parser.error("--min-accuracy must be between 0 and 1.")

        if args.min_test_samples < 1:
            parser.error("--min-test-samples must be at least 1.")

        promote(
            candidate=args.candidate,
            target=args.target,
            evaluation_path=args.evaluation,
            validation_path=args.validation,
            min_accuracy=args.min_accuracy,
            min_test_samples=args.min_test_samples,
        )
        return 0

    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Promotion/rollback refused: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())