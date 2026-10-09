import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "retraining" / "promote_model.py"


def run_script(*args: str) -> subprocess.CompletedProcess:
    """Run the promotion script and capture its output."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def write_json(path: Path, data: dict) -> None:
    """Write a JSON test report."""
    path.write_text(json.dumps(data), encoding="utf-8")


def setup_promotion_files(tmp_path: Path) -> dict[str, Path]:
    """Create isolated files for promotion tests."""
    paths = {
        "candidate": tmp_path / "candidate.onnx",
        "target": tmp_path / "production.onnx",
        "evaluation": tmp_path / "metrics.json",
        "validation": tmp_path / "validation.json",
    }

    paths["candidate"].write_bytes(b"candidate-model")
    paths["target"].write_bytes(b"original-production-model")

    write_json(
        paths["evaluation"],
        {"accuracy": 0.95, "test_samples": 200},
    )
    write_json(
        paths["validation"],
        {"overall_pass": True},
    )

    return paths


def promotion_args(paths: dict[str, Path]) -> list[str]:
    """Build common command-line arguments."""
    return [
        "--candidate", str(paths["candidate"]),
        "--target", str(paths["target"]),
        "--evaluation", str(paths["evaluation"]),
        "--validation", str(paths["validation"]),
    ]


def test_promotion_rejects_insufficient_test_samples(
    tmp_path: Path,
) -> None:
    paths = setup_promotion_files(tmp_path)
    write_json(
        paths["evaluation"],
        {"accuracy": 0.99, "test_samples": 10},
    )

    result = run_script(
        *promotion_args(paths),
        "--min-test-samples", "100",
    )

    assert result.returncode != 0
    assert "test samples" in result.stderr.lower()
    assert paths["target"].read_bytes() == b"original-production-model"
    assert not list(tmp_path.glob("production.onnx.backup_*"))


def test_promotion_rejects_low_accuracy(tmp_path: Path) -> None:
    paths = setup_promotion_files(tmp_path)
    write_json(
        paths["evaluation"],
        {"accuracy": 0.70, "test_samples": 200},
    )

    result = run_script(
        *promotion_args(paths),
        "--min-accuracy", "0.90",
    )

    assert result.returncode != 0
    assert "below the required" in result.stderr.lower()
    assert paths["target"].read_bytes() == b"original-production-model"


def test_promotion_rejects_failed_validation(
    tmp_path: Path,
) -> None:
    paths = setup_promotion_files(tmp_path)
    write_json(
        paths["validation"],
        {"overall_pass": False},
    )

    result = run_script(*promotion_args(paths))

    assert result.returncode != 0
    assert "validation" in result.stderr.lower()
    assert paths["target"].read_bytes() == b"original-production-model"


def test_promotion_rejects_missing_candidate(
    tmp_path: Path,
) -> None:
    paths = setup_promotion_files(tmp_path)
    paths["candidate"].unlink()

    result = run_script(*promotion_args(paths))

    assert result.returncode != 0
    assert paths["target"].read_bytes() == b"original-production-model"


def test_successful_promotion_creates_backup_and_record(
    tmp_path: Path,
) -> None:
    paths = setup_promotion_files(tmp_path)

    result = run_script(*promotion_args(paths))

    assert result.returncode == 0, result.stderr
    assert paths["target"].read_bytes() == b"candidate-model"

    backups = list(tmp_path.glob("production.onnx.backup_*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"original-production-model"

    record = tmp_path / "model_deployment.json"
    assert record.is_file()

    deployment = json.loads(record.read_text(encoding="utf-8"))
    assert deployment["test_samples"] == 200
    assert deployment["test_accuracy"] == pytest.approx(0.95)


def test_rollback_restores_latest_backup(tmp_path: Path) -> None:
    target = tmp_path / "production.onnx"
    older_backup = tmp_path / "production.onnx.backup_20261008_120000"
    newer_backup = tmp_path / "production.onnx.backup_20261009_120000"

    target.write_bytes(b"current-model")
    older_backup.write_bytes(b"older-model")
    newer_backup.write_bytes(b"latest-previous-model")

    result = run_script("--target", str(target), "--rollback")

    assert result.returncode == 0, result.stderr
    assert target.read_bytes() == b"latest-previous-model"

    preserved_files = list(
        tmp_path.glob("production.onnx.backup_*_before_rollback")
    )
    assert len(preserved_files) == 1
    assert preserved_files[0].read_bytes() == b"current-model"


def test_rollback_fails_when_no_backup_exists(
    tmp_path: Path,
) -> None:
    target = tmp_path / "production.onnx"
    target.write_bytes(b"current-model")

    result = run_script("--target", str(target), "--rollback")

    assert result.returncode != 0
    assert "no backups" in result.stderr.lower()
    assert target.read_bytes() == b"current-model"


def test_promotion_refuses_candidate_equal_to_target(
    tmp_path: Path,
) -> None:
    paths = setup_promotion_files(tmp_path)
    paths["candidate"] = paths["target"]

    result = run_script(*promotion_args(paths))

    assert result.returncode != 0
    assert "must be different" in result.stderr.lower()
    assert paths["target"].read_bytes() == b"original-production-model"
