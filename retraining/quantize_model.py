
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ImageCalibrationReader(CalibrationDataReader):
    """Provide representative normalized images to ONNX Runtime."""

    def __init__(
        self,
        images: np.ndarray,
        input_name: str,
        batch_size: int = 1,
    ) -> None:
        self.images = images
        self.input_name = input_name
        self.batch_size = batch_size
        self.position = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.position >= len(self.images):
            return None

        end = min(
            self.position + self.batch_size,
            len(self.images),
        )
        batch = self.images[self.position:end]
        self.position = end

        return {self.input_name: batch}

    def rewind(self) -> None:
        self.position = 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a statically quantized INT8 ONNX candidate."
    )
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_ROOT / path

    onnx_path = resolve(args.onnx)
    dataset_path = resolve(args.dataset)

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_path = onnx_path.with_name(
        f"{onnx_path.stem}_int8.onnx"
    )
    metadata_path = output_path.with_suffix(".json")

    if output_path.exists() or metadata_path.exists():
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            "Choose a new output name or move the existing artifact."
        )

    with np.load(dataset_path, allow_pickle=False) as data:
        if "X_train" not in data:
            raise ValueError("Dataset must contain X_train.")
        images = data["X_train"]

    if len(images) == 0:
        raise ValueError("No training images available for calibration.")

    if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
        raise ValueError(
            f"Expected images shaped (N, 1, 28, 28), got {images.shape}."
        )

    if (
        not np.isfinite(images).all()
        or images.min() < 0
        or images.max() > 1
    ):
        raise ValueError(
            "Calibration images must be finite and normalized to [0, 1]."
        )

    images = images.astype(np.float32, copy=False)

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    input_names = [item.name for item in model.graph.input]
    if "images" not in input_names:
        raise ValueError(
            f"Expected input 'images'; found {input_names}."
        )

    reader = ImageCalibrationReader(
        images=images,
        input_name="images",
        batch_size=args.batch_size,
    )

    print(f"FP32 ONNX: {onnx_path}")
    print(f"Calibration images: {len(images)}")
    print("Quantization: static INT8, QDQ format")

    quantize_static(
        model_input=str(onnx_path),
        model_output=str(output_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
        op_types_to_quantize=["Conv", "Gemm", "MatMul"],
    )

    quantized_model = onnx.load(str(output_path))
    onnx.checker.check_model(quantized_model)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_model": str(onnx_path.resolve()),
        "source_dataset": str(dataset_path.resolve()),
        "output_model": output_path.name,
        "quantization": "static INT8",
        "quantization_format": "QDQ",
        "activation_type": "QUInt8",
        "weight_type": "QInt8",
        "calibration_method": "MinMax",
        "calibration_sample_count": len(images),
        "calibration_split": "X_train",
        "per_channel": True,
        "validation_status": (
            "ONNX structure checked; numerical accuracy and latency "
            "still need comparison against FP32."
        ),
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nINT8 export successful.")
    print(f"Model:    {output_path}")
    print(f"Metadata: {metadata_path}")
    print(f"FP32 size: {onnx_path.stat().st_size / 1024:.2f} KB")
    print(f"INT8 size: {output_path.stat().st_size / 1024:.2f} KB")
    print("Next: validate prediction agreement and benchmark performance.")


if __name__ == "__main__":
    main()