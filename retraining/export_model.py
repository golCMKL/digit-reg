
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from train import DigitCNN

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a trained DigitCNN checkpoint to ONNX."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output ONNX path; defaults beside the checkpoint.",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    output_path = args.output
    if output_path is None:
        output_path = checkpoint_path.with_suffix(".onnx")
    elif not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint.get("architecture") != "DigitCNN":
        raise ValueError(
            "This exporter expects a checkpoint with architecture='DigitCNN'."
        )

    model = DigitCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # One grayscale image with 28x28 pixels.
    dummy_input = torch.zeros(1, 1, 28, 28, dtype=torch.float32)

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=["images"],
            output_names=["logits"],
            dynamic_axes={
                "images": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=17,
            dynamo=False,
        )

    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "ONNX export completed, but the 'onnx' package is missing. "
            "Install it with: python -m pip install onnx"
        ) from exc

    exported_model = onnx.load(str(output_path))
    onnx.checker.check_model(exported_model)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "architecture": "DigitCNN",
        "source_checkpoint": str(checkpoint_path.resolve()),
        "onnx_file": output_path.name,
        "opset_version": 17,
        "input_name": "images",
        "input_shape": ["batch_size", 1, 28, 28],
        "output_name": "logits",
        "output_shape": ["batch_size", 10],
        "checkpoint_epoch": checkpoint.get("epoch"),
        "quantization": "none",
        "validation_status": (
            "ONNX structure checked; numerical equivalence not yet tested."
        ),
    }

    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("ONNX export successful.")
    print(f"Model:    {output_path}")
    print(f"Metadata: {metadata_path}")
    print("Input:    [batch_size, 1, 28, 28]")
    print("Output:   [batch_size, 10]")
    print("Status:   Structure checked; runtime comparison still required.")


if __name__ == "__main__":
    main()