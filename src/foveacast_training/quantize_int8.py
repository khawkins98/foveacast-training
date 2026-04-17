"""Quantise an MSINet ONNX model to INT8 via static post-training quantisation.

Multi-duration release work for issue #19. Three `.onnx` artefacts (1s /
3s / 7s) at 57 MB each = 171 MB for a Foveacast download — above our
working budget for a browser-fetched model set. INT8 static quantisation
drops each artefact to ~26 MB, so three durations become ~78 MB total.

Static quantisation needs a calibration dataset to compute per-tensor
scale and zero-point values that minimise activation clipping across
realistic inputs. We use random UEyes training images (default 100 — ORT
docs recommend 100-500, with diminishing returns past a couple hundred).
Dynamic quantisation would skip the calibration pass but is typically
lossier on conv-heavy networks like MSI-Net where activation
distributions vary meaningfully per layer.

Borrowed / referenced:

- `onnxruntime.quantization.quantize_static` — the static PTQ entrypoint.
  https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- `CalibrationDataReader` — abstract base for feeding calibration batches;
  we subclass it against `UEyesDataset` to reuse the exact preprocessing
  the Phase 3 loader already does.
- `quant_pre_process` — ORT-recommended shape inference + model cleanup
  pass before quantising. Skipping it is the top cause of INT8 models that
  load but fail at inference time with "missing shape info" errors.

Run:
    # Produce an INT8 release artefact for the fine-tuned 1s model:
    .venv/bin/python -m foveacast_training.quantize_int8 \\
        --checkpoint runs/full-heatmaps_1s-*/best.pt \\
        --calibration-data data/ueyes/UEyes_dataset \\
        --out releases/foveacast-v3-1s-int8.onnx \\
        --report releases/foveacast-v3-1s-int8.parity.json

The INT8 artefact's PyTorch parity gate is looser than FP16's (1e-2
default vs 1e-3). INT8 has 256 discrete values per quantised tensor —
per-pixel error of ~1% of the output range is expected. The gate exists
to catch a quantisation that produces a *wildly* wrong model (e.g. dead
layers from bad calibration), not to enforce bit-level parity.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from foveacast_training.export_onnx import (
    CANONICAL_INPUT_SHAPE,
    EXPORT_DEVICE,
    export_to_onnx,
    load_pytorch_model,
    validate_parity,
)
from foveacast_training.ueyes_dataset import UEyesDataset


class UEyesCalibrationReader(CalibrationDataReader):
    """Feeds `quantize_static` batches of real UEyes training images.

    why: ORT's calibration pass needs inputs representative of inference-
    time distribution. Random uniform noise would compute scale/zero-point
    against values that never occur in practice, producing a model that's
    calibrated for the wrong input. Real UEyes images match the pixel-
    value statistics, aspect ratios, and preprocessing Foveacast actually
    feeds at browser time.

    The reader yields one sample at a time (batch_size=1) because the
    exported ONNX graph has a dynamic batch axis — ORT picks up the
    sample-wise statistics and aggregates internally.
    """

    def __init__(self, data_root: Path, n_samples: int, seed: int = 0):
        # why: saliency_variant is irrelevant here — we only need the
        # image tensor, not the target map. heatmaps_3s is a safe default.
        self.dataset = UEyesDataset(
            data_root, split="train", saliency_variant="heatmaps_3s"
        )
        rng = np.random.default_rng(seed)
        n_available = len(self.dataset)
        if n_samples > n_available:
            n_samples = n_available
        self.indices = rng.choice(n_available, size=n_samples, replace=False).tolist()
        self._idx_iter = iter(self.indices)

    def get_next(self) -> dict[str, np.ndarray] | None:
        try:
            i = next(self._idx_iter)
        except StopIteration:
            return None
        # UEyesDataset.__getitem__ returns (image_tensor, saliency_tensor)
        # where image_tensor is [3, H, W] float32 in [0, 255]. Add the
        # batch axis for ONNX input shape [1, 3, H, W].
        image, _ = self.dataset[i]
        return {"input": image.unsqueeze(0).numpy().astype(np.float32)}


def quantize_onnx_to_int8(
    fp32_path: Path,
    int8_path: Path,
    calibration_reader: CalibrationDataReader,
) -> None:
    """Run static post-training quantisation on an FP32 ONNX model.

    Two steps:
    1. Shape inference + cleanup via `quant_pre_process` — populates
       missing tensor shape info in the graph, removes initializer dups,
       and inlines a couple of shape-only ops that the quantiser can't
       otherwise reason about. Skipping this step is the top cause of
       "AssertionError: Could not infer shape of tensor X" failures at
       `quantize_static` time.
    2. `quantize_static` itself — runs the reader through the graph,
       accumulates activation distributions, computes scale + zero-point
       per tensor, and emits an INT8 graph with `QuantizeLinear` /
       `DequantizeLinear` ops wrapping the quantised regions.

    Why QUInt8 for activations, QInt8 for weights: this is ORT's
    recommended default for conv-heavy nets. Activations are always
    non-negative post-ReLU so QUInt8 (asymmetric, [0, 255]) wastes no
    range on negative values; weights are typically zero-centred so
    QInt8 (symmetric, [-128, 127]) matches the distribution better.

    CalibrationMethod.MinMax is the cheapest choice and works well for
    our input range [0, 255]. Entropy-based (KL-divergence) calibration
    would be more accurate on highly skewed distributions but adds
    compute with marginal gains on saliency outputs that stay in a
    well-behaved range.
    """
    with tempfile.TemporaryDirectory() as tmp:
        preprocessed = Path(tmp) / "preprocessed.onnx"
        quant_pre_process(str(fp32_path), str(preprocessed), skip_symbolic_shape=False)
        quantize_static(
            model_input=str(preprocessed),
            model_output=str(int8_path),
            calibration_data_reader=calibration_reader,
            quant_format=None,  # default: QOperator (fused quant ops, smaller)
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
            per_channel=True,
        )


def validate_int8_parity(
    fp32_pytorch_model,
    int8_path: Path,
    n_random_trials: int,
    tolerance: float,
) -> dict[str, float]:
    """Thin wrapper around `export_onnx.validate_parity` that runs the
    same 6-trial bank against an INT8 ONNX session.

    Kept as a wrapper so the trial definition lives in one place — if
    Phase 8's trial bank ever expands (e.g. adding image-like fixtures
    from UEyes), both FP16 and INT8 reports pick up the change.
    """
    return validate_parity(
        fp32_pytorch_model,
        int8_path,
        n_random_trials=n_random_trials,
        tolerance=tolerance,
    )


def _onnx_size_mb(path: Path) -> float:
    # why: INT8 models sometimes also produce a sidecar .data file
    # depending on ORT's output mode. Sum both if present so the reported
    # size matches what Foveacast actually has to fetch.
    total = path.stat().st_size
    sidecar = path.with_suffix(path.suffix + ".data")
    if sidecar.exists():
        total += sidecar.stat().st_size
    return total / 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "PyTorch state_dict (.pt) to export and quantise. FP32 ONNX "
            "is produced as a temporary intermediate; only the INT8 "
            "artefact is written to --out."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for the INT8 .onnx artefact.",
    )
    parser.add_argument(
        "--calibration-data",
        type=Path,
        default=Path("data/ueyes/UEyes_dataset"),
        help="UEyes dataset root for calibration samples.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=100,
        help=(
            "Number of UEyes training images to calibrate against. "
            "ORT docs recommend 100-500; diminishing returns past a "
            "couple hundred."
        ),
    )
    parser.add_argument(
        "--calibration-seed",
        type=int,
        default=0,
        help="RNG seed for which training images get sampled for calibration.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-2,
        help=(
            "Max absolute per-pixel error allowed between PyTorch FP32 "
            "and INT8 ONNX. Default 1e-2 — INT8 has 256 discrete values "
            "per quantised tensor so sub-1 percent error is the right ballpark."
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of random-uniform parity trials (+ bs=2 + saturated).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path for size + parity numbers.",
    )
    args = parser.parse_args()

    print(f"→ loading {args.checkpoint}")
    pytorch_model = load_pytorch_model(args.checkpoint).to(EXPORT_DEVICE)

    with tempfile.TemporaryDirectory() as tmp:
        fp32_path = Path(tmp) / "fp32.onnx"
        print(f"→ exporting FP32 ONNX (opset {args.opset})")
        export_to_onnx(pytorch_model, fp32_path, opset=args.opset)
        fp32_size_mb = fp32_path.stat().st_size / 1e6
        print(f"  {fp32_size_mb:.1f} MB (FP32, temporary)")

        print(
            f"→ calibrating on {args.calibration_samples} UEyes train images "
            f"(seed={args.calibration_seed})"
        )
        reader = UEyesCalibrationReader(
            args.calibration_data,
            n_samples=args.calibration_samples,
            seed=args.calibration_seed,
        )

        print(f"→ quantising FP32 → INT8 → {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        quantize_onnx_to_int8(fp32_path, args.out, reader)

    int8_size_mb = _onnx_size_mb(args.out)
    reduction_vs_fp32 = (1 - int8_size_mb / fp32_size_mb) * 100
    print(f"  {int8_size_mb:.1f} MB (INT8) — {reduction_vs_fp32:.0f}% smaller than FP32")

    print(
        f"→ validating PyTorch FP32 ↔ onnxruntime INT8 parity "
        f"(tolerance={args.tolerance:.0e}, random-trials={args.trials} + bs=2 + saturated)"
    )
    # why: INT8 graphs emit `QuantizeLinear` / `DequantizeLinear` that
    # the CPU EP handles fine; other EPs have variable INT8 op coverage.
    # CPUExecutionProvider matches our FP32 parity convention and works
    # on every platform.
    parity = validate_int8_parity(
        pytorch_model,
        args.out,
        n_random_trials=args.trials,
        tolerance=args.tolerance,
    )
    print(f"  trials:       {parity['n_trials']} ({parity['trial_kinds']})")
    print(f"  max abs err:  {parity['max_abs_err']:.2e}")
    print(f"  mean abs err: {parity['mean_abs_err']:.2e}")

    report = {
        "checkpoint": str(args.checkpoint),
        "artefact": str(args.out),
        "precision": "int8",
        "size_mb": int8_size_mb,
        "fp32_reference_size_mb": fp32_size_mb,
        "reduction_vs_fp32_pct": reduction_vs_fp32,
        "opset": args.opset,
        "calibration": {
            "dataset_root": str(args.calibration_data),
            "n_samples": args.calibration_samples,
            "seed": args.calibration_seed,
            "method": "MinMax",
            "per_channel": True,
            "activation_type": "QUInt8",
            "weight_type": "QInt8",
        },
        "input_shape": list(CANONICAL_INPUT_SHAPE),
        "parity": parity,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(f"✓ wrote {args.report}")

    if parity["within_tolerance"]:
        print("✓ INT8 parity gate closed")
    else:
        print(
            f"✗ parity FAILED: max abs err {parity['max_abs_err']:.2e} > {args.tolerance:.0e} — "
            "consider increasing --calibration-samples or shipping FP16 instead"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
