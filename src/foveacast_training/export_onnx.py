"""Export an MSINet checkpoint to ONNX with onnxruntime CPU parity validation.

Phase 8 of issue #1. Produces the single `.onnx` release artefact that
Foveacast consumes via `onnxruntime-web` in the browser. The gate is:
PyTorch forward pass and `onnxruntime` CPU forward pass agree within
float tolerance on a fixed-seed input.

The script mirrors Foveacast V2's `scripts/unisal-onnx-export.py`
pattern:

- Single-file output: `save_as_external_data=False` on `onnx.save_model`,
  so the artefact is one blob rather than a sidecar-weights layout. This
  matters because `onnxruntime-web` works best with self-contained files;
  external-data format would need separate fetching in the browser.
- Dynamic batch dim only: spatial dims stay fixed at (240, 320) per
  ARCHITECTURE.md's `.onnx` contract. Batch dim is dynamic so a future
  caller can batch multiple images if latency budget permits.
- Opset 17: current PyTorch default at export time. Widely supported
  by `onnxruntime-web`.
- Input/output names: `input` / `output`, matching the contract in
  ARCHITECTURE.md §"The contract with Foveacast".

Run:
    # Export stock weights to sanity-check the pipeline (Phase 8 gate —
    # closes independent of Phase 5 since it's about the export mechanism,
    # not the specific weights):
    .venv/bin/python -m foveacast_training.export_onnx \\
        --checkpoint weights/msinet_salicon.pt \\
        --out releases/foveacast-stock-dev.onnx

    # Export the Phase 5 fine-tuned best.pt for the real release:
    .venv/bin/python -m foveacast_training.export_onnx \\
        --checkpoint runs/full-*/best.pt \\
        --out releases/foveacast-v3.onnx

The `releases/` directory is gitignored — the artefact ships via GitHub
Releases, not via committed bytes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from foveacast_training.msinet import MSINet

# why: export happens on CPU regardless of available accelerators. MPS
# tracing has known issues with a few ops (e.g. `torch.onnx` shape
# inference sometimes trips on MPS tensors). CPU tracing is the safe,
# portable choice and the exported graph runs on any target regardless.
EXPORT_DEVICE = torch.device("cpu")

# Canonical input shape per ARCHITECTURE.md §"The contract with Foveacast":
# (N=1, C=3, H=240, W=320). Batch dim is dynamic in the export; spatial
# dims are fixed.
CANONICAL_INPUT_SHAPE: tuple[int, int, int, int] = (1, 3, 240, 320)


def load_pytorch_model(checkpoint_path: Path) -> MSINet:
    """Load a checkpoint onto CPU in eval mode. Shared with eval.py's
    load_model helper in spirit but kept local to avoid a cross-module
    dependency — export_onnx should be runnable without the eval extras.
    """
    model = MSINet()
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


def export_to_onnx(
    model: MSINet,
    out_path: Path,
    input_shape: tuple[int, int, int, int] = CANONICAL_INPUT_SHAPE,
    opset: int = 17,
) -> None:
    """Trace `model` via `torch.onnx.export` and write a single-file
    ONNX artefact to `out_path`. Creates parent dirs as needed.
    """
    # Dummy input matches MSINet's documented contract: RGB, [0, 255],
    # float32, NCHW. Values don't affect the graph shape but do affect
    # which branches get traced — keep them in the realistic range so
    # the trace reflects the ops an inference run would actually hit.
    dummy = (torch.randn(*input_shape) * 50 + 128).float()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Phase 8 choice: dynamic batch dim, fixed spatial dims.
    # why: a single-image inference from Foveacast's browser is always
    # batch_size=1, but keeping the dim dynamic means a batch-of-N future
    # caller doesn't need a re-export. Spatial dims fixed at (240, 320)
    # matches the documented contract — dynamic spatial would require
    # changes to _tf1_bilinear_upsample's output-size computation which
    # is scope creep for Phase 8.
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # Inline external data — produce a single-file blob.
    # why: torch.onnx.export can externalise large weight tensors into a
    # sidecar file by default for >2GB models. At ~100 MB MSINet is well
    # below that threshold, but the safer belt-and-braces is to round-trip
    # through onnx.save_model with save_as_external_data=False to force
    # everything into the main file. onnxruntime-web loads this as one
    # fetch instead of two.
    model_proto = onnx.load(str(out_path))
    onnx.save_model(model_proto, str(out_path), save_as_external_data=False)


def validate_parity(
    pytorch_model: MSINet,
    onnx_path: Path,
    n_trials: int = 3,
    tolerance: float = 1e-4,
    input_shape: tuple[int, int, int, int] = CANONICAL_INPUT_SHAPE,
) -> dict[str, float]:
    """Run fixed-seed inputs through both PyTorch and onnxruntime CPU,
    report max and mean absolute error across trials.

    A passing gate (within tolerance) means the exported artefact
    faithfully reproduces the PyTorch forward pass — so anything
    Foveacast sees from `onnxruntime-web` matches what our training /
    evaluation numbers claim.
    """
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed=0)

    max_abs_err = 0.0
    mean_abs_err_sum = 0.0

    for _trial in range(n_trials):
        x_np = rng.uniform(0, 255, size=input_shape).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        with torch.no_grad():
            y_torch = pytorch_model(x_torch).numpy()

        y_onnx = sess.run(["output"], {"input": x_np})[0]

        assert y_torch.shape == y_onnx.shape, (
            f"output shape mismatch: PyTorch {y_torch.shape} vs ONNX {y_onnx.shape}"
        )

        diff = np.abs(y_torch - y_onnx)
        max_abs_err = max(max_abs_err, float(diff.max()))
        mean_abs_err_sum += float(diff.mean())

    return {
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err_sum / n_trials,
        "tolerance": tolerance,
        "within_tolerance": max_abs_err < tolerance,
        "n_trials": n_trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to PyTorch state_dict (.pt).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for the .onnx artefact.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Max absolute per-pixel error allowed between PyTorch and ONNX.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of fixed-seed forward passes to compare.",
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
        help="Optional JSON report path for the parity numbers.",
    )
    args = parser.parse_args()

    print(f"→ loading {args.checkpoint}")
    model = load_pytorch_model(args.checkpoint).to(EXPORT_DEVICE)

    print(f"→ exporting to {args.out} (opset {args.opset})")
    export_to_onnx(model, args.out, opset=args.opset)
    size_mb = args.out.stat().st_size / 1e6
    print(f"  {size_mb:.1f} MB")

    print(f"→ validating PyTorch ↔ onnxruntime CPU parity "
          f"(tolerance={args.tolerance:.0e}, trials={args.trials})")
    parity = validate_parity(model, args.out, n_trials=args.trials, tolerance=args.tolerance)
    print(f"  max abs err:  {parity['max_abs_err']:.2e}")
    print(f"  mean abs err: {parity['mean_abs_err']:.2e}")

    report = {
        "checkpoint": str(args.checkpoint),
        "artefact": str(args.out),
        "size_mb": size_mb,
        "opset": args.opset,
        "input_shape": list(CANONICAL_INPUT_SHAPE),
        "parity": parity,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(f"✓ wrote {args.report}")

    if parity["within_tolerance"]:
        print("✓ Phase 8 parity gate closed")
    else:
        print(
            f"✗ parity FAILED: max abs err {parity['max_abs_err']:.2e} > {args.tolerance:.0e}"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
