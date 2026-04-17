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

    # Export the fine-tuned best.pt for the real release (FP32, ~106 MB):
    .venv/bin/python -m foveacast_training.export_onnx \\
        --checkpoint runs/full-*/best.pt \\
        --out releases/foveacast-v3.onnx

    # Same but FP16 quantised (~57 MB, shipped format per v0.1.0+):
    .venv/bin/python -m foveacast_training.export_onnx --fp16 \\
        --checkpoint runs/full-*/best.pt \\
        --out releases/foveacast-v3-3s-fp16.onnx

The FP16 path runs the naive `convert_float_to_float16(keep_io_types=True)`
from `onnxconverter_common`. The 2026-04-17 LEARNINGS entry covers why
naive beats op-list-selective (cross-format boundaries introduce rounding
that an all-FP16 path avoids). Parity tolerance is relaxed to 1e-3 on the
FP16 path because FP16 arithmetic has ~6e-5 minimum-normal precision —
the FP32 gate's 1e-4 is not physically achievable.

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


def convert_to_fp16(onnx_path: Path) -> None:
    """In-place conversion of an FP32 ONNX model to FP16.

    Uses `onnxconverter_common.float16.convert_float_to_float16` with
    `keep_io_types=True` so the graph's input/output tensors stay FP32 —
    downstream callers (onnxruntime-web, Foveacast) don't need to change
    input preparation or output consumption when switching from FP32 to
    FP16 artefacts. Internal conv weights and activations go FP16, which
    is where the ~50% size saving comes from.

    Naive conversion (all ops → FP16) was chosen over op-list-selective
    (`op_block_list=[...]` to keep `_normalize`'s division in FP32) per
    the 2026-04-17 LEARNINGS entry: selective was 29% worse on max error
    because each FP16↔FP32 boundary introduces a rounding step that the
    all-FP16 path avoids.
    """
    from onnxconverter_common.float16 import convert_float_to_float16

    model_proto = onnx.load(str(onnx_path))
    fp16_proto = convert_float_to_float16(model_proto, keep_io_types=True)
    onnx.save_model(fp16_proto, str(onnx_path), save_as_external_data=False)


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
    #
    # why dynamo=True: PyTorch 2.5+ is mid-migration from the legacy
    # TorchScript exporter to the new torch.export-based path; the two
    # emit different subgraphs for fancy indexing (which
    # _tf1_bilinear_upsample uses heavily). Pinning dynamo=True makes
    # our choice durable across torch bumps so the artefact's op mix
    # doesn't silently drift.
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=True,
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


def _parity_trial(
    pytorch_model: MSINet,
    sess: ort.InferenceSession,
    x_np: np.ndarray,
) -> tuple[float, float]:
    """One PyTorch vs onnxruntime forward pass. Returns (max, mean) abs error."""
    x_torch = torch.from_numpy(x_np)
    with torch.no_grad():
        y_torch = pytorch_model(x_torch).numpy()
    y_onnx = sess.run(["output"], {"input": x_np})[0]
    assert y_torch.shape == y_onnx.shape, (
        f"output shape mismatch: PyTorch {y_torch.shape} vs ONNX {y_onnx.shape}"
    )
    diff = np.abs(y_torch - y_onnx)
    return float(diff.max()), float(diff.mean())


def validate_parity(
    pytorch_model: MSINet,
    onnx_path: Path,
    n_random_trials: int = 3,
    tolerance: float = 1e-4,
    input_shape: tuple[int, int, int, int] = CANONICAL_INPUT_SHAPE,
) -> dict[str, float]:
    """Run varied inputs through both PyTorch and onnxruntime CPU,
    report max and mean absolute error across all trials.

    Trials cover:
    1. `n_random_trials` fixed-seed uniform random inputs at batch_size=1.
    2. One batch_size=2 trial — exercises the dynamic batch axis and the
       broadcast paths in `_tf1_bilinear_upsample`'s advanced indexing,
       which might behave differently from a bs=1 trace.
    3. Saturated edge inputs: all-zero and all-255. Checks the min-max
       normaliser's `1 / (eps + max)` divisor doesn't diverge between
       PyTorch and ONNX on the degenerate-input path.

    A passing gate (all trials within tolerance) means the exported
    artefact faithfully reproduces the PyTorch forward pass across the
    kinds of inputs Foveacast will actually feed it.
    """
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed=0)

    max_abs_err = 0.0
    mean_abs_err_sum = 0.0
    n_trials = 0

    # Trial bank 1: random uniform inputs at the canonical batch_size=1.
    for _ in range(n_random_trials):
        x = rng.uniform(0, 255, size=input_shape).astype(np.float32)
        tm, mm = _parity_trial(pytorch_model, sess, x)
        max_abs_err = max(max_abs_err, tm)
        mean_abs_err_sum += mm
        n_trials += 1

    # Trial bank 2: batched input (batch_size=2). Exercises the dynamic-
    # batch-dim path + broadcast in _tf1_bilinear_upsample's index gather.
    b2_shape = (2, *input_shape[1:])
    x_b2 = rng.uniform(0, 255, size=b2_shape).astype(np.float32)
    tm, mm = _parity_trial(pytorch_model, sess, x_b2)
    max_abs_err = max(max_abs_err, tm)
    mean_abs_err_sum += mm
    n_trials += 1

    # Trial bank 3: saturated inputs. Stresses the min-max normaliser's
    # divisor — `1 / (eps + max)` at max=0 (all-zero) and max=255 (all-255)
    # are the two most divergent cases arithmetically.
    for value in (0.0, 255.0):
        x_sat = np.full(input_shape, value, dtype=np.float32)
        tm, mm = _parity_trial(pytorch_model, sess, x_sat)
        max_abs_err = max(max_abs_err, tm)
        mean_abs_err_sum += mm
        n_trials += 1

    return {
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err_sum / n_trials,
        "tolerance": tolerance,
        "within_tolerance": max_abs_err < tolerance,
        "n_trials": n_trials,
        "trial_kinds": "random@bs=1 + random@bs=2 + saturated(0, 255)@bs=1",
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
        "--fp16",
        action="store_true",
        help=(
            "After FP32 export, convert the model to FP16 via "
            "onnxconverter_common (keep_io_types=True). Halves the "
            "artefact size (~106 MB → ~57 MB) at ~1e-3 max error."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help=(
            "Max absolute per-pixel error allowed between PyTorch and ONNX. "
            "Default: 1e-4 for FP32, 1e-3 for FP16. See LEARNINGS.md "
            "2026-04-17 for why FP16 needs a looser tolerance."
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of random-uniform parity trials (on top of fixed batched + saturated).",
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

    # why: tolerance defaults depend on FP precision. FP16 has ~6e-5 min-
    # normal precision, so FP32's 1e-4 gate is physically unreachable;
    # the 2026-04-17 LEARNINGS entry measured 7e-4 max error on naive FP16
    # conversion. 1e-3 gives a comfortable margin without being so loose
    # it masks regressions.
    tolerance = args.tolerance
    if tolerance is None:
        tolerance = 1e-3 if args.fp16 else 1e-4

    print(f"→ loading {args.checkpoint}")
    model = load_pytorch_model(args.checkpoint).to(EXPORT_DEVICE)

    print(f"→ exporting to {args.out} (opset {args.opset})")
    export_to_onnx(model, args.out, opset=args.opset)
    size_mb_fp32 = args.out.stat().st_size / 1e6
    print(f"  {size_mb_fp32:.1f} MB (FP32)")

    if args.fp16:
        print("→ converting to FP16 (naive, keep_io_types=True)")
        convert_to_fp16(args.out)
        size_mb = args.out.stat().st_size / 1e6
        print(f"  {size_mb:.1f} MB (FP16) — {(1 - size_mb / size_mb_fp32) * 100:.0f}% smaller")
    else:
        size_mb = size_mb_fp32

    print(f"→ validating PyTorch ↔ onnxruntime CPU parity "
          f"(tolerance={tolerance:.0e}, random-trials={args.trials} + bs=2 + saturated)")
    parity = validate_parity(model, args.out, n_random_trials=args.trials, tolerance=tolerance)
    print(f"  trials:       {parity['n_trials']} ({parity['trial_kinds']})")
    print(f"  max abs err:  {parity['max_abs_err']:.2e}")
    print(f"  mean abs err: {parity['mean_abs_err']:.2e}")

    report = {
        "checkpoint": str(args.checkpoint),
        "artefact": str(args.out),
        "precision": "fp16" if args.fp16 else "fp32",
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
