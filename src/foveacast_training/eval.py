"""Evaluate a saliency-prediction checkpoint on UEyes.

Phase 6 of issue #1: produce CC, KLD, NSS on the held-out UEyes test
split. Optionally compares two checkpoints side-by-side — the standard
use is "fine-tuned model vs stock SALICON-pretrained MSI-Net" to answer
"does fine-tuning actually help?"

Run:
    # Evaluate the Phase 5 best checkpoint on the test split:
    .venv/bin/python -m foveacast_training.eval \\
        --checkpoint runs/full-*/best.pt

    # Compare fine-tuned vs stock side-by-side:
    .venv/bin/python -m foveacast_training.eval \\
        --checkpoint runs/full-*/best.pt \\
        --compare weights/msinet_salicon.pt

The test split (108 images) is the upstream-defined Test set from
`image_types.csv` — it was never touched during training. Val split
(188 images) is available via `--split val` for sanity checks, but
the gate is "beats stock on test."

NSS requires a binary fixation map (fixmaps_{window}), CC and KLD use the
continuous heatmap (heatmaps_{window}). The script loads both Dataset
instances and iterates them in lock-step, relying on UEyesDataset's
deterministic row ordering so the image-to-metric alignment stays correct.

The `--time-window` flag (default 3s) picks which viewing-duration ground
truth to evaluate against. For multi-duration models (issue #19), each
model is evaluated at its matched window — the 1s model against 1s
ground truth, 7s model against 7s ground truth — not cross-window.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Callable

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader

from foveacast_training.losses import (
    correlation_coefficient,
    kl_divergence_saliency,
    normalized_scanpath_saliency,
)
from foveacast_training.msinet import MSINet
from foveacast_training.ueyes_dataset import UEyesDataset


def _auto_device() -> torch.device:
    """MPS → CUDA → CPU, matching train.py and CLAUDE.md."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path: Path, device: torch.device) -> MSINet:
    """Construct MSINet, load a state_dict from `checkpoint_path` with
    `strict=True`, move to `device`, and put in eval mode.

    Extracted as a reusable helper so Phase 7 (qualitative benchmark
    rendering) and any future inference-only script can reuse the exact
    loading convention — `strict=True` + `weights_only=True` — without
    copy-pasting the five-line setup.
    """
    model = MSINet()
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def make_inferencer(
    checkpoint_path: Path, device: torch.device
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a callable `(images_tensor) -> pred_tensor` that dispatches
    to either PyTorch or ONNX Runtime based on file extension.

    Why auto-detect over a separate `--onnx` flag: the eval pipeline
    (split loading, metric computation, comparison-table formatting) is
    identical for both PyTorch and ONNX inference — only the forward
    pass differs. Unifying at the inferencer boundary keeps the rest
    of the module oblivious to which backend ran, which is what lets
    the same `evaluate_checkpoint` function grade stock PyTorch vs
    FP16 ONNX vs INT8 ONNX side-by-side without branching.

    ONNX inference is always CPU (CPUExecutionProvider) regardless of
    the `device` argument — the CPU EP has the most complete op
    coverage and the 108-image eval is fast enough that MPS throughput
    doesn't matter. Predictions are moved to `device` on return so the
    downstream metric computations stay on-device.
    """
    if checkpoint_path.suffix == ".onnx":
        sess = ort.InferenceSession(
            str(checkpoint_path), providers=["CPUExecutionProvider"]
        )

        def run_onnx(images: torch.Tensor) -> torch.Tensor:
            x_np = images.detach().cpu().numpy().astype(np.float32)
            y_np = sess.run(["output"], {"input": x_np})[0]
            return torch.from_numpy(y_np).to(device)

        return run_onnx

    # .pt / .pth PyTorch checkpoint — load the usual way.
    model = load_model(checkpoint_path, device)
    return model


def evaluate_checkpoint(
    checkpoint_path: Path,
    split: str,
    data_root: Path,
    device: torch.device,
    batch_size: int = 4,
    time_window: str = "3s",
) -> dict[str, dict[str, float]]:
    """Load a checkpoint, run the split through it, return per-metric
    mean + stdev computed over per-batch values.

    Parameters
    ----------
    checkpoint_path : Path
        state_dict .pt file produced by `scripts/import_msinet_weights.py`
        (stock) or `src/foveacast_training/train.py --full` (fine-tuned).
    split : str
        'val' or 'test'. Phase 6's gate uses 'test'.
    data_root : Path
        Points at the unpacked `UEyes_dataset/` tree.
    device : torch.device
        Auto-detected by `_auto_device`.
    batch_size : int
        For eval. 4 is a safe default for MSINet on M4 MPS.
    time_window : str
        One of '1s', '3s', '7s'. Picks the viewing-duration variant for
        both heatmap (CC/KLD target) and fixmap (NSS target). Models
        trained against `heatmaps_{w}` should be evaluated with
        `time_window=w` — cross-window eval answers a different question
        ("does the 1s model agree with 3s ground truth?") and isn't the
        default.
    """
    # why: make_inferencer dispatches to PyTorch or ONNX Runtime based
    # on file extension, so the rest of this function is backend-agnostic.
    # For PyTorch .pt files it still goes through load_model + strict=True
    # so a stale state_dict fails loudly rather than quietly evaluating
    # 108 images with most layers at init.
    model = make_inferencer(checkpoint_path, device)

    # Two datasets — heatmap target for CC/KLD, fixmap target for NSS,
    # both at the same viewing-duration window. Phase 0 + 3 guarantee
    # the split is deterministic given the same seed/fraction, so the
    # two datasets' filenames() lists are identical.
    heatmap_variant = f"heatmaps_{time_window}"
    fixmap_variant = f"fixmaps_{time_window}"
    ds_heatmap = UEyesDataset(data_root, split=split, saliency_variant=heatmap_variant)
    ds_fixmap = UEyesDataset(data_root, split=split, saliency_variant=fixmap_variant)
    if ds_heatmap.filenames() != ds_fixmap.filenames():
        raise RuntimeError(
            "heatmap and fixmap datasets disagree on split filenames — "
            "UEyesDataset determinism assumption broken"
        )

    # shuffle=False + num_workers=0 to keep the eval deterministic and
    # avoid the MPS fork foot-gun documented in train.py FULL_CONFIG.
    loader_hm = DataLoader(ds_heatmap, batch_size=batch_size, shuffle=False, num_workers=0)
    loader_fm = DataLoader(ds_fixmap, batch_size=batch_size, shuffle=False, num_workers=0)

    metrics: dict[str, list[float]] = {"cc": [], "kld": [], "nss": []}
    with torch.no_grad():
        for (images, heatmaps), (_, fixmaps) in zip(loader_hm, loader_fm, strict=True):
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            fixmaps = fixmaps.to(device)

            pred = model(images)
            metrics["cc"].append(correlation_coefficient(pred, heatmaps).item())
            metrics["kld"].append(kl_divergence_saliency(pred, heatmaps).item())
            metrics["nss"].append(normalized_scanpath_saliency(pred, fixmaps).item())

    return {
        name: {
            "mean": statistics.mean(vals),
            "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "n_batches": len(vals),
            "n_images": len(ds_heatmap),
        }
        for name, vals in metrics.items()
    }


def _format_metric_row(
    name: str,
    main: dict[str, float],
    compare: dict[str, float] | None,
) -> str:
    """Render one row of the metrics table."""
    # why: CC higher better (→ ↑ means fine-tune helped on CC),
    # KLD lower better (→ ↓ means fine-tune helped on KLD),
    # NSS higher better (→ ↑ means fine-tune helped on NSS). The
    # sign of the delta is the signal.
    row = f"{name:6}  {main['mean']:>7.4f} ± {main['stdev']:.3f}"
    if compare is not None:
        row += f"  {compare['mean']:>7.4f} ± {compare['stdev']:.3f}"
        delta = main["mean"] - compare["mean"]
        if name in {"cc", "nss"}:
            marker = "better" if delta > 0 else "worse" if delta < 0 else "same"
        elif name == "kld":
            marker = "better" if delta < 0 else "worse" if delta > 0 else "same"
        else:
            marker = "?"
        row += f"  {delta:+.4f}  {marker}"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Primary checkpoint to evaluate (typically runs/full-*/best.pt).",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Optional second checkpoint for side-by-side (typically weights/msinet_salicon.pt).",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Split to evaluate on. Phase 6 gate uses 'test'.",
    )
    parser.add_argument(
        "--time-window",
        choices=["1s", "3s", "7s"],
        default="3s",
        help=(
            "Viewing-duration variant for the ground-truth heatmap + fixmap. "
            "Default 3s matches the v0.1.0 shipped model; use 1s / 7s when "
            "evaluating the multi-duration models from issue #19."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/ueyes/UEyes_dataset"),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path. If unset, results print only to stdout.",
    )
    args = parser.parse_args()

    device = _auto_device()
    print(f"→ device: {device}")
    print(f"→ split:  {args.split}")
    print(f"→ window: {args.time_window}  "
          f"(heatmaps_{args.time_window} + fixmaps_{args.time_window})")

    print(f"→ evaluating {args.checkpoint}...")
    main_metrics = evaluate_checkpoint(
        args.checkpoint, args.split, args.data_root, device, args.batch_size,
        time_window=args.time_window,
    )

    compare_metrics: dict[str, dict[str, float]] | None = None
    if args.compare is not None:
        print(f"→ evaluating {args.compare}...")
        compare_metrics = evaluate_checkpoint(
            args.compare, args.split, args.data_root, device, args.batch_size,
            time_window=args.time_window,
        )

    print()
    header = f"{'metric':6}  {args.checkpoint.name:>13}"
    if compare_metrics is not None:
        header += f"  {args.compare.name:>13}  {'delta':>8}"
    print(header)
    print("─" * len(header))
    for name in ("cc", "kld", "nss"):
        print(
            _format_metric_row(
                name,
                main_metrics[name],
                compare_metrics[name] if compare_metrics else None,
            )
        )

    print()
    print(f"n_images = {main_metrics['cc']['n_images']} "
          f"(split={args.split}, batch_size={args.batch_size})")

    if args.out is not None:
        payload: dict[str, object] = {
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "time_window": args.time_window,
            "metrics": main_metrics,
        }
        if compare_metrics is not None:
            payload["compare_checkpoint"] = str(args.compare)
            payload["compare_metrics"] = compare_metrics
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"✓ wrote {args.out}")


if __name__ == "__main__":
    main()
