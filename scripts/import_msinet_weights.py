"""Import Kroner's pretrained MSI-Net weights from HuggingFace to a PyTorch state_dict.

One-time import script. The HuggingFace deposit at
https://huggingface.co/alexanderkroner/MSI-Net holds the SALICON-trained
MSI-Net as a TensorFlow SavedModel that was exported *frozen* — the weights
live as `Const` ops inside the inference graph, not as restorable
`tf.Variable` objects. Neither `tf.saved_model.load().variables` nor
`tf.keras.models.load_model(...).weights` surfaces them (both return empty
lists because there are literally no live variables in the re-loaded model).

This script walks the graph down to the real inference body (two
`PartitionedCall` indirections under `serving_default`), locates each conv
layer's kernel and bias as named `Const` ops, transposes kernels from TF's
HWIO layout to PyTorch's OIHW, and writes the resulting state_dict to
`weights/msinet_salicon.pt`.

Run once per contributor workstation. The output `.pt` file is gitignored;
contributors can either run this script or (eventually) pull a pre-computed
artefact from a GitHub release.

Requirements (installed via the [weights-import] extras group):
    tensorflow >= 2.15
    huggingface_hub >= 0.20

Usage:
    uv pip install --python .venv/bin/python -e '.[weights-import]'
    .venv/bin/python scripts/import_msinet_weights.py
    # → writes weights/msinet_salicon.pt and prints a parity-ready summary.

Optional:
    --discover    : print the full list of Const ops in the inference body
                    and exit without writing anything. Useful if the graph
                    structure changes on a newer SavedModel revision.
    --out PATH    : override the output path (default: weights/msinet_salicon.pt).
    --cache DIR   : override the HuggingFace snapshot cache dir.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Map Kroner-scope TF layer names (from his original model.py) to our PyTorch
# MSINet attribute names. The left side is what appears in the SavedModel's
# Const op names (e.g. "conv1/conv1_1/kernel"); the right side is what
# MSINet uses (e.g. "conv1_1.weight"). Keeping the mapping explicit catches
# silent renames or reorderings on newer HF revisions.
TF_TO_PYTORCH: dict[str, str] = {
    "conv1/conv1_1": "conv1_1",
    "conv1/conv1_2": "conv1_2",
    "conv2/conv2_1": "conv2_1",
    "conv2/conv2_2": "conv2_2",
    "conv3/conv3_1": "conv3_1",
    "conv3/conv3_2": "conv3_2",
    "conv3/conv3_3": "conv3_3",
    "conv4/conv4_1": "conv4_1",
    "conv4/conv4_2": "conv4_2",
    "conv4/conv4_3": "conv4_3",
    "conv5/conv5_1": "conv5_1",
    "conv5/conv5_2": "conv5_2",
    "conv5/conv5_3": "conv5_3",
    "aspp/conv1_1":  "aspp_b1",
    "aspp/conv1_2":  "aspp_b2",
    "aspp/conv1_3":  "aspp_b3",
    "aspp/conv1_4":  "aspp_b4",
    "aspp/conv1_5":  "aspp_b5",
    "aspp/conv2":    "aspp_proj",
    "decoder/conv1": "decoder_conv1",
    "decoder/conv2": "decoder_conv2",
    "decoder/conv3": "decoder_conv3",
    "decoder/conv4": "decoder_conv4",
}

# why: every layer in the MSI-Net SavedModel has both a kernel and a bias.
# This set is reserved for future-proofing — if a re-export ever drops a
# bias (e.g. decoder/conv4's bias is near-zero at ~2.4e-5), a layer can be
# added here so the importer doesn't error on the missing tensor.
LAYERS_WITHOUT_BIAS: set[str] = set()


def download_savedmodel(cache_dir: str | None) -> str:
    """Fetch the HuggingFace MSI-Net SavedModel and return the local path."""
    from huggingface_hub import snapshot_download

    # why: snapshot_download is idempotent — re-runs reuse the cached files
    # at HF's default cache location. Passing local_dir lets the caller pin
    # to a project-relative path if they want everything in one place.
    local_path = snapshot_download(
        repo_id="alexanderkroner/MSI-Net",
        local_dir=cache_dir,
    )
    return local_path


def _descend_partitioned_calls(graph, depth: int = 0, max_depth: int = 8):
    """Walk through `PartitionedCall` / `StatefulPartitionedCall` indirections
    until we reach a function body with more than a trivial number of ops.

    The HF SavedModel wraps the real inference body inside two nested
    PartitionedCalls (serving_default → __inference__wrapped_model_X →
    __inference_pruned_Y). We follow that chain automatically rather than
    hard-coding depth 2 so a future re-export with different wrapping still
    works.
    """
    ops = list(graph.get_operations())
    pcs = [o for o in ops if o.type in ("PartitionedCall", "StatefulPartitionedCall")]

    # A "real" body has many ops of many types; a wrapper has 3–5 ops dominated
    # by a single PartitionedCall. 20 is the cutoff — safely above any wrapper
    # size and well below the real body (300+ ops).
    if len(ops) >= 20 or not pcs or depth >= max_depth:
        return graph

    inner = graph._get_function(pcs[0].get_attr("f").name).graph
    return _descend_partitioned_calls(inner, depth + 1, max_depth)


def load_tf_constants(savedmodel_path: str) -> dict[str, "object"]:
    """Return {tf_const_name: numpy_array} for every named Const in the
    real inference body. Only includes float32 tensors — integer shape
    constants (dilation rates, reduction axes, etc.) are filtered out.
    """
    import tensorflow as tf

    loaded = tf.saved_model.load(savedmodel_path)
    serving = loaded.signatures["serving_default"]
    body = _descend_partitioned_calls(serving.graph)

    consts: dict[str, object] = {}
    for op in body.get_operations():
        if op.type != "Const":
            continue
        dtype = op.outputs[0].dtype
        if dtype != tf.float32:
            # why: filter out integer constants used for reshape/reduce axes
            # and similar plumbing; only float32 tensors are candidate weights.
            continue
        value = tf.make_ndarray(op.get_attr("value"))
        consts[op.name] = value

    print(f"  collected {len(consts)} float32 Const tensors from {body.name}")
    return consts


def print_discovery(consts: dict[str, "object"]) -> None:
    """Dump the float32 Const tensors we found. Run with --discover if the
    name-based mapping below breaks on a newer SavedModel revision.
    """
    print(f"Found {len(consts)} float32 constants:")
    print()
    print(f"  {'shape':24} name")
    print(f"  {'-' * 24} {'-' * 60}")
    for name in sorted(consts):
        value = consts[name]
        print(f"  {str(value.shape):24} {name}")


def build_state_dict(consts: dict[str, "object"]) -> dict[str, "object"]:
    """Map TF Const tensors onto MSINet's layer names and return a
    PyTorch-compatible state_dict.
    """
    import numpy as np
    import torch

    state_dict: dict[str, object] = {}
    missing: list[str] = []

    for tf_scope, pt_name in TF_TO_PYTORCH.items():
        kernel_key = f"{tf_scope}/kernel"
        bias_key = f"{tf_scope}/bias"

        if kernel_key not in consts:
            missing.append(kernel_key)
            continue

        # HWIO → OIHW transpose.
        # why: TensorFlow stores Conv2d kernels as (H, W, in_channels,
        # out_channels); PyTorch expects (out_channels, in_channels, H, W).
        # The permutation (3, 2, 0, 1) is the canonical fix.
        kernel = consts[kernel_key]
        kernel_pt = np.transpose(kernel, (3, 2, 0, 1))
        state_dict[f"{pt_name}.weight"] = torch.from_numpy(kernel_pt.copy())

        if bias_key in consts:
            bias = consts[bias_key]
            state_dict[f"{pt_name}.bias"] = torch.from_numpy(bias.copy())
        elif tf_scope in LAYERS_WITHOUT_BIAS:
            # Expected — the MSINet definition has bias=False for this layer.
            pass
        else:
            missing.append(bias_key)

    if missing:
        raise RuntimeError(
            "Could not locate the following Const tensors in the SavedModel:\n"
            + "\n".join(f"  - {k}" for k in missing)
            + "\n\nRun with --discover to see the full list of what is present."
        )

    print(f"  built state_dict with {len(state_dict)} tensors "
          f"({len(TF_TO_PYTORCH)} layers, one weight + optional bias each)")
    return state_dict


def verify(state_dict: dict[str, "object"]) -> None:
    """Instantiate MSINet, load the state_dict with strict=True, and run a
    single forward pass to confirm no layer was silently skipped.
    """
    import torch

    # why: deferred import so running `--discover` doesn't require our
    # own package to be installed.
    from foveacast_training.msinet import MSINet

    model = MSINet()
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    x = torch.randn(1, 3, 240, 320) * 50 + 128
    with torch.no_grad():
        y = model(x)

    print(f"  loaded {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  forward pass on (1, 3, 240, 320) → {tuple(y.shape)}, "
          f"range [{y.min():.4f}, {y.max():.4f}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--discover",
        action="store_true",
        help="print the SavedModel's float32 Const list and exit (no write).",
    )
    parser.add_argument(
        "--out",
        default="weights/msinet_salicon.pt",
        help="output path for the PyTorch state_dict (default: weights/msinet_salicon.pt).",
    )
    parser.add_argument(
        "--cache",
        default=None,
        help="override the HuggingFace snapshot cache directory.",
    )
    args = parser.parse_args()

    print("→ Downloading HuggingFace SavedModel (alexanderkroner/MSI-Net)...")
    savedmodel_path = download_savedmodel(args.cache)
    print(f"  local path: {savedmodel_path}")

    print("→ Walking inference graph to find weight Const tensors...")
    consts = load_tf_constants(savedmodel_path)

    if args.discover:
        print_discovery(consts)
        return

    print("→ Building PyTorch state_dict...")
    state_dict = build_state_dict(consts)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(state_dict, out_path)
    print(f"✓ Wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")

    print("→ Verifying by loading into MSINet with strict=True...")
    verify(state_dict)
    print("✓ Import complete. Run `pytest tests/test_msinet_parity.py` "
          "to confirm numerical parity with the reference forward pass.")


if __name__ == "__main__":
    main()
