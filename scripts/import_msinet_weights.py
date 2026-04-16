"""Import Kroner's pretrained MSI-Net weights from HuggingFace to a PyTorch state_dict.

One-time import script. The HuggingFace deposit at
https://huggingface.co/alexanderkroner/MSI-Net holds the SALICON-trained
MSI-Net as a TensorFlow 2.15 SavedModel. This script walks that SavedModel's
variables, reorders the conv kernels from TF's HWIO layout to PyTorch's OIHW,
and writes the resulting state_dict to weights/msinet_salicon.pt for use by
`foveacast_training.msinet.MSINet`.

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
    --discover    : print the full list of TF variables and exit without
                    writing anything. Useful if the variable-order assumption
                    below breaks on a newer SavedModel revision.
    --out PATH    : override the output path (default: weights/msinet_salicon.pt).
    --cache DIR   : override the HuggingFace snapshot cache dir.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Layer-order contract between this importer and foveacast_training.msinet.
# MSINet declares its Conv2d layers in this exact order in __init__; the
# HuggingFace SavedModel's variables appear in the same order because both
# derive from the same reference network definition. Keeping the list
# explicit makes a mismatch easy to diagnose.
#
# Each entry is (layer_name_in_MSINet, expected_HWIO_kernel_shape). We pair
# kernels and biases positionally — kernel N is followed by bias N in the
# TF variable list, which is the standard Keras Dense/Conv ordering.
EXPECTED_LAYERS: list[tuple[str, tuple[int, int, int, int]]] = [
    ("conv1_1",       (3, 3,    3,   64)),
    ("conv1_2",       (3, 3,   64,   64)),
    ("conv2_1",       (3, 3,   64,  128)),
    ("conv2_2",       (3, 3,  128,  128)),
    ("conv3_1",       (3, 3,  128,  256)),
    ("conv3_2",       (3, 3,  256,  256)),
    ("conv3_3",       (3, 3,  256,  256)),
    ("conv4_1",       (3, 3,  256,  512)),
    ("conv4_2",       (3, 3,  512,  512)),
    ("conv4_3",       (3, 3,  512,  512)),
    ("conv5_1",       (3, 3,  512,  512)),
    ("conv5_2",       (3, 3,  512,  512)),
    ("conv5_3",       (3, 3,  512,  512)),
    ("aspp_b1",       (1, 1, 1280,  256)),
    ("aspp_b2",       (3, 3, 1280,  256)),
    ("aspp_b3",       (3, 3, 1280,  256)),
    ("aspp_b4",       (3, 3, 1280,  256)),
    ("aspp_b5",       (1, 1, 1280,  256)),
    ("aspp_proj",     (1, 1, 1280,  256)),
    ("decoder_conv1", (3, 3,  256,  128)),
    ("decoder_conv2", (3, 3,  128,   64)),
    ("decoder_conv3", (3, 3,   64,   32)),
    ("decoder_conv4", (3, 3,   32,    1)),
]


def download_savedmodel(cache_dir: str | None) -> str:
    """Fetch the HuggingFace MSI-Net SavedModel and return the local path."""
    from huggingface_hub import snapshot_download

    # why: snapshot_download is idempotent — re-runs reuse the cached files
    # at HF's default cache location (~/.cache/huggingface). Passing
    # local_dir lets the caller pin to a project-relative path if they
    # want everything in one place.
    local_path = snapshot_download(
        repo_id="alexanderkroner/MSI-Net",
        local_dir=cache_dir,
    )
    return local_path


def load_tf_variables(savedmodel_path: str) -> list[tuple[str, "object"]]:
    """Return every variable in the SavedModel as (name, numpy_array) pairs,
    preserving declaration order.

    Tries two loaders in priority order:

    1. `tf.keras.models.load_model` — the right tool for HF-hosted Keras
       SavedModels. Exposes weights via `.weights`, which enumerates both
       trainable and non-trainable parameters in declaration order.
    2. `tf.saved_model.load` — the generic fallback for non-Keras
       SavedModels. Exposes weights via `.variables`.

    The first loader that returns a non-empty list wins. If both return
    empty, raises with the per-loader diagnostic so the user sees what
    happened.
    """
    import tensorflow as tf  # imported lazily — not a runtime dep of this repo

    attempts: list[tuple[str, object]] = []

    # Attempt 1: Keras loader. compile=False skips reconstructing the optimiser
    # (we only need forward-pass weights), and tolerates SavedModels saved
    # without training-time metadata.
    try:
        model = tf.keras.models.load_model(savedmodel_path, compile=False)
        weights = list(model.weights)
        attempts.append(("tf.keras.models.load_model → model.weights", weights))
        if weights:
            print(f"  loader: tf.keras.models.load_model ({len(weights)} weights)")
            return [(w.name, w.numpy()) for w in weights]
    except Exception as e:
        attempts.append(("tf.keras.models.load_model", f"raised {type(e).__name__}: {e}"))

    # Attempt 2: low-level SavedModel.
    try:
        loaded = tf.saved_model.load(savedmodel_path)
        variables = list(loaded.variables)
        attempts.append(("tf.saved_model.load → loaded.variables", variables))
        if variables:
            print(f"  loader: tf.saved_model.load ({len(variables)} variables)")
            return [(v.name, v.numpy()) for v in variables]
    except Exception as e:
        attempts.append(("tf.saved_model.load", f"raised {type(e).__name__}: {e}"))

    # Both loaders produced nothing. Report what happened so the user can
    # paste the output back for diagnosis.
    lines = [
        "No loader returned any variables. Per-loader outcome:",
        "",
    ]
    for name, result in attempts:
        if isinstance(result, list):
            lines.append(f"  {name}: found {len(result)}")
        else:
            lines.append(f"  {name}: {result}")
    raise RuntimeError("\n".join(lines))


def print_discovery(variables: list[tuple[str, "object"]]) -> None:
    """Verbose dump of what we found in the SavedModel. Run with --discover
    if the positional assignment below breaks on a newer revision of the
    weights.
    """
    print(f"Found {len(variables)} variables in the SavedModel:")
    print()
    print(f"  {'index':>5} {'shape':24} {'dtype':10} {'name'}")
    print(f"  {'-' * 5} {'-' * 24} {'-' * 10} {'-' * 40}")
    for i, (name, arr) in enumerate(variables):
        shape = str(tuple(arr.shape))
        dtype = str(arr.dtype)
        print(f"  {i:5d} {shape:24} {dtype:10} {name}")


def build_state_dict(variables: list[tuple[str, "object"]]) -> dict[str, "object"]:
    """Pair up TF variables with MSINet's expected layers and return a
    PyTorch-compatible state_dict.
    """
    import numpy as np
    import torch

    # Filter to 4D kernels and 1D biases. A well-formed MSI-Net SavedModel
    # contains exactly 23 of each (one kernel + one bias per Conv2d layer).
    kernels = [(n, a) for n, a in variables if a.ndim == 4]
    biases = [(n, a) for n, a in variables if a.ndim == 1]

    if len(kernels) != len(EXPECTED_LAYERS) or len(biases) != len(EXPECTED_LAYERS):
        raise RuntimeError(
            f"Expected {len(EXPECTED_LAYERS)} conv kernels and {len(EXPECTED_LAYERS)} "
            f"biases; found {len(kernels)} kernels and {len(biases)} biases. "
            f"Run with --discover to see the full variable list."
        )

    state_dict: dict[str, object] = {}

    # why: iterate the three lists in lock-step. Shape assertion at every
    # step catches any ordering drift between TF's variable enumeration
    # and MSINet's declaration order — silent misalignment would produce
    # a model that loads without error but predicts garbage.
    for (layer_name, expected_hwio), (k_name, k_arr), (b_name, b_arr) in zip(
        EXPECTED_LAYERS, kernels, biases, strict=True
    ):
        if tuple(k_arr.shape) != expected_hwio:
            raise RuntimeError(
                f"Kernel shape mismatch at layer '{layer_name}': "
                f"expected HWIO {expected_hwio}, got {tuple(k_arr.shape)} "
                f"(TF variable: {k_name}). Run with --discover for context."
            )
        expected_bias = (expected_hwio[3],)
        if tuple(b_arr.shape) != expected_bias:
            raise RuntimeError(
                f"Bias shape mismatch at layer '{layer_name}': "
                f"expected {expected_bias}, got {tuple(b_arr.shape)} "
                f"(TF variable: {b_name})."
            )

        # HWIO → OIHW.
        # why: TensorFlow stores Conv2d kernels as (H, W, in_channels,
        # out_channels); PyTorch expects (out_channels, in_channels, H, W).
        # The numpy.transpose permutation (3, 2, 0, 1) is the canonical fix.
        k_pyt = np.transpose(k_arr, (3, 2, 0, 1))

        state_dict[f"{layer_name}.weight"] = torch.from_numpy(k_pyt.copy())
        state_dict[f"{layer_name}.bias"] = torch.from_numpy(b_arr.copy())

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
        help="print the SavedModel's variable list and exit (no write).",
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

    print("→ Loading TF variables...")
    variables = load_tf_variables(savedmodel_path)

    if args.discover:
        print_discovery(variables)
        return

    print(f"→ Building PyTorch state_dict from {len(variables)} TF variables...")
    state_dict = build_state_dict(variables)

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
