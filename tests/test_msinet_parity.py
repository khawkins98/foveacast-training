"""Numerical-parity test: PyTorch MSINet vs reference TF SavedModel.

This is the Phase 2 gate from #1 — "forward pass matches the Keras reference"
means *this test passes*. It runs the same fixed-seed random input through
both the PyTorch port and Kroner's pretrained HF SavedModel, then checks the
outputs match within float tolerance.

The test skips cleanly if the weights file or TensorFlow are missing, so a
contributor who has only done the core install can still run the test suite
without failures — they just won't exercise this particular gate. To run the
test properly:

    uv pip install --python .venv/bin/python -e '.[weights-import,dev]'
    .venv/bin/python scripts/import_msinet_weights.py
    .venv/bin/pytest tests/test_msinet_parity.py -v

Tolerance is intentionally loose (atol=1e-3, rtol=1e-3) at first. If the
port reproduces the reference bit-closely, we can tighten later. If it
drifts badly, that's a signal something is misaligned — pool padding,
dilation semantics, bilinear corner handling — and the test's reported
statistics (mean/max absolute error, histogram of deltas) are what we
debug against.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from foveacast_training.msinet import MSINet

# Resolve paths relative to the repo root. why: tests can be invoked from
# either the repo root or from within tests/, and hard-coded relative paths
# break under the latter.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "msinet_salicon.pt"
WEIGHTS_PATH = Path(os.environ.get("MSINET_WEIGHTS", str(DEFAULT_WEIGHTS)))

try:
    import tensorflow as tf  # noqa: F401
    from huggingface_hub import snapshot_download  # noqa: F401

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


pytestmark = [
    pytest.mark.skipif(
        not WEIGHTS_PATH.exists(),
        reason=(
            f"{WEIGHTS_PATH} missing — run `python scripts/import_msinet_weights.py` "
            f"first (requires `[weights-import]` extras)."
        ),
    ),
    pytest.mark.skipif(
        not TF_AVAILABLE,
        reason=(
            "tensorflow + huggingface_hub not installed "
            "(run `uv pip install -e '.[weights-import]'`)."
        ),
    ),
]


@pytest.fixture(scope="module")
def fixed_input() -> np.ndarray:
    """A reproducible random input in [0, 255] RGB, shape (1, 240, 320, 3)
    NHWC — TF's native layout. Pytorch-side tests transpose on the fly.

    why: fixed seed means the test is deterministic across runs and CI. The
    specific values don't matter for parity — any uniform random input
    exercises all ops.
    """
    rng = np.random.default_rng(seed=0)
    return rng.uniform(0, 255, size=(1, 240, 320, 3)).astype(np.float32)


@pytest.fixture(scope="module")
def tf_reference():
    """Load Kroner's HF SavedModel once for the whole test module. why: a TF
    SavedModel load is slow (seconds); reusing it across parametrised cases
    is free.
    """
    import tensorflow as tf
    from huggingface_hub import snapshot_download

    local_path = snapshot_download(repo_id="alexanderkroner/MSI-Net")
    return tf.saved_model.load(local_path)


@pytest.fixture(scope="module")
def torch_model() -> MSINet:
    model = MSINet()
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _run_tf(tf_model, x_nhwc: np.ndarray) -> np.ndarray:
    """Forward pass through the TF SavedModel. Handles the common cases:
    - A signatures dict with 'serving_default'.
    - A plain callable.
    Returns the output as a numpy array in NHWC layout (squeezed to NHW1 if
    applicable, then squeezed to HW for easier comparison downstream).
    """
    import tensorflow as tf

    x_tf = tf.constant(x_nhwc)

    # Try the canonical Keras-exported signature first. This is what most
    # from_pretrained_keras-saved models expose.
    signatures = getattr(tf_model, "signatures", None)
    if signatures and "serving_default" in signatures:
        fn = signatures["serving_default"]
        # The signature's input name can be anything — grab the first
        # positional arg of the function's structured signature.
        input_name = list(fn.structured_input_signature[1].keys())[0]
        out = fn(**{input_name: x_tf})
        # signatures return a dict of output tensors; usually one entry.
        assert len(out) == 1, f"unexpected multi-output signature: {list(out)}"
        y = list(out.values())[0].numpy()
    else:
        # Fall back to calling the loaded object directly.
        y = tf_model(x_tf).numpy()

    return y


def test_parity_on_fixed_input(tf_reference, torch_model, fixed_input):
    """Run the same input through both models and assert outputs match.

    Reports mean/max absolute error and the 99th-percentile error even on
    success, so a commit that barely passes gets a visible regression signal
    next time.
    """
    x_nhwc = fixed_input
    x_nchw = np.transpose(x_nhwc, (0, 3, 1, 2))  # (N, 3, H, W)

    y_tf_raw = _run_tf(tf_reference, x_nhwc)
    with torch.no_grad():
        y_pt_raw = torch_model(torch.from_numpy(x_nchw)).numpy()

    # Normalise layouts before comparing. TF export could be (N, H, W, 1)
    # NHWC or (N, 1, H, W) NCHW depending on what data_format Kroner exported
    # with; PyTorch is always (N, 1, H, W). Squeezing to (H, W) removes both
    # the channel dim and the batch dim.
    y_tf = np.squeeze(y_tf_raw)
    y_pt = np.squeeze(y_pt_raw)

    assert y_tf.shape == y_pt.shape, (
        f"output shape mismatch: TF {y_tf.shape} vs PyTorch {y_pt.shape}"
    )

    diff = np.abs(y_tf - y_pt)
    stats = {
        "mean_abs_err": float(diff.mean()),
        "max_abs_err": float(diff.max()),
        "p99_abs_err": float(np.percentile(diff, 99)),
        "tf_output_range": (float(y_tf.min()), float(y_tf.max())),
        "pt_output_range": (float(y_pt.min()), float(y_pt.max())),
    }

    # Print unconditionally so `pytest -s` shows the numbers even when the
    # test passes. Parity numbers are the interesting signal either way.
    print()
    print(f"  mean abs err:    {stats['mean_abs_err']:.2e}")
    print(f"  max abs err:     {stats['max_abs_err']:.2e}")
    print(f"  p99 abs err:     {stats['p99_abs_err']:.2e}")
    print(f"  TF output range: [{stats['tf_output_range'][0]:.4f}, "
          f"{stats['tf_output_range'][1]:.4f}]")
    print(f"  PT output range: [{stats['pt_output_range'][0]:.4f}, "
          f"{stats['pt_output_range'][1]:.4f}]")

    # Primary gate: close enough to be considered parity. The first
    # successful run produced mean ~1.1e-7 and max ~1.4e-6 — float32
    # machine-epsilon territory. atol=1e-5 leaves roughly an order of
    # magnitude of headroom while still catching structural regressions
    # (a re-introduced bilinear-semantics mismatch would blow this by
    # 3–4 orders, as the initial port did).
    np.testing.assert_allclose(
        y_pt, y_tf,
        atol=1e-5,
        rtol=1e-5,
        err_msg="PyTorch port diverges from TF reference beyond tolerance",
    )


def test_output_range_sanity(torch_model, fixed_input):
    """The PyTorch forward pass should produce normalised output in [0, 1]
    without needing the TF reference. Cheap, fast, always runs.
    """
    x_nchw = np.transpose(fixed_input, (0, 3, 1, 2))
    with torch.no_grad():
        y = torch_model(torch.from_numpy(x_nchw))

    # _normalize scales per image to [0, 1]; the exact max depends on
    # eps, so allow a tiny slack below 1.0.
    assert y.min().item() >= 0.0
    assert y.max().item() <= 1.0 + 1e-6
