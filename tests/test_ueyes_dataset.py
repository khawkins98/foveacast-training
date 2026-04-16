"""Tests for the UEyes PyTorch Dataset (Phase 3).

Tests run against the real unpacked UEyes dataset — not synthetic fixtures.
Skipped cleanly when the dataset isn't present (contributors who haven't
run `data/fetch.sh` yet, CI machines without the 13 GB zip).

The reference counts here come from the Phase 0 inventory documented in
`data/README.md`:

    total:  1,980 images (495 per category × 4 categories)
    train:  1,872 upstream-Train
    test:     108 upstream-Test

With default val_fraction=0.1 stratified by category:
    train: 1,684 (421 per category)
    val:     188 (47 per category)
    test:    108 (27 per category)
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import pytest
import torch

from foveacast_training.ueyes_dataset import (
    DEFAULT_INPUT_SIZE,
    UEyesDataset,
    _carve_validation_split,
    _load_split_csv,
)

# Same "repo-relative path" approach as the parity test. UEyes lives under
# data/ueyes/UEyes_dataset/ after data/fetch.sh runs; the env var lets CI or
# out-of-tree users point elsewhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_UEYES_ROOT = REPO_ROOT / "data" / "ueyes" / "UEyes_dataset"
UEYES_ROOT = Path(os.environ.get("UEYES_ROOT", str(DEFAULT_UEYES_ROOT)))


# why: module-level skip rather than per-test. If the dataset is missing,
# none of these tests can run — skipping the whole module produces one
# clear reason instead of 12 identical skip messages.
pytestmark = pytest.mark.skipif(
    not (UEYES_ROOT / "image_types.csv").is_file(),
    reason=(
        f"UEyes dataset not found at {UEYES_ROOT} — "
        "run `bash data/fetch.sh` first, or set UEYES_ROOT."
    ),
)


@pytest.fixture(scope="module")
def ueyes_root() -> Path:
    return UEYES_ROOT


@pytest.fixture(scope="module")
def train_ds(ueyes_root: Path) -> UEyesDataset:
    return UEyesDataset(ueyes_root, split="train")


@pytest.fixture(scope="module")
def val_ds(ueyes_root: Path) -> UEyesDataset:
    return UEyesDataset(ueyes_root, split="val")


@pytest.fixture(scope="module")
def test_ds(ueyes_root: Path) -> UEyesDataset:
    return UEyesDataset(ueyes_root, split="test")


# ---------- split correctness ---------------------------------------------


def test_split_sizes_sum_to_total(train_ds, val_ds, test_ds):
    """Every image in image_types.csv ends up in exactly one split."""
    total = len(train_ds) + len(val_ds) + len(test_ds)
    assert total == 1980, f"expected 1,980 images across all splits, got {total}"


def test_test_split_matches_upstream(test_ds):
    """The test split is exactly what upstream labelled 'Test'."""
    assert len(test_ds) == 108


def test_train_val_stratified_by_category(train_ds, val_ds):
    """Both train and val have the same per-category count — stratification."""
    train_by_cat = Counter(train_ds.categories())
    val_by_cat = Counter(val_ds.categories())

    # All four categories present in both splits.
    assert set(train_by_cat) == {"desktop", "mobile", "poster", "web"}
    assert set(val_by_cat) == {"desktop", "mobile", "poster", "web"}

    # Same count per category (stratified hold-out).
    train_counts = set(train_by_cat.values())
    val_counts = set(val_by_cat.values())
    assert len(train_counts) == 1, f"train not stratified: {dict(train_by_cat)}"
    assert len(val_counts) == 1, f"val not stratified: {dict(val_by_cat)}"


def test_no_filename_overlap_across_splits(train_ds, val_ds, test_ds):
    """No image appears in more than one split."""
    all_filenames = train_ds.filenames() + val_ds.filenames() + test_ds.filenames()
    assert len(all_filenames) == len(set(all_filenames))


def test_val_split_reproducible(ueyes_root):
    """Same seed + same CSV → same val filenames. Regression guard against
    accidentally making the split depend on global RNG state.
    """
    a = UEyesDataset(ueyes_root, split="val")
    b = UEyesDataset(ueyes_root, split="val")
    assert a.filenames() == b.filenames()


def test_val_seed_changes_split(ueyes_root):
    """Changing val_seed changes the val membership (otherwise the seed
    plumbing is broken).
    """
    default = UEyesDataset(ueyes_root, split="val", val_seed=42)
    alt = UEyesDataset(ueyes_root, split="val", val_seed=99)
    assert default.filenames() != alt.filenames()


# ---------- sample shape and dtype ----------------------------------------


def test_sample_shapes_and_dtypes(train_ds):
    """The image/saliency tensors match MSINet's input contract at the
    default 240x320 size.
    """
    image, saliency = train_ds[0]

    assert isinstance(image, torch.Tensor)
    assert isinstance(saliency, torch.Tensor)
    assert image.dtype == torch.float32
    assert saliency.dtype == torch.float32

    assert tuple(image.shape) == (3, *DEFAULT_INPUT_SIZE)
    assert tuple(saliency.shape) == (1, *DEFAULT_INPUT_SIZE)


def test_sample_value_ranges(train_ds):
    """Stimulus stays in [0, 255] (MSINet does mean subtraction internally);
    saliency stays in [0, 1].
    """
    image, saliency = train_ds[0]

    assert image.min().item() >= 0.0
    assert image.max().item() <= 255.0
    assert saliency.min().item() >= 0.0
    # Saliency can hit slightly below 1.0 because we divide uint8/255 —
    # max observed in Phase 3 smoke was 0.996. Allow up to 1.0 inclusive.
    assert saliency.max().item() <= 1.0 + 1e-6


def test_custom_input_size(ueyes_root):
    """input_size is respected end-to-end; output tensors reflect it."""
    ds = UEyesDataset(ueyes_root, split="val", input_size=(120, 160))
    image, saliency = ds[0]
    assert tuple(image.shape) == (3, 120, 160)
    assert tuple(saliency.shape) == (1, 120, 160)


# ---------- saliency-variant selection ------------------------------------


def test_overlay_heatmaps_filename_prefix_handled(ueyes_root):
    """`overlay_heatmaps_*` is the one subdir where filenames carry an
    'overlay_' prefix. The loader must handle it without a branch on
    __getitem__.
    """
    ds = UEyesDataset(ueyes_root, split="val", saliency_variant="overlay_heatmaps_3s")
    image, saliency = ds[0]
    assert tuple(image.shape[1:]) == tuple(saliency.shape[1:])


def test_fixmaps_variant_loads(ueyes_root):
    """Non-default saliency_variant resolves correctly."""
    ds = UEyesDataset(ueyes_root, split="val", saliency_variant="fixmaps_1s")
    image, saliency = ds[0]
    assert saliency.dtype == torch.float32
    # fixmaps are binary at source — after normalisation they should be
    # mostly 0 with some non-zero points. This test just confirms the
    # variant is readable; content validation is out of scope.
    assert saliency.min().item() >= 0.0


# ---------- error handling ------------------------------------------------


def test_unknown_split_rejected(ueyes_root):
    with pytest.raises(ValueError, match="split must be"):
        UEyesDataset(ueyes_root, split="validation")  # type: ignore[arg-type]


def test_invalid_saliency_variant_rejected(ueyes_root):
    """A typo or unknown variant is caught by the enum check, not by
    stumbling into a FileNotFoundError on a path the user didn't name.
    """
    with pytest.raises(ValueError, match="saliency_variant must be one of"):
        UEyesDataset(ueyes_root, split="val", saliency_variant="heatmap_3s")


def test_val_fraction_bounds_rejected(ueyes_root):
    """val_fraction must be strictly in (0.0, 1.0). Catches typos like
    `val_fraction: 10` (meant 10%) and obvious nonsense like negatives.
    """
    for bad in (1.5, 0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="val_fraction must be"):
            UEyesDataset(ueyes_root, split="val", val_fraction=bad)


def test_missing_root_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        UEyesDataset(tmp_path / "does-not-exist", split="val")


# ---------- smaller guarantees that Phase 4 will rely on -------------------


def test_sample_tensors_are_contiguous(train_ds):
    """PyTorch collate and ONNX tracing both want contiguous tensors.
    Guards against someone silently dropping the np.ascontiguousarray call
    in _preprocess_stimulus / _preprocess_saliency.
    """
    image, saliency = train_ds[0]
    assert image.is_contiguous()
    assert saliency.is_contiguous()


def test_dataset_is_pickleable(val_ds):
    """num_workers>0 in DataLoader forks and pickles the Dataset. A future
    edit that stashes an open file handle or a TF session on self would
    break this. Test catches it up front.
    """
    import pickle

    blob = pickle.dumps(val_ds)
    restored = pickle.loads(blob)
    assert restored.filenames() == val_ds.filenames()
    assert len(restored) == len(val_ds)


def test_repr_is_informative(val_ds):
    text = repr(val_ds)
    # Not a strict format — just confirm the four load-bearing pieces are
    # visible. TensorBoard run names will splice this in.
    assert "UEyesDataset" in text
    assert "split='val'" in text
    assert "n=188" in text
    assert "heatmaps_3s" in text


# ---------- unit tests for helpers ----------------------------------------


def test_load_split_csv_round_trip(ueyes_root):
    """Directly exercise the CSV parser to confirm it survived the Block
    column's mix of plain integers and Excel-style scientific notation.
    """
    rows = _load_split_csv(ueyes_root / "image_types.csv")
    assert len(rows) == 1980
    splits_seen = {r.upstream_split for r in rows}
    assert splits_seen == {"Train", "Test"}


def test_carve_validation_split_deterministic(ueyes_root):
    """The stratified carve is a pure function of (rows, fraction, seed).
    Same inputs → same outputs regardless of call site.
    """
    rows = _load_split_csv(ueyes_root / "image_types.csv")
    train_rows = [r for r in rows if r.upstream_split == "Train"]

    a_train, a_val = _carve_validation_split(train_rows, val_fraction=0.1, seed=42)
    b_train, b_val = _carve_validation_split(train_rows, val_fraction=0.1, seed=42)
    assert [r.filename for r in a_train] == [r.filename for r in b_train]
    assert [r.filename for r in a_val] == [r.filename for r in b_val]
