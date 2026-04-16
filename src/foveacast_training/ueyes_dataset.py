"""UEyes dataset loader for PyTorch fine-tuning of MSI-Net.

Wraps the UEyes deposit (Jiang et al. 2023, CC BY 4.0, Zenodo record 8010312)
as a `torch.utils.data.Dataset`. Returns `(image, saliency)` tensor pairs
shaped to MSINet's input contract — RGB [0, 255] float32 NCHW for the image,
[0, 1] float32 single-channel for the saliency ground truth.

See `data/README.md` for the deposit's actual directory layout (as observed
on 2026-04-16 after unpacking) and for the choices deferred to this module.
The upstream README inside the archive refers to a file called `info.csv`
that ships as `image_types.csv`; this loader keys on the latter.

Attribution:
    Jiang, Y., Leiva, L. A., Rezazadegan Tavakoli, H., Houssel, P. R. B.,
    Kylmälä, J., & Oulasvirta, A. (2023). UEyes: Understanding Visual
    Saliency across User Interface Types. In CHI '23, Article 285.
    https://doi.org/10.1145/3544548.3581096

Design decisions landed here (documented in LEARNINGS.md 2026-04-16 Phase 3):
    * Saliency target variant: `heatmaps_3s` — Gaussian-smoothed continuous
      maps at a 3-second aggregate viewing window. Closest to MSI-Net's
      SALICON prior (~5s aggregate). `fixmaps_*` are binary; `overlay_*`
      are for visualisation only. Configurable via the `saliency_variant`
      constructor argument so Phase 6 can compare.
    * Validation split: stratified by category, 10% of the upstream train
      set (~187 images, 47 per category), fixed seed. The upstream deposit
      provides only a train/test split; val is our construction.
    * Preprocessing: aspect-ratio-preserving resize via PIL BICUBIC, then
      symmetric pad to `(240, 320)`. Padding value 126 for stimulus images
      (mid-grey, Kroner's convention), 0 for saliency maps.
    * Input size: `(240, 320)` — SALICON-native, matching the HF pretrained
      weights. Configurable.
    * No data augmentation. 1,980 images + a close pretrained prior means
      augmentation adds noise without a training-size payoff.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# why: hoisted so tests can reference them and so callers see the exact
# values Kroner's reference used for padding (stimuli at mid-grey, maps
# at black).
STIMULUS_PAD_VALUE: float = 126.0
SALIENCY_PAD_VALUE: float = 0.0

# why: default target size matches SALICON — the HF pretrained MSI-Net was
# trained at (240, 320), so fine-tuning at the same size keeps weights
# close to their pretrained prior. See LEARNINGS.md 2026-04-16 Phase 1
# for the scaled-back-later rationale.
DEFAULT_INPUT_SIZE: tuple[int, int] = (240, 320)

# why: 3s aggregates are the closest analogue to SALICON's ~5s ground truth,
# which is what MSI-Net's pretrained weights were fit against.
DEFAULT_SALIENCY_VARIANT: str = "heatmaps_3s"

# why: one-in-ten of train carved to val is the conventional fine-tune
# split; stratifying by category keeps all four UI types proportional.
# Fixed seed for reproducibility across runs.
DEFAULT_VAL_FRACTION: float = 0.1
DEFAULT_VAL_SEED: int = 42

# Upstream deposit subfolder names. why: captured as constants so a future
# re-deposit with renamed folders is a one-line fix here rather than spread
# across the file.
_IMAGES_SUBDIR = "images"
_SALIENCY_ROOT = "saliency_maps"
_SPLIT_CSV_NAME = "image_types.csv"

# The four categories in the dataset, per the paper and confirmed by the
# split CSV during Phase 0. Captured so stratified splitting can enumerate
# them without re-scanning the CSV.
_CATEGORIES = ("desktop", "mobile", "poster", "web")


@dataclass(frozen=True)
class _SplitRow:
    """A single row of `image_types.csv`, already normalised and split-tagged.

    The CSV ships with CRLF line endings and a single header row; fields are
    `Image Name`, `Category`, `Block`, `Train/Test`. `_load_split_csv`
    normalises CRLF and returns these dataclasses directly.

    `block` is kept as a raw string. why: some rows in the upstream CSV have
    Excel-formatted numbers like "0,00E+00" alongside plain integers like "0"
    — likely because the file was opened in Excel at some point. The loader
    doesn't use `block` for anything, so preserving the original string
    avoids a parse dance for no gain.
    """
    filename: str
    category: str
    block: str
    upstream_split: Literal["Train", "Test"]


def _load_split_csv(csv_path: Path) -> list[_SplitRow]:
    """Parse `image_types.csv`, handling CRLF line endings and the one-line
    header. Raises if the file is missing, any row has the wrong column
    count, or the split label isn't `Train` / `Test`.
    """
    rows: list[_SplitRow] = []
    # why: newline='' tells Python's csv module to handle line endings itself
    # rather than the file-object layer mangling CRLF. Matches the upstream's
    # Windows-style line endings.
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter=";")
        header = next(reader)
        expected_header = ["Image Name", "Category", "Block", "Train/Test"]
        if header != expected_header:
            raise ValueError(
                f"{csv_path} has unexpected header {header!r}, "
                f"expected {expected_header!r}"
            )
        for lineno, record in enumerate(reader, start=2):
            if len(record) != 4:
                raise ValueError(
                    f"{csv_path}:{lineno} has {len(record)} fields, expected 4"
                )
            filename, category, block_str, split_label = (s.strip() for s in record)
            if split_label not in ("Train", "Test"):
                raise ValueError(
                    f"{csv_path}:{lineno} has split label {split_label!r}, "
                    "expected 'Train' or 'Test'"
                )
            rows.append(
                _SplitRow(
                    filename=filename,
                    category=category,
                    block=block_str,
                    upstream_split=split_label,  # type: ignore[arg-type]
                )
            )
    return rows


def _carve_validation_split(
    train_rows: list[_SplitRow],
    val_fraction: float,
    seed: int,
) -> tuple[list[_SplitRow], list[_SplitRow]]:
    """Split upstream-Train rows into (train-only, val) stratified by category.

    A fresh `numpy.random.default_rng(seed)` is used rather than the global
    NumPy random state to keep the val set reproducible across runs without
    leaking seed state to surrounding code.
    """
    rng = np.random.default_rng(seed)
    train_only: list[_SplitRow] = []
    val: list[_SplitRow] = []

    by_category: dict[str, list[_SplitRow]] = {cat: [] for cat in _CATEGORIES}
    for row in train_rows:
        if row.category not in by_category:
            # Unknown category — the dataset should be validated against
            # _CATEGORIES before reaching this helper.
            raise ValueError(f"unknown category {row.category!r} in row for {row.filename}")
        by_category[row.category].append(row)

    for cat in _CATEGORIES:
        # why: deterministic order in — rng.permutation on a stable-sorted
        # per-category list gives a reproducible shuffle regardless of the
        # CSV's row order.
        cat_rows = sorted(by_category[cat], key=lambda r: r.filename)
        if not cat_rows:
            continue
        perm = rng.permutation(len(cat_rows))
        n_val = max(1, int(round(len(cat_rows) * val_fraction)))
        val_indices = set(perm[:n_val].tolist())
        for i, row in enumerate(cat_rows):
            (val if i in val_indices else train_only).append(row)

    return train_only, val


def _resize_with_aspect(
    arr: np.ndarray,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """Resize an H×W or H×W×3 uint8 array to fit within (target_h, target_w),
    preserving aspect ratio. Returns a new uint8 array at the scaled dimensions
    (not yet padded). BICUBIC for both up- and down-scale.

    why: picking a single interpolation mode rather than Kroner's area/bicubic
    split keeps the loader simple. For fine-tuning we don't need bit-exact
    preprocessing parity with SALICON — only consistency between train and
    eval within this repo. Bit-exact parity for Phase 6 comparison tests is
    handled by a dedicated eval preprocessor.
    """
    src_h, src_w = arr.shape[:2]
    scale = min(target_h / src_h, target_w / src_w)
    new_h = max(1, int(round(src_h * scale)))
    new_w = max(1, int(round(src_w * scale)))

    mode = "RGB" if arr.ndim == 3 else "L"
    img = Image.fromarray(arr, mode=mode)
    img = img.resize((new_w, new_h), resample=Image.BICUBIC)
    return np.array(img, dtype=np.uint8)


def _pad_to(
    arr: np.ndarray,
    target_h: int,
    target_w: int,
    pad_value: float,
) -> np.ndarray:
    """Symmetric-ish constant-value padding of an H×W or H×W×3 array to
    `(target_h, target_w)`. Matches Kroner's _pad_image: extra pixel on
    the bottom/right when padding is odd.
    """
    src_h, src_w = arr.shape[:2]
    if src_h > target_h or src_w > target_w:
        raise ValueError(
            f"cannot pad to ({target_h}, {target_w}): input is ({src_h}, {src_w})"
        )
    pad_top = (target_h - src_h) // 2
    pad_bot = target_h - src_h - pad_top
    pad_lft = (target_w - src_w) // 2
    pad_rgt = target_w - src_w - pad_lft

    if arr.ndim == 3:
        pads = ((pad_top, pad_bot), (pad_lft, pad_rgt), (0, 0))
    else:
        pads = ((pad_top, pad_bot), (pad_lft, pad_rgt))
    # why: np.pad's constant-values kwarg accepts a scalar; we cast to the
    # same dtype as the input to avoid a quiet float→uint8 truncation later.
    return np.pad(arr, pads, mode="constant", constant_values=arr.dtype.type(pad_value))


def _preprocess_stimulus(path: Path, target_size: tuple[int, int]) -> torch.Tensor:
    """Load `path` as RGB, resize-and-pad to `target_size`, return a
    (3, H, W) float32 tensor in [0, 255].
    """
    with Image.open(path) as im:
        # why: unconditional RGB convert handles the handful of mode='L'
        # images in the deposit without a shape branch at the tensor level.
        arr = np.array(im.convert("RGB"), dtype=np.uint8)

    resized = _resize_with_aspect(arr, target_size[0], target_size[1])
    padded = _pad_to(resized, target_size[0], target_size[1], STIMULUS_PAD_VALUE)
    # HWC → CHW, cast to float32. Values stay in [0, 255] — MSINet's input
    # contract handles mean subtraction internally.
    chw = np.transpose(padded, (2, 0, 1))
    return torch.from_numpy(chw.astype(np.float32))


def _preprocess_saliency(path: Path, target_size: tuple[int, int]) -> torch.Tensor:
    """Load `path` as a single-channel saliency map, resize-and-pad to
    `target_size`, return a (1, H, W) float32 tensor in [0, 1].
    """
    with Image.open(path) as im:
        # why: saliency maps in the UEyes deposit are stored as 'L' mode
        # PNGs/JPGs. Unconditional 'L' convert handles any stray mode
        # (e.g. re-exported as RGB by some tool) by averaging channels.
        arr = np.array(im.convert("L"), dtype=np.uint8)

    resized = _resize_with_aspect(arr, target_size[0], target_size[1])
    padded = _pad_to(resized, target_size[0], target_size[1], SALIENCY_PAD_VALUE)
    # Add channel dim, cast to float32, divide by 255 to normalise to [0, 1].
    # why: [0, 1] matches MSINet's output range (which is min-max-normalised
    # per image); the training loop's loss can re-normalise to sum-to-1 if
    # KL divergence is used.
    chw = padded[np.newaxis, :, :]
    return torch.from_numpy(chw.astype(np.float32) / 255.0)


class UEyesDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset over UEyes for MSI-Net fine-tuning.

    Parameters
    ----------
    root : Path | str
        Path to the unpacked `UEyes_dataset/` directory (the child of
        `data/ueyes/` once `data/fetch.sh` has run).
    split : {"train", "val", "test"}
        Which partition to iterate. `test` is the upstream Test rows; `train`
        and `val` are carved from the upstream Train rows by category-stratified
        random hold-out.
    saliency_variant : str
        Subfolder under `saliency_maps/` to read ground truth from. Defaults
        to `heatmaps_3s`. Other valid values on the deposit: `fixmaps_{1,3,7}s`,
        `heatmaps_{1,3,7}s`. `overlay_heatmaps_*` is supported but note that
        its filenames are prefixed with `overlay_` — that's handled here.
    input_size : (h, w)
        Target `(H, W)` after resize+pad. Defaults to `(240, 320)` matching
        the SALICON-pretrained MSI-Net.
    val_fraction : float
        Fraction of upstream-Train rows to hold out as val. Ignored when
        `split == "test"`.
    val_seed : int
        Seed for the stratified hold-out RNG. Ignored when `split == "test"`.
    """

    def __init__(
        self,
        root: Path | str,
        split: Literal["train", "val", "test"],
        *,
        saliency_variant: str = DEFAULT_SALIENCY_VARIANT,
        input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
        val_fraction: float = DEFAULT_VAL_FRACTION,
        val_seed: int = DEFAULT_VAL_SEED,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.saliency_variant = saliency_variant
        self.input_size = input_size

        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train|val|test, got {split!r}")

        csv_path = self.root / _SPLIT_CSV_NAME
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"{csv_path} not found — is `data/fetch.sh` unpacked at {self.root}?"
            )
        all_rows = _load_split_csv(csv_path)

        upstream_train = [r for r in all_rows if r.upstream_split == "Train"]
        upstream_test = [r for r in all_rows if r.upstream_split == "Test"]

        if split == "test":
            self._rows = upstream_test
        else:
            train_only, val = _carve_validation_split(upstream_train, val_fraction, val_seed)
            self._rows = train_only if split == "train" else val

        # Cache the saliency directory path and the overlay-prefix flag once.
        # why: overlay_heatmaps_* subdirs prefix every filename with
        # `overlay_` — the only naming irregularity in the deposit. Rather
        # than check per-getitem, resolve once up front.
        self._saliency_dir = self.root / _SALIENCY_ROOT / saliency_variant
        self._saliency_prefix = "overlay_" if saliency_variant.startswith("overlay_") else ""

        if not self._saliency_dir.is_dir():
            raise FileNotFoundError(
                f"{self._saliency_dir} not found — is saliency_variant "
                f"{saliency_variant!r} correct?"
            )
        self._images_dir = self.root / _IMAGES_SUBDIR
        if not self._images_dir.is_dir():
            raise FileNotFoundError(f"{self._images_dir} not found")

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self._rows[idx]
        image_path = self._images_dir / row.filename
        saliency_path = self._saliency_dir / f"{self._saliency_prefix}{row.filename}"

        image = _preprocess_stimulus(image_path, self.input_size)
        saliency = _preprocess_saliency(saliency_path, self.input_size)
        return image, saliency

    def filenames(self) -> list[str]:
        """Return the filenames in the current split, preserving iteration
        order. Useful for correlating outputs back to source images in
        evaluation / benchmarking.
        """
        return [r.filename for r in self._rows]

    def categories(self) -> list[str]:
        """Return the per-sample category labels ('desktop' | 'mobile' |
        'poster' | 'web') in the same order as `filenames()`.
        """
        return [r.category for r in self._rows]
