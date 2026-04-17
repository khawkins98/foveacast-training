"""Render saliency-map overlays for qualitative comparison (Phase 7).

Takes a source screenshot + a PyTorch checkpoint, runs the image through
MSINet, and produces a heatmap-on-screenshot overlay PNG. The overlay
uses the 'inferno' colormap at 50% alpha — chosen to match the visual
register of the UEyes ground-truth heatmaps in the Foveacast comparison
set.

Usage:
    # Single image:
    .venv/bin/python scripts/render_saliency.py \\
        --source benchmark/screenshots/acs-welcome-source.png \\
        --checkpoint runs/full-*/best.pt \\
        --out benchmark/screenshots/acs-welcome-v3-finetuned.png

    # Batch mode (multiple source images):
    .venv/bin/python scripts/render_saliency.py \\
        --source img1.png img2.png img3.png \\
        --checkpoint runs/full-*/best.pt \\
        --out-dir benchmark/screenshots/v3-renders/

For Phase 7's qualitative comparison, this script is called once per
checkpoint (stock vs fine-tuned) on the same set of source images,
then the outputs are reviewed visually alongside the ground-truth
heatmaps from UEyes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image

from foveacast_training.msinet import MSINet


def load_model(checkpoint_path: Path) -> MSINet:
    """Load MSINet on CPU in eval mode."""
    model = MSINet()
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    return model.eval()


def predict_saliency(model: MSINet, source: Image.Image) -> np.ndarray:
    """Run a PIL image through MSINet and return a saliency heatmap
    resized back to the source image's original dimensions.

    Returns a float32 array in [0, 1] at (H_orig, W_orig).
    """
    orig_w, orig_h = source.size

    # why: preprocessing matches UEyesDataset's _preprocess_stimulus
    # (aspect-preserving resize + constant-126 pad to 240×320, RGB
    # float32 in [0, 255], NCHW). Inlined here to avoid a dependency
    # on the Dataset module for a standalone rendering script.
    target_h, target_w = 240, 320
    img = source.convert("RGB")
    src_w, src_h = img.size
    scale = min(target_h / src_h, target_w / src_w)
    new_h = max(1, int(round(src_h * scale)))
    new_w = max(1, int(round(src_w * scale)))
    img = img.resize((new_w, new_h), resample=Image.BICUBIC)
    arr = np.array(img, dtype=np.uint8)

    # Pad to target
    pad_top = (target_h - new_h) // 2
    pad_bot = target_h - new_h - pad_top
    pad_lft = (target_w - new_w) // 2
    pad_rgt = target_w - new_w - pad_lft
    arr = np.pad(arr, ((pad_top, pad_bot), (pad_lft, pad_rgt), (0, 0)),
                 mode="constant", constant_values=126)

    # HWC → NCHW float32
    x = torch.from_numpy(arr.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        sal = model(x)  # (1, 1, 240, 320) in [0, 1]

    sal_np = sal.squeeze().numpy()  # (240, 320)

    # Unpad: crop back to the pre-pad region
    sal_cropped = sal_np[pad_top:pad_top + new_h, pad_lft:pad_lft + new_w]

    # Resize saliency back to original image dimensions
    sal_pil = Image.fromarray((sal_cropped * 255).astype(np.uint8), mode="L")
    sal_pil = sal_pil.resize((orig_w, orig_h), resample=Image.BICUBIC)
    return np.array(sal_pil, dtype=np.float32) / 255.0


def overlay_heatmap(
    source: Image.Image,
    saliency: np.ndarray,
    colormap: str = "inferno",
    alpha: float = 0.5,
) -> Image.Image:
    """Composite a saliency heatmap onto a source screenshot.

    Returns an RGBA PIL image at the source's original dimensions.
    """
    # Colorize saliency via matplotlib colormap
    cmap = cm._colormaps.get_cmap(colormap)
    heatmap_rgba = cmap(saliency)  # (H, W, 4) float in [0, 1]
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_rgb)

    # Blend
    source_rgb = source.convert("RGB").resize(
        (saliency.shape[1], saliency.shape[0]), resample=Image.BICUBIC
    )
    blended = Image.blend(source_rgb, heatmap_pil, alpha=alpha)
    return blended


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source", type=Path, nargs="+", required=True,
        help="Source screenshot(s) to render saliency on.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="PyTorch state_dict (.pt).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output path (single-image mode). Mutually exclusive with --out-dir.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory (batch mode). Files named {stem}-saliency.png.",
    )
    parser.add_argument(
        "--colormap", default="inferno",
        help="Matplotlib colormap for the heatmap overlay.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Heatmap overlay opacity (0=invisible, 1=opaque).",
    )
    args = parser.parse_args()

    if len(args.source) == 1 and args.out:
        out_paths = [args.out]
    elif args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_paths = [args.out_dir / f"{s.stem}-saliency.png" for s in args.source]
    else:
        parser.error("use --out for single image or --out-dir for batch")
        return

    print(f"→ loading {args.checkpoint}")
    model = load_model(args.checkpoint)

    for src_path, out_path in zip(args.source, out_paths, strict=True):
        print(f"→ {src_path.name}", end="")
        source = Image.open(src_path)
        saliency = predict_saliency(model, source)
        overlay = overlay_heatmap(source, saliency, args.colormap, args.alpha)
        overlay.save(out_path)
        print(f" → {out_path}")

    print(f"✓ rendered {len(args.source)} image(s)")


if __name__ == "__main__":
    main()
