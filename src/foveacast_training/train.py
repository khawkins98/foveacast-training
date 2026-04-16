"""Fine-tune MSI-Net on UEyes.

Two modes:

- `--prototype` (Phase 4 gate): 100 train / 25 val images, 2 epochs. Fast
  sanity check for "does the loss curve look plausible" — should run in
  a few minutes on M4 MPS and show train loss decreasing monotonically
  epoch-over-epoch. If it NaNs, diverges, or hangs, something structural
  is wrong and Phase 5 is not safe to start.
- Default (Phase 5): full 1,684-image train set, 30 epochs, lower learning
  rate, best-checkpoint saving. (Phase 5 will tune this; current defaults
  are placeholder — the prototype mode is the Phase 4 deliverable.)

Both modes share the same loop and the same device auto-detection so
device-specific bugs surface in the prototype rather than midway through
a 4-hour full run.

Run:
    .venv/bin/python -m foveacast_training.train --prototype
    .venv/bin/python -m foveacast_training.train --out-dir runs/my-run

Writes `history.json` into `runs/{prototype|full}-{timestamp}/` with the
full per-step training loss and per-epoch validation loss + CC. The output
directory is gitignored per the repo's `.gitignore`.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from foveacast_training.losses import correlation_coefficient, kl_divergence_saliency
from foveacast_training.msinet import MSINet
from foveacast_training.ueyes_dataset import UEyesDataset


def _auto_device() -> torch.device:
    """MPS → CUDA → CPU, matching the convention in CLAUDE.md."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# why: prototype and full configs live next to each other so the delta
# between them is reviewable at a glance. Phase 5 will revisit the full
# config's numbers; for Phase 4 the prototype config is the load-bearing
# thing.
PROTOTYPE_CONFIG: dict[str, object] = {
    "n_epochs": 2,
    "batch_size": 4,
    "learning_rate": 1e-5,  # Kroner's SALICON default; no reduction for prototype
    "n_train": 100,
    "n_val": 25,
    "num_workers": 2,
}

FULL_CONFIG: dict[str, object] = {
    "n_epochs": 30,
    "batch_size": 8,
    "learning_rate": 1e-6,  # 10× reduction from paper default for fine-tuning
    "n_train": None,         # full train set
    "n_val": None,           # full val set
    "num_workers": 4,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--prototype",
        action="store_true",
        help="Phase 4 sanity run: 100 train / 25 val, 2 epochs, ~minutes on MPS.",
    )
    parser.add_argument(
        "--data-root",
        default="data/ueyes/UEyes_dataset",
        help="Path to the unpacked UEyes directory.",
    )
    parser.add_argument(
        "--weights",
        default="weights/msinet_salicon.pt",
        help="PyTorch state_dict produced by scripts/import_msinet_weights.py.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output dir for history.json (default: runs/{mode}-{timestamp}).",
    )
    args = parser.parse_args()

    config = dict(PROTOTYPE_CONFIG if args.prototype else FULL_CONFIG)
    mode_label = "prototype" if args.prototype else "full"

    device = _auto_device()
    print(f"→ device: {device}")
    print(f"→ mode:   {mode_label}")
    print(f"→ config: {config}")

    # Load model + weights.
    model = MSINet()
    state_dict = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)

    # Data. why: Subset preserves the same __getitem__ semantics as
    # UEyesDataset, so the training loop doesn't need a code branch on
    # whether it's in prototype mode.
    train_ds: Subset | UEyesDataset = UEyesDataset(args.data_root, split="train")
    val_ds: Subset | UEyesDataset = UEyesDataset(args.data_root, split="val")
    if config["n_train"] is not None:
        train_ds = Subset(train_ds, range(min(config["n_train"], len(train_ds))))
    if config["n_val"] is not None:
        val_ds = Subset(val_ds, range(min(config["n_val"], len(val_ds))))

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )
    print(f"→ data:   {len(train_ds)} train / {len(val_ds)} val")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Output dir + history record.
    if args.out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = Path("runs") / f"{mode_label}-{timestamp}"
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history: dict[str, object] = {
        "mode": mode_label,
        "config": config,
        "device": str(device),
        "train_loss_per_step": [],
        "val_loss_per_epoch": [],
        "val_cc_per_epoch": [],
    }

    print(f"→ output: {out_dir}")
    print("──" * 30)

    for epoch in range(1, config["n_epochs"] + 1):
        model.train()
        epoch_train_losses: list[float] = []
        for step, (images, saliency) in enumerate(train_loader, start=1):
            images = images.to(device)
            saliency = saliency.to(device)

            pred = model(images)
            loss = kl_divergence_saliency(pred, saliency)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            epoch_train_losses.append(loss_val)
            history["train_loss_per_step"].append(loss_val)

            if step == 1 or step % 10 == 0 or step == len(train_loader):
                print(
                    f"  epoch {epoch}/{config['n_epochs']}  "
                    f"step {step:>4d}/{len(train_loader)}  "
                    f"loss={loss_val:.6f}"
                )

        # Validation
        model.eval()
        val_losses: list[float] = []
        val_ccs: list[float] = []
        with torch.no_grad():
            for images, saliency in val_loader:
                images = images.to(device)
                saliency = saliency.to(device)
                pred = model(images)
                val_losses.append(kl_divergence_saliency(pred, saliency).item())
                val_ccs.append(correlation_coefficient(pred, saliency).item())

        train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        val_cc = sum(val_ccs) / len(val_ccs)
        history["val_loss_per_epoch"].append(val_loss)
        history["val_cc_per_epoch"].append(val_cc)
        print(
            f"epoch {epoch}/{config['n_epochs']}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_cc={val_cc:.4f}"
        )
        print("──" * 30)

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"✓ wrote {out_dir / 'history.json'}")

    # Sanity signal for the Phase 4 gate. Does NOT fail the run; surfaces
    # the answer in stdout so a human (or CI) can act on it.
    if len(history["val_cc_per_epoch"]) >= 2:
        cc_first, cc_last = history["val_cc_per_epoch"][0], history["val_cc_per_epoch"][-1]
        loss_first, loss_last = (
            history["val_loss_per_epoch"][0],
            history["val_loss_per_epoch"][-1],
        )
        direction = "↑" if cc_last > cc_first else "↓" if cc_last < cc_first else "="
        loss_dir = "↓" if loss_last < loss_first else "↑" if loss_last > loss_first else "="
        print(f"  val_cc   {cc_first:.4f} → {cc_last:.4f}  {direction}")
        print(f"  val_loss {loss_first:.4f} → {loss_last:.4f}  {loss_dir}")


if __name__ == "__main__":
    main()
