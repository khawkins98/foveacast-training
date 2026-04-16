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
    # why: num_workers=0 for the prototype keeps the run deterministic
    # under a fixed seed. With workers>0 each fork randomises its own
    # DataLoader state and prototype numbers drift between runs, which
    # defeats the point of quoting the first-run values as a gate
    # signal. Phase 5 tunes this for throughput.
    "num_workers": 0,
    "seed": 0,
    # Phase 5 safety machinery: all disabled for the prototype. Grad
    # clipping is off so the Phase 4 gate numbers stay frozen — adding
    # clipping changes the gradient updates and shifts the reported
    # metrics slightly (seen in #11 PR review), which would invalidate
    # the quoted "gate closed at these numbers" baseline. Prototype is a
    # frozen sanity check; Phase 5 is where the safety machinery exercises.
    "grad_clip": None,             # None disables
    "lr_plateau_factor": None,     # None disables the scheduler
    "lr_plateau_patience": None,
    "early_stop_patience": None,   # None disables early stopping
    "early_stop_min_delta": 0.0,
    "save_best_checkpoint": False, # prototype is a sanity run; no artefacts needed
}

FULL_CONFIG: dict[str, object] = {
    "n_epochs": 30,
    "batch_size": 8,
    "learning_rate": 1e-6,  # 10× reduction from paper default for fine-tuning
    "n_train": None,         # full train set
    "n_val": None,           # full val set
    "num_workers": 4,
    "seed": None,             # why: no per-run seeding for throughput in Phase 5
    # Phase 5 safety machinery: all on. These defaults are the "safe first
    # run" numbers; #11 tracks tuning them empirically once we have signal
    # from an actual full fine-tune.
    "grad_clip": 1.0,              # clips gradient norm per step
    "lr_plateau_factor": 0.5,      # halve LR on val-CC plateau
    "lr_plateau_patience": 3,      # epochs of no improvement before LR drop
    "early_stop_patience": 5,      # epochs of no improvement before stopping
    "early_stop_min_delta": 1e-4,  # minimum val CC improvement to count as progress
    "save_best_checkpoint": True,  # write best.pt + best.json on every val-CC improvement
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--prototype",
        action="store_true",
        help="Phase 4 sanity run: 100 train / 25 val, 2 epochs, ~minutes on MPS.",
    )
    # why: --full is an explicit opt-in rather than the default. Running
    # FULL_CONFIG is a 4-hour commitment on M4 MPS; forcing a flag avoids
    # the "oh I forgot --prototype" failure mode that turns an intended
    # sanity run into an overnight fine-tune against untuned
    # hyperparameters.
    mode_group.add_argument(
        "--full",
        action="store_true",
        help="Phase 5 full fine-tune: 1,684 train, 30 epochs, hours. Writes best.pt.",
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

    # why: FULL_CONFIG numbers are placeholder until the first real Phase
    # 5 run tunes them empirically. Loud warning rather than silent default
    # so a contributor who opts into --full before that tuning happens
    # knows what they're signing up for.
    if args.full:
        print(
            "⚠ FULL_CONFIG hyperparameters are placeholder until Phase 5. "
            "Safety machinery (grad clip, LR scheduler, early stop, best-"
            "checkpoint saving) IS in place, but the learning rate, epoch "
            "count, and batch size have not been tuned empirically yet. "
            "Your first run is effectively a hyperparameter probe."
        )

    # why: optional fixed-seed for reproducibility. Only the prototype
    # config sets this; the full run leaves seed=None for throughput.
    if config.get("seed") is not None:
        import random

        import numpy as np
        seed = int(config["seed"])
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f"→ seed:   {seed} (deterministic mode)")

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

    # why: ReduceLROnPlateau on validation CC. mode='max' because higher
    # CC is better. Disabled by passing factor=None; in that case we
    # keep `scheduler = None` and skip the .step() call at val time.
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
    if config["lr_plateau_factor"] is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config["lr_plateau_factor"],
            patience=config["lr_plateau_patience"],
        )

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
        "lr_per_epoch": [],
        "best_val_cc": None,
        "best_epoch": None,
        "stopped_early_at_epoch": None,
    }

    # Best-checkpoint + early-stopping tracking.
    best_val_cc = float("-inf")
    epochs_since_improvement = 0

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
            # why: gradient clipping protects against a single-batch loss
            # spike silently corrupting the best checkpoint in a long run.
            # Disabled in the prototype because 50 steps is not enough
            # to benefit.
            if config["grad_clip"] is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=config["grad_clip"],
                )
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
        current_lr = optimizer.param_groups[0]["lr"]
        history["val_loss_per_epoch"].append(val_loss)
        history["val_cc_per_epoch"].append(val_cc)
        history["lr_per_epoch"].append(current_lr)

        # Best-checkpoint saving.
        improved = val_cc > best_val_cc + config["early_stop_min_delta"]
        if improved:
            best_val_cc = val_cc
            epochs_since_improvement = 0
            history["best_val_cc"] = best_val_cc
            history["best_epoch"] = epoch
            if config["save_best_checkpoint"]:
                torch.save(model.state_dict(), out_dir / "best.pt")
                (out_dir / "best.json").write_text(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "val_cc": val_cc,
                            "val_loss": val_loss,
                            "train_loss": train_loss,
                            "learning_rate": current_lr,
                        },
                        indent=2,
                    )
                )
        else:
            epochs_since_improvement += 1

        # LR scheduler step (after val, on CC).
        if scheduler is not None:
            scheduler.step(val_cc)

        improvement_marker = " ★ new best" if improved else ""
        print(
            f"epoch {epoch}/{config['n_epochs']}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_cc={val_cc:.4f}  "
            f"lr={current_lr:.2e}{improvement_marker}"
        )
        print("──" * 30)

        # Early stopping.
        if (
            config["early_stop_patience"] is not None
            and epochs_since_improvement >= config["early_stop_patience"]
        ):
            print(
                f"✓ early stop: val CC hasn't improved by "
                f"{config['early_stop_min_delta']} in "
                f"{config['early_stop_patience']} epochs."
            )
            history["stopped_early_at_epoch"] = epoch
            break

    # Always save a final snapshot of weights too — useful for debugging
    # if the best.pt checkpoint looks wrong.
    if config["save_best_checkpoint"]:
        torch.save(model.state_dict(), out_dir / "final.pt")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"✓ wrote {out_dir / 'history.json'}")
    if history["best_epoch"] is not None:
        print(
            f"  best val_cc={history['best_val_cc']:.4f} at epoch "
            f"{history['best_epoch']}  →  {out_dir / 'best.pt'}"
        )

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
