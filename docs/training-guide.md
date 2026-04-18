# Training Guide

How to reproduce the shipped model, how to run your own experiments, and how to compare weight sets against the baseline. Written so that someone coming back to this repo in six months can run a new fine-tune without re-reading the full LEARNINGS arc.

For the *what* and *why* behind the architecture, see [`ARCHITECTURE.md`](../ARCHITECTURE.md). For the dated decision log, see [`LEARNINGS.md`](../LEARNINGS.md). This document is the *how to operate* companion.

---

## The shipped configuration (v3-baseline)

The model that ships in Foveacast V3 was produced by the run documented below. If you're reproducing rather than experimenting, match these settings exactly.

### Training run

| parameter | value | why |
|---|---|---|
| checkpoint init | `weights/msinet_salicon.pt` (stock SALICON-pretrained MSI-Net, imported from HuggingFace) | pretrained prior is load-bearing; fine-tuning from random init on 1,980 images would underfit |
| saliency target | `heatmaps_3s` | continuous Gaussian-smoothed maps; 3s aggregate window is closest to SALICON's ~5s aggregate that the pretrained weights were fit against |
| input size | (240, 320) | SALICON-native; matches the pretrained weights' resolution |
| learning rate | 1e-6 | 10× reduction from Kroner's SALICON default (1e-5); standard fine-tuning practice |
| optimizer | Adam | matches Kroner's reference |
| batch size | 8 | fits M4 MPS comfortably |
| gradient clipping | norm 1.0 | prevents single-batch loss spikes from corrupting the best checkpoint |
| LR scheduler | ReduceLROnPlateau(mode='max', factor=0.5, patience=3) on val CC | halves LR when val CC plateaus for 3 epochs |
| early stopping | patience=5, min_delta=1e-4 on val CC | stops training when CC improvement stalls |
| epochs | 30 (ran to completion, early-stop triggered at epoch 30) | |
| val split | 10% of upstream Train, stratified by category, seed=42 | 1,684 train / 188 val / 108 test |
| data augmentation | none | 1,980 images + close pretrained prior; augmentation adds noise |
| device | M4 MacBook Air (MPS) | ~3.5 hours wall-clock |

### Results

**Best checkpoint: epoch 25, val_cc=0.7247.** Saved to `runs/full-20260416-225926/best.pt`.

Evaluation on held-out test split (108 images, never seen during training):

| metric | fine-tuned | stock SALICON | delta |
|---|---|---|---|
| CC (higher better) | **0.7068** | 0.4934 | +43% |
| KLD (lower better) | **0.6574** | 1.1682 | -44% |
| NSS (higher better) | **2.2879** | 1.5776 | +45% |

Release artefact: `releases/foveacast-v3.onnx` (106.4 MB, opset 17, PyTorch ↔ onnxruntime CPU parity at max abs err 6.14e-06).

### Selection rationale

This was the first full fine-tune run. Fine-tuned beats stock on all three standard saliency metrics by 43-45% each. The improvement is consistent across metrics and across the standard-deviation bands. The UEyes paper (Jiang et al. 2023) reported +10 AUC points fine-tuning on UEyes; our CC improvement of +0.21 absolute is in the same ballpark on a different architecture. No reason to hold the release for further hyperparameter search — the baseline is strong.

If a future run produces materially better numbers, it becomes the new shipped config and this section gets updated.

---

## Knobs to turn

These are the parameters worth varying in future experiments, in rough order of expected impact.

### 1. Saliency target variant

**Current:** `heatmaps_3s` (the v0.1.0 shipped default)

**Options:** `heatmaps_1s`, `heatmaps_3s`, `heatmaps_7s`, `fixmaps_1s`, `fixmaps_3s`, `fixmaps_7s`

**What changes:** the ground-truth map the model learns to predict. 1s = first-glance attention (useful for "what grabs the eye first"), 3s = early exploration, 7s = thorough viewing. `fixmaps_*` are binary (fixation points only) vs continuous (`heatmaps_*`). The choice affects what the model optimises for.

**How to try:**

```sh
# First-class CLI flag. Thread through to both train and val datasets
# and includes the variant in the run-dir name for self-identification.
.venv/bin/python -m foveacast_training.train --full --saliency-variant heatmaps_1s

# Eval must be run at the matched time window, otherwise CC/KLD/NSS are
# comparing apples to oranges:
.venv/bin/python -m foveacast_training.eval \
    --checkpoint runs/full-heatmaps_1s-*/best.pt \
    --compare weights/msinet_salicon.pt \
    --time-window 1s --split test
```

**Expected effect:** `heatmaps_1s` biases toward first-fixation targets (headlines, hero images, primary CTAs); `heatmaps_7s` spreads attention more broadly across secondary content. The right choice depends on what Foveacast users care about — "where does the eye land first" vs "what gets looked at overall." Issue #19 ships all three as user-selectable.

**Note on hyperparameters:** `FULL_CONFIG` was tuned against `heatmaps_3s` for the v0.1.0 baseline. A new-variant run is effectively a hyperparameter probe — the LR schedule, epoch count, and batch size haven't been tested for 1s/7s targets. If the first run plateaus early or diverges, LR is the first knob to try.

### 2. Learning rate

**Current:** 1e-6

**Options:** 1e-7 to 1e-5

**What changes:** how aggressively the fine-tune shifts from the pretrained prior. Lower LR = more conservative, stays closer to stock MSI-Net, less risk of overfitting but slower convergence. Higher LR = faster adaptation, higher risk of overshooting and losing the prior.

**How to try:** edit `FULL_CONFIG["learning_rate"]` in `src/foveacast_training/train.py`.

**Expected effect:** the v3-baseline run at 1e-6 showed steady improvement for 25 epochs before plateauing. A higher LR (e.g. 5e-6) might peak earlier and higher, or it might overshoot — worth a 10-epoch probe before committing to a 30-epoch run.

### 3. Input resolution

**Current:** (240, 320) — SALICON-native

**Options:** (120, 160) for V1-matching browser perf, (360, 360) for higher spatial resolution

**What changes:** both the fine-tune resolution and the shipped `.onnx` input size. Higher resolution captures smaller UI elements (tiny buttons, fine text) but costs more inference time in the browser. Lower resolution is cheaper but the pretrained weights were trained at (240, 320) — changing resolution means the prior is less close.

**How to try:** edit `DEFAULT_INPUT_SIZE` in `src/foveacast_training/ueyes_dataset.py` and re-run. The export step and Foveacast's loader.js also need to match.

**Expected effect:** (120, 160) would roughly halve inference time and roughly halve accuracy on small UI targets. (360, 360) would increase compute ~2× but might pick up small targets better. The v3-baseline at (240, 320) is a good middle ground — change this only if browser perf or small-target accuracy becomes a real production concern.

### 4. Epochs + early stopping patience

**Current:** 30 epochs, patience=5

**Expected effect:** more epochs gives the model longer to converge but risks overfitting on 1,684 training images. The LR scheduler + early stopping make this self-limiting — the model stops when it stops improving. Increasing `n_epochs` to 50 or 100 with the current patience=5 is safe; the model will just early-stop at the same point unless the LR schedule produces a late-stage recovery.

### 5. Validation split

**Current:** 10% stratified hold-out from upstream Train, seed=42

**How to try:** change `val_fraction` and `val_seed` on the `UEyesDataset` constructor. Different seeds produce different val sets; different fractions change the train/val balance.

**Expected effect:** a larger val set (15-20%) gives more stable val CC estimates but reduces the training set. A different seed reshuffles which images are in val — useful for checking whether the v3-baseline's numbers are seed-dependent (they shouldn't be, but it's worth one check).

---

## Running a custom experiment

### 1. Edit the config

For now, configs live as Python dicts in `src/foveacast_training/train.py` (`FULL_CONFIG`). Edit the values you want to vary. Future work (tracked in the codebase roadmap) will add `--config path/to/experiment.yaml` support.

### 2. Run

```sh
.venv/bin/python -m foveacast_training.train --full
```

Output goes to `runs/full-{variant}-{timestamp}/` with:
- `history.json` — per-step train loss + per-epoch val loss / CC / LR
- `best.pt` — weights at the best val CC epoch
- `best.json` — metadata for the best epoch
- `final.pt` — weights at training end (may differ from best if early-stopped)
- `state.pt` — full resumable snapshot (model + optimizer + scheduler + counters), overwritten each epoch

If a run gets interrupted (sleep event, Ctrl-C, OOM), pick it up at the last completed epoch:

```sh
.venv/bin/python -m foveacast_training.train --resume runs/full-{variant}-{timestamp}/state.pt
```

The resume output continues in the same dir. Mode, config, and saliency variant are read from `state.pt`; `--prototype` / `--full` / `--saliency-variant` cannot be combined with `--resume` and will error. Resume is not bit-exact (DataLoader shuffle RNG is not restored) — the goal is recovering a trained model, not reproducing a specific loss curve.

For long runs on a MacBook, wrap the command in `caffeinate -i -s` to block idle and system sleep, pipe through `tee` so progress lands in both the terminal and a log file, and pass `python -u` to force unbuffered output:

```sh
caffeinate -i -s .venv/bin/python -u -m foveacast_training.train --full \
    --saliency-variant heatmaps_1s 2>&1 | tee runs/1s-run.log
```

The `-u` matters when piping: Python switches stdout from line-buffered to block-buffered once it's connected to a pipe rather than a TTY, so `print` calls accumulate (~4-8 KB) before flushing. Without `-u`, the log file stays empty for minutes at a time and it's hard to tell whether the run is progressing or hung. This is a real gotcha — preserved here because it cost us ~10 min of "is it actually running?" anxiety on the first issue #19 run.

### 3. Evaluate

```sh
# Compare against the v3-baseline stock numbers:
.venv/bin/python -m foveacast_training.eval \
    --checkpoint runs/full-{your-timestamp}/best.pt \
    --compare weights/msinet_salicon.pt \
    --split test \
    --out runs/full-{your-timestamp}/eval-vs-stock.json

# Compare against the v3-baseline fine-tuned numbers:
.venv/bin/python -m foveacast_training.eval \
    --checkpoint runs/full-{your-timestamp}/best.pt \
    --compare runs/full-20260416-225926/best.pt \
    --split test
```

### 4. Render qualitative comparison

```sh
.venv/bin/python scripts/render_saliency.py \
    --source benchmark/screenshots/acs-welcome-source.png \
    --checkpoint runs/full-{your-timestamp}/best.pt \
    --out benchmark/screenshots/acs-welcome-{your-experiment-name}.png
```

### 5. Export if it's a keeper

Three precision levels, pick per your size/quality budget:

```sh
# FP32 — 106 MB, max parity error ~6e-6. Reference only; too large for browser.
.venv/bin/python -m foveacast_training.export_onnx \
    --checkpoint runs/full-{your-timestamp}/best.pt \
    --out releases/foveacast-{name}.onnx \
    --report releases/foveacast-{name}.parity.json

# FP16 — 57 MB, max parity error ~7e-4. v0.1.0 ships this format.
.venv/bin/python -m foveacast_training.export_onnx --fp16 \
    --checkpoint runs/full-{your-timestamp}/best.pt \
    --out releases/foveacast-{name}-fp16.onnx \
    --report releases/foveacast-{name}-fp16.parity.json

# INT8 — ~26 MB, max parity error ~1e-2. Smallest; needs calibration data.
# Ship only if the CC/KLD/NSS delta vs FP16 is acceptable for your use.
.venv/bin/python -m foveacast_training.quantize_int8 \
    --checkpoint runs/full-{your-timestamp}/best.pt \
    --calibration-data data/ueyes/UEyes_dataset \
    --out releases/foveacast-{name}-int8.onnx \
    --report releases/foveacast-{name}-int8.parity.json
```

### 6. Record the results

Add a row to the experiment log below and — if the experiment taught something non-obvious — a dated entry to `LEARNINGS.md`.

---

## Experiment log

Track every full run here so the selection rationale is visible. The shipped config is always the first row.

| name | date | saliency | LR | epochs (best) | val CC | test CC | test KLD | test NSS | notes |
|---|---|---|---|---|---|---|---|---|---|
| **v3-1s** (shipped v0.2.0) | 2026-04-17 | heatmaps_1s | 1e-6 | 21 (16) | 0.6117 | 0.5870 | 1.1221 | 2.3984 | first-glance attention; early-stop @ 21 |
| **v3-3s** (shipped v0.1.0, re-shipped in v0.2.0) | 2026-04-16 | heatmaps_3s | 1e-6 | 30 (25) | 0.7247 | 0.7068 | 0.6574 | 2.2879 | early exploration; v0.1.0 baseline |
| **v3-7s** (shipped v0.2.0) | 2026-04-18 | heatmaps_7s | 1e-6 | 30 (28) | 0.7647 | 0.7523 | 0.4402 | 1.8892 | full-viewing; highest absolute CC |
| stock SALICON @ 1s window | — | — | — | — | — | 0.3670 | 1.7642 | 1.4607 | stock eval'd at 1s ground truth |
| stock SALICON @ 3s window | — | — | — | — | — | 0.4934 | 1.1682 | 1.5776 | stock eval'd at 3s ground truth (v0.1.0 reference) |
| stock SALICON @ 7s window | — | — | — | — | — | 0.6019 | 0.7829 | 1.4794 | stock eval'd at 7s ground truth |

*Add rows as new experiments land. Keep shipped rows bold.*

### Fine-tuned vs stock across all three windows

Relative improvement is largest at the 1s window (+60% CC) and smallest at the 7s window (+25% CC). The pretrained SALICON prior is least useful at 1s — natural-scene saliency models don't know what eyes hit in a UI's first second. KLD improvement is roughly constant (~40%) across all three windows, which is a nice consistency signal about the training pipeline.

| window | CC improvement | KLD improvement | NSS improvement |
|---|---|---|---|
| 1s | +60% | −36% | +64% |
| 3s | +43% | −44% | +45% |
| 7s | +25% | −44% | +28% |

Cross-window note: absolute CC of the 7s fine-tuned model (0.7523) is higher than the 3s (0.7068) and 1s (0.5870). That's not because the 7s model is "better" — the 7s ground truth is smoother (more fixations averaged), so it's an easier prediction task. Each model answers a different product question; pick based on what you want to measure.

## Precision variants and quality

The v0.2.0 release ships each model at two precisions. Both come from the same trained weights; only the post-training quantisation differs.

**FP16 (~57 MB per model)** — half-precision via `onnxconverter_common.convert_float_to_float16`. Max-pixel parity against PyTorch FP32 around 1–2e-3 depending on model. Metric-level parity (CC/KLD/NSS on the test split) matches FP32 PyTorch to 4 decimal places across all three models. Essentially lossless at the level anyone will measure.

**INT8 (~32 MB per model)** — static post-training quantisation via `onnxruntime.quantization.quantize_static`, calibrated on 100 random UEyes training images. More aggressive compression, small but measurable quality regression. Per-model numbers on the UEyes test split (108 images):

| window | metric | PyTorch FP32 | INT8 ONNX | relative delta |
|---|---|---|---|---|
| 1s | CC | 0.5870 | 0.5869 | −0.02% |
| 1s | KLD | 1.1221 | 1.1407 | +1.66% worse |
| 1s | NSS | 2.3984 | 2.3968 | −0.07% |
| 3s | CC | 0.7068 | 0.7062 | −0.08% |
| 3s | KLD | 0.6574 | 0.6745 | +2.60% worse |
| 3s | NSS | 2.2879 | 2.2859 | −0.09% |
| 7s | CC | 0.7523 | 0.7519 | −0.05% |
| 7s | KLD | 0.4402 | 0.4488 | +1.95% worse |
| 7s | NSS | 1.8892 | 1.8874 | −0.10% |

KLD is the most sensitive metric — all three INT8 models regress 1.7–2.6%. That's expected: KLD measures distribution divergence, and INT8's discretisation most affects low-confidence tail values, which shifts the predicted distribution's shape slightly. CC and NSS move by less than 0.1% in all cases; the peaks survive INT8 cleanly.

**How to choose:**

- **FP16 (57 MB)** — default choice. Near-PyTorch quality, fits modern browser-cache budgets.
- **INT8 (32 MB)** — use when download size is the binding constraint (multi-model fetches, low-bandwidth users, strict cold-load budgets). Accept ~2% KLD regression; visual quality on benchmark images is indistinguishable.

FP32 ONNX is available via `export_onnx.py` without the `--fp16` flag (~106 MB per model) but isn't attached to the release — re-export from the `best.pt` checkpoint if you need FP32.

The investigation story behind why INT8's random-uniform structural parity numbers (8–14% max-pixel error) looked alarming but the real metric quality was fine is in [`LEARNINGS.md`](../LEARNINGS.md) under 2026-04-18.

---

## What's NOT configurable without code changes

- **Architecture.** MSI-Net is fixed. Swapping to a different backbone (ResNet, EfficientNet) is a research project, not a config knob.
- **Dataset.** UEyes only. Adding a second dataset (e.g. WebSight, UMSI++) requires code in `ueyes_dataset.py` or a new loader.
- **Loss function.** KL divergence, ported from Kroner's reference. Alternatives (MSE, BCE, combined CC+KL) would need changes in `losses.py` and `train.py`.
- **Data augmentation.** Currently none. Adding random crops, flips, colour jitter would need a `transform` pipeline in `ueyes_dataset.py`.

Each of these is a bigger change than a config tweak. If you're considering one, open an issue to scope it before coding.

---

## Typical experiment checklist

Before running:
- [ ] Config changes documented (which knob, which value, why you're trying it)
- [ ] `weights/msinet_salicon.pt` exists (the pretrained init)
- [ ] `data/ueyes/UEyes_dataset/` exists (the training data)
- [ ] `.venv/bin/pytest` still passes (no code regression from your edits)

After running:
- [ ] Row added to the experiment log above
- [ ] `--compare` eval against stock AND against v3-baseline
- [ ] If better: render qualitative comparison on the ACS welcome page + UEyes samples
- [ ] If shipping: export `.onnx`, validate parity, update this guide's "shipped" row
- [ ] LEARNINGS entry if something surprising happened
