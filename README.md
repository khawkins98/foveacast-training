# foveacast-training

**Status: pre-release.** The fine-tuned model is trained and evaluated but not yet tagged as a GitHub Release. See [Running the pipeline](#running-the-pipeline) for what works today.

Training pipeline for the saliency model that ships in [Foveacast](https://github.com/khawkins98/Foveacast).

This repo is the upstream producer; Foveacast is the downstream consumer. A training run here produces a single `.onnx` artefact that gets committed into Foveacast's `docs/models/` folder and loaded at runtime by `onnxruntime-web`. The split exists because Foveacast is a buildless static web app and model training is not — different languages, different dependencies, different release cadences, different concerns about dataset storage. Keeping them separate lets Foveacast stay shaped as "clone, `pnpm install`, `pnpm dev`" and lets this repo be shaped as "clone, set up a GPU or MPS environment, download a 12.9 GB dataset, train a saliency model."

## Results

Fine-tuned MSI-Net vs stock (SALICON-pretrained) on the held-out UEyes test split (108 images, never seen during training):

| metric | fine-tuned | stock | improvement |
|---|---|---|---|
| **CC** (correlation coefficient, higher better) | 0.7068 ± 0.105 | 0.4934 ± 0.094 | **+43%** |
| **KLD** (KL divergence, lower better) | 0.6574 ± 0.210 | 1.1682 ± 0.246 | **-44%** |
| **NSS** (normalised scanpath saliency, higher better) | 2.2879 ± 0.605 | 1.5776 ± 0.451 | **+45%** |

Fine-tuning on UI content closes the gap between "natural-scene saliency model" and "model that knows what a button is." Here's what that looks like on a real web page (from the held-out UEyes test split, with ground-truth eye-tracking for reference):

| Source screenshot | Ground truth (real eye-tracking) | Stock MSI-Net (SALICON-only) | Fine-tuned on UEyes |
|---|---|---|---|
| ![source](benchmark/screenshots/ueyes-8f9844-source.png) | ![ground truth](benchmark/screenshots/ueyes-8f9844-ground-truth.png) | ![stock](benchmark/screenshots/ueyes-8f9844-stock.png) | ![fine-tuned](benchmark/screenshots/ueyes-8f9844-finetuned.png) |

The stock model produces a diffuse centrality blob. The fine-tuned model picks up the navigation, content headings, and interactive elements — matching where real users actually looked. Full visual comparison across four UI categories at [`benchmark/screenshots/comparison.html`](benchmark/screenshots/comparison.html).

The release artefact is a 57 MB `.onnx` file (FP16 quantised, opset 17). PyTorch ↔ onnxruntime CPU parity validated at max abs err < 1e-3.

## Quick start — inference only

If you just want to run the model on a screenshot (no training, no dataset):

```python
import numpy as np
import onnxruntime as ort
from PIL import Image

# Load the ONNX model (download from GitHub Releases when available,
# or export locally via: python -m foveacast_training.export_onnx ...)
sess = ort.InferenceSession("foveacast-v3.onnx")

# Preprocess: resize to 240×320, RGB float32 in [0, 255]
img = Image.open("my-screenshot.png").convert("RGB").resize((320, 240))
x = np.array(img, dtype=np.float32).transpose(2, 0, 1)[np.newaxis]  # (1, 3, 240, 320)

# Run inference
saliency = sess.run(["output"], {"input": x})[0]  # (1, 1, 240, 320), values in [0, 1]
```

The output is a single-channel saliency map normalised to [0, 1]. Higher values = higher predicted attention. Resize back to your original image dimensions and overlay as a heatmap.

## Limitations and intended use

The model is trained on the [UEyes dataset](https://doi.org/10.1145/3544548.3581096) — 1,980 UI screenshots across four categories (desktop, mobile, web, poster) with eye-tracking from 62 participants. It predicts where users are likely to look on a UI screenshot.

**Known limitations:**

- **Training distribution.** UEyes is primarily Western-language desktop and mobile UI from 2020–2022. The model may not generalise well to right-to-left layouts, non-Latin text, dark-mode UIs, or design patterns that emerged after the dataset was collected.
- **Participant demographics.** The 62 participants in the UEyes study were from a specific demographic pool (see the [UEyes paper](https://doi.org/10.1145/3544548.3581096) for details). Saliency predictions reflect the viewing patterns of that group, not a universal human baseline.
- **Resolution.** The model operates at 240×320 — small UI elements (fine text, tiny icons) below that resolution's ability to resolve may not produce meaningful saliency signal.
- **Not a click predictor.** Saliency predicts visual attention ("where do eyes go"), not interaction intent ("where will users click"). High saliency on a decorative element does not mean users will interact with it.
- **Single-image, no context.** The model sees one screenshot at a time. It has no concept of user task, scroll position, prior page, or dynamic content.

**Intended use:** early-stage design feedback on UI layouts — "does the hero image compete with the CTA for attention?" Not a replacement for real user testing.

## What we're trying to do, specifically

V1 of Foveacast shipped [MSI-Net](https://github.com/alexanderkroner/saliency) (Kroner et al., 2020) via TensorFlow.js. V2 spiked [UNISAL](https://github.com/rdroste/unisal) through ONNX Runtime Web. Both are SALICON-trained — which means both are primarily trained on natural photographs, not UI content. A benchmark against real eye-tracking ground truth from the UEyes study made that limitation visible: the models miss CTAs, don't weigh buttons differently from surrounding text, and produce diffuse centrality blobs on structured UI layouts.

The fix is not swapping to yet another natural-scene-trained model. The fix is training on UI content. This repo fine-tunes MSI-Net (permissively licensed, architecturally well-understood) on the [UEyes dataset](https://zenodo.org/records/8010312) (1,980 UI screenshots with real participant eye-tracking, permissively licensed). The goal is a saliency model that actually knows what a "Begin" button is and weighs it accordingly.

The longer-form history — V1 build, V2 ONNX spike, the ground-truth benchmark that exposed the SALICON limitation, the wider model survey, the pivot from a UMSI++ drop-in to this fine-tuning approach — lives in Foveacast's [`LEARNINGS.md`](https://github.com/khawkins98/Foveacast/blob/main/LEARNINGS.md). Read that first if the project's trajectory matters to what you're trying to do here.

For the end-to-end shape of the pipeline in *this* repo — what each module does, how the two repos fit together, and the contract on the `.onnx` artefact Foveacast consumes — see [`ARCHITECTURE.md`](ARCHITECTURE.md). If you're about to open a PR, [`CONTRIBUTING.md`](CONTRIBUTING.md) has the workflow conventions and a docs-lockstep checklist to run through before marking it ready.

## Attribution and citation

Three pieces of other people's work make this possible. If you use the model or the pipeline, carry the citations.

- **MSI-Net — model architecture.** Kroner, A., Senden, M., Driessens, K., & Goebel, R. (2020). Contextual Encoder-Decoder Network for Visual Saliency Prediction. *Neural Networks*, 129, 261–270. [doi:10.1016/j.neunet.2020.05.004](https://doi.org/10.1016/j.neunet.2020.05.004). Preprint: [arXiv:1902.06634](https://arxiv.org/abs/1902.06634). MIT licensed. Repo: [github.com/alexanderkroner/saliency](https://github.com/alexanderkroner/saliency).

- **UEyes — fine-tuning dataset.** Jiang, Y., Leiva, L. A., Rezazadegan Tavakoli, H., Houssel, P. R. B., Kylmälä, J., & Oulasvirta, A. (2023). UEyes: Understanding Visual Saliency across User Interface Types. In *Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems (CHI '23)*, Article 285, 1–21. [doi:10.1145/3544548.3581096](https://doi.org/10.1145/3544548.3581096). Dataset hosted on Zenodo at record [8010312](https://zenodo.org/records/8010312) under Creative Commons Attribution 4.0 International. Attribution required, commercial use permitted.

- **Foveacast — consumer application.** [github.com/khawkins98/Foveacast](https://github.com/khawkins98/Foveacast). MIT licensed.

This repo itself is MIT-licensed (see [LICENSE](LICENSE)). The trained model artefacts released from this repo inherit the attribution requirements of the UEyes dataset — if you ship a model trained with this pipeline, cite Jiang et al. 2023 in whatever consumes it. Foveacast does this in its attribution footer; a different downstream user would need to do the equivalent.

## Layout

```
foveacast-training/
├── README.md              # this file
├── ARCHITECTURE.md        # end-to-end pipeline shape + .onnx contract with Foveacast
├── CONTRIBUTING.md        # PR workflow + docs-lockstep checklist
├── LEARNINGS.md           # dated prose log of decisions and dead ends
├── CLAUDE.md              # conventions for AI assistants working in the repo
├── LICENSE                # MIT, Ken Hawkins
├── CITATION.cff           # structured citation for the project itself
├── pyproject.toml         # Python 3.12, uv-managed
├── .gitignore             # excludes data/, runs/, checkpoints, venv
│
├── data/
│   └── README.md          # how to fetch UEyes; not committed to the repo
│
├── src/foveacast_training/
│   ├── __init__.py
│   ├── msinet.py          # PyTorch port of MSI-Net (architecture + forward pass)
│   ├── ueyes_dataset.py   # PyTorch Dataset wrapping UEyes
│   ├── losses.py          # KL divergence loss + CC and NSS metrics
│   ├── train.py           # fine-tuning loop (prototype + full modes)
│   ├── eval.py            # quantitative evaluation (CC, KLD, NSS)
│   └── export_onnx.py     # ONNX export + onnxruntime parity validation
│
├── scripts/
│   ├── import_msinet_weights.py  # one-time TF→PyTorch weight port
│   └── render_saliency.py       # qualitative heatmap overlay renderer
│
├── tests/
│   ├── test_msinet_parity.py     # PyTorch-vs-TF numerical parity
│   ├── test_ueyes_dataset.py     # dataset split + shape invariants
│   └── test_losses.py            # loss/metric unit tests
│
├── benchmark/screenshots/        # qualitative comparison images + HTML viewer
├── docs/training-guide.md        # how to reproduce, customise, and compare weight sets
├── weights/                      # gitignored; imported pretrained weights (.pt)
└── runs/                         # gitignored; training checkpoints + history
```

## Setup

You need Python 3.12, `uv`, and an Apple-Silicon Mac (MPS) or a CUDA GPU. The training pipeline uses PyTorch's automatic device detection so both work without code changes.

```sh
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
```

**Convention on invoking Python.** Every command in this README uses absolute paths into `.venv/bin/` (`.venv/bin/python`, `.venv/bin/pytest`) rather than `source .venv/bin/activate` first. This is deliberate: commands copy-paste cleanly regardless of shell state, and there is no ambient-activation footgun where a forgotten `deactivate` sends a command into the system Python. If you prefer activation, the commands still work — just drop the `.venv/bin/` prefix.

The extras groups are:

- `[dev]` — Ruff, Jupyter, pytest. Always install this; covers test-suite running.
- `[training]` — TensorBoard, tqdm, PyYAML. For the full fine-tune with logging.
- `[eval]` — SciPy, scikit-image. For saliency metrics beyond the core CC/KLD/NSS.
- `[weights-import]` — TensorFlow (~500 MB), huggingface_hub. Only needed for the one-time weight port described in the next section. Disposable afterwards — `uv pip uninstall tensorflow` frees the disk space once you have `weights/msinet_salicon.pt`.

Install multiple at once by comma-separating: `.[dev,weights-import]`.

Fetch the UEyes dataset (12.9 GB zip). Instructions in [`data/README.md`](data/README.md). Only needed for training and evaluation — not for running the parity test or the ONNX export.

## Import the pretrained MSI-Net weights

A one-time step per contributor. Kroner's SALICON-pretrained MSI-Net ships as a TensorFlow SavedModel on [HuggingFace](https://huggingface.co/alexanderkroner/MSI-Net); `scripts/import_msinet_weights.py` downloads that deposit, walks the frozen inference graph to extract weights stored as named `Const` ops (see [`LEARNINGS.md`](LEARNINGS.md) 2026-04-16 for why this is harder than it sounds), transposes conv kernels from TF's HWIO layout to PyTorch's OIHW, and writes a PyTorch `state_dict` that `foveacast_training.msinet.MSINet` loads with `strict=True`.

**Cost to budget:** ~500 MB on disk for TensorFlow (uninstallable afterwards), ~100 MB network for the HF snapshot (cached at `~/.cache/huggingface/hub` by default — reused across re-runs), ~30 seconds wall-clock for the import itself. ~5–10 minutes total on a cold `uv venv` dominated by TF's install.

```sh
uv pip install --python .venv/bin/python -e '.[weights-import]'
.venv/bin/python scripts/import_msinet_weights.py
```

The importer is idempotent: re-running it re-uses the cached HF snapshot and over-writes the same `.pt` each time. If the variable ordering breaks on a newer HF revision, pass `--discover` to dump the graph's float32 constants and diagnose.

After the import, the TensorFlow install is no longer needed for anything else in this repo. `uv pip uninstall tensorflow tf_keras` frees the disk space; the resulting `weights/msinet_salicon.pt` is all that subsequent steps depend on.

## Tests

```sh
.venv/bin/pytest                                      # the whole suite
.venv/bin/pytest tests/test_msinet_parity.py -v -s    # parity test, verbose
.venv/bin/pytest tests/test_ueyes_dataset.py -v       # dataset loader tests
.venv/bin/pytest tests/test_losses.py -v              # loss/metric tests
```

Tests skip cleanly when their prerequisites aren't installed:

- `test_msinet_parity.py` skips without `[weights-import]` extras + `weights/msinet_salicon.pt`.
- `test_ueyes_dataset.py` skips without `data/ueyes/UEyes_dataset/`. Override path with `UEYES_ROOT=/some/other/path`.
- `test_losses.py` always runs (no external dependencies).

A green summary with skips is NOT the same as all tests passing. Check the `-v` output to confirm which tests actually ran.

**Numerical parity test:** `test_msinet_parity.py` runs a fixed-seed input through both the PyTorch port and Kroner's reference TF SavedModel, then asserts outputs match within `atol=1e-5`. Observed residual: mean ~1.1e-7, max ~1.4e-6 — float32 machine-epsilon territory. This is the critical gate that confirms the PyTorch port is faithful to the original.

**Dataset loader tests:** `test_ueyes_dataset.py` validates split invariants (counts sum to 1,980, no filename overlap, train/val stratified by category), sample shapes and dtypes, value ranges, reproducibility of the val split, and pickleability for DataLoader multiprocessing.

## Running the pipeline

The full pipeline: import weights → fetch dataset → train → evaluate → export. Each step builds on the previous.

### 1. Prototype sanity check

```sh
.venv/bin/python -m foveacast_training.train --prototype
```

Runs 2 epochs on 100 train / 25 val images under a fixed seed (bit-reproducible across reruns). Writes `runs/prototype-{timestamp}/history.json`. On an M4 MacBook Air this runs in ~90 seconds. Requires `weights/msinet_salicon.pt` and `data/ueyes/UEyes_dataset/`.

| metric | epoch 1 | epoch 2 | direction |
|---|---|---|---|
| train loss (avg) | 0.8928 | 0.7612 | ↓ 15% |
| val loss | 0.9006 | 0.8297 | ↓ 8% |
| val CC | 0.6012 | 0.6421 | ↑ 6.8% |

### 2. Full fine-tune

```sh
.venv/bin/python -m foveacast_training.train --full
```

Explicit `--full` flag required (no default mode) so it's not possible to accidentally kick off a 4-hour fine-tune by forgetting `--prototype`. Runs the full 1,684-image train set with gradient clipping, best-checkpoint saving on validation CC, ReduceLROnPlateau scheduler, and early stopping. Writes `runs/full-{timestamp}/{best.pt,final.pt,history.json,best.json}`.

The first full run (M4 MPS, ~3.5 hours, 30 epochs) produced a best checkpoint at epoch 25 with val CC = 0.7247. See [`docs/training-guide.md`](docs/training-guide.md) for the full training config, the knobs to turn for custom experiments, and the experiment log.

### 3. Evaluate

```sh
# Compare fine-tuned vs stock:
.venv/bin/python -m foveacast_training.eval \
    --checkpoint runs/full-*/best.pt \
    --compare weights/msinet_salicon.pt
```

Computes CC, KLD, and NSS over the held-out UEyes test split (108 images). Results table in [Results](#results) above.

### 4. Export to ONNX

```sh
.venv/bin/python -m foveacast_training.export_onnx \
    --checkpoint runs/full-*/best.pt \
    --out releases/foveacast-v3.onnx
```

Produces a single-file `.onnx` artefact with PyTorch ↔ onnxruntime CPU parity validated across 6 trial inputs (random, batched, saturated). FP16 quantisation available post-export to halve the artefact from ~106 MB to ~57 MB with <1% pixel-level quality loss. See [`docs/training-guide.md`](docs/training-guide.md) for details.

## Qualitative comparison

Visual comparison of stock (SALICON-pretrained) vs fine-tuned (UEyes-trained) vs real eye-tracking ground truth:

[`benchmark/screenshots/comparison.html`](benchmark/screenshots/comparison.html) — open locally to see the full comparison page. Includes weight provenance for every column, the quantitative numbers for context, and an FP32 vs FP16 quality comparison section.

## Customising the fine-tune

See [`docs/training-guide.md`](docs/training-guide.md) for:

- The exact shipped training configuration with selection rationale
- The five knobs worth turning (saliency variant, learning rate, input resolution, epochs, validation split)
- Step-by-step custom experiment workflow
- A growing experiment log that tracks every run with its key metrics

## Why not train from scratch?

MSI-Net was trained on SALICON — 10,000 images with mouse-as-proxy-for-gaze data and 5,000 with real eye-tracking. That training gave the model a good prior for how humans look at images in general: contrast, edges, faces, centrality. Fine-tuning from those pretrained weights on 1,980 UI screenshots is cheap and effective; training from random initialisation on 1,980 screenshots alone would produce a weaker model because 1,980 is small.

The exception would be if we wanted to train an architecture for which no UI-trained checkpoint is available — which is an interesting research question but out of scope for a repo whose job is to produce something Foveacast can ship.

## How to cite this work

If you use the fine-tuned model or this training pipeline in your research, please cite both the upstream works and this project:

```bibtex
@software{hawkins2026foveacast_training,
  title   = {foveacast-training: UI-aware saliency model via MSI-Net fine-tuned on UEyes},
  author  = {Hawkins, Ken},
  year    = {2026},
  url     = {https://github.com/khawkins98/foveacast-training},
  license = {MIT}
}

@article{kroner2020contextual,
  title   = {Contextual Encoder-Decoder Network for Visual Saliency Prediction},
  author  = {Kroner, Alexander and Senden, Mario and Driessens, Kurt and Goebel, Rainer},
  journal = {Neural Networks},
  volume  = {129},
  pages   = {261--270},
  year    = {2020},
  doi     = {10.1016/j.neunet.2020.05.004}
}

@inproceedings{jiang2023ueyes,
  title     = {UEyes: Understanding Visual Saliency across User Interface Types},
  author    = {Jiang, Yue and Leiva, Luis A. and Rezazadegan Tavakoli, Hamed and
               Houssel, Paul R. B. and Kylm{\"a}l{\"a}, Julia and Oulasvirta, Antti},
  booktitle = {Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems},
  articleno = {285},
  pages     = {1--21},
  year      = {2023},
  doi       = {10.1145/3544548.3581096}
}
```

Structured citation metadata also available in [`CITATION.cff`](CITATION.cff).

## Licences, in one place

This repo — MIT. Model architecture ported from MSI-Net — MIT. Training dataset — UEyes, CC BY 4.0, attribution to Jiang et al. 2023. Released model artefacts — MIT code, but downstream use must carry the UEyes citation per CC BY 4.0. Python dependencies — each under its own licence; PyTorch is BSD-3, numpy is BSD, Pillow is MIT-CMU.

Nothing here depends on closed-source or non-commercial components.
