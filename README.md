# foveacast-training

Training pipeline for the saliency model that ships in [Foveacast](https://github.com/khawkins98/Foveacast).

This repo is the upstream producer; Foveacast is the downstream consumer. A training run here produces a single `.onnx` artefact that gets committed into Foveacast's `docs/models/` folder and loaded at runtime by `onnxruntime-web`. The split exists because Foveacast is a buildless static web app and model training is not — different languages, different dependencies, different release cadences, different concerns about dataset storage. Keeping them separate lets Foveacast stay shaped as "clone, `pnpm install`, `pnpm dev`" and lets this repo be shaped as "clone, set up a GPU or MPS environment, download a 12.9 GB dataset, train a saliency model."

## What we're trying to do, specifically

V1 of Foveacast shipped [MSI-Net](https://github.com/alexanderkroner/saliency) (Kroner et al., 2020) via TensorFlow.js. V2 spiked [UNISAL](https://github.com/rdroste/unisal) through ONNX Runtime Web. Both are SALICON-trained — which means both are primarily trained on natural photographs, not UI content. A benchmark against real eye-tracking ground truth from the UEyes study made that limitation visible: the models miss CTAs, don't weigh buttons differently from surrounding text, and produce diffuse centrality blobs on structured UI layouts.

The fix is not swapping to yet another natural-scene-trained model. The fix is training on UI content. This repo fine-tunes MSI-Net (permissively licensed, architecturally well-understood) on the [UEyes dataset](https://zenodo.org/records/8010312) (1,980 UI screenshots with real participant eye-tracking, permissively licensed). The goal is a saliency model that actually knows what a "Begin" button is and weighs it accordingly.

The longer-form history — V1 build, V2 ONNX spike, the ground-truth benchmark that exposed the SALICON limitation, the wider model survey, the pivot from a UMSI++ drop-in to this fine-tuning approach — lives in Foveacast's [`LEARNINGS.md`](https://github.com/khawkins98/Foveacast/blob/main/LEARNINGS.md). Read that first if the project's trajectory matters to what you're trying to do here.

## Attribution chain

Three pieces of other people's work make this possible. Credit where it's due, and licences where they're needed.

- **MSI-Net — model architecture.** Kroner et al. 2020, "Contextual Encoder-Decoder Network for Visual Saliency Prediction," *Neural Networks*. MIT licensed. Repo: [github.com/alexanderkroner/saliency](https://github.com/alexanderkroner/saliency). Preprint: [arXiv:1902.06634](https://arxiv.org/abs/1902.06634).
- **UEyes — fine-tuning dataset.** Jiang, Y., Leiva, L. A., Rezazadegan Tavakoli, H., Houssel, P. R. B., Kylmälä, J., & Oulasvirta, A. (2023). UEyes: Understanding Visual Saliency across User Interface Types. In *Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems*, Article 285, 1–21. [doi:10.1145/3544548.3581096](https://doi.org/10.1145/3544548.3581096). Dataset hosted on Zenodo at record [8010312](https://zenodo.org/records/8010312) under Creative Commons Attribution 4.0 International. Attribution required, commercial use permitted.
- **Foveacast — consumer application.** [github.com/khawkins98/Foveacast](https://github.com/khawkins98/Foveacast). MIT licensed.

This repo itself is MIT-licensed (see [LICENSE](LICENSE)). The trained model artefacts released from this repo inherit the attribution requirements of the UEyes dataset — if you ship a model trained with this pipeline, cite Jiang et al. 2023 in whatever consumes it. Foveacast does this in its attribution footer; a different downstream user would need to do the equivalent.

## Layout

```
foveacast-training/
├── README.md              # this file
├── LEARNINGS.md           # dated prose log of decisions and dead ends
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
│   ├── msinet.py          # model architecture (vendored or ported)
│   ├── ueyes_dataset.py   # PyTorch Dataset wrapping UEyes
│   ├── train.py           # training loop
│   ├── eval.py            # saliency metrics (CC, KLD, NSS) on held-out UEyes
│   └── export_onnx.py     # produce the release artefact
│
├── notebooks/
│   └── prototype.ipynb    # "100 images, 2 epochs" sanity check
│
├── benchmark/
│   └── screenshots/       # Foveacast's comparison set, for qualitative review
│
└── runs/                  # gitignored; checkpoints + TensorBoard logs
```

## Setup

You need Python 3.12, `uv`, and an Apple-Silicon Mac (MPS) or a CUDA GPU. The training pipeline uses PyTorch's automatic device detection so both work without code changes.

```sh
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[training]'
```

Fetch the UEyes dataset (12.9 GB zip). Instructions in [`data/README.md`](data/README.md).

## Reproduce the current release

Placeholder — populated as the first training run lands.

1. Prototype sanity check: `python -m foveacast_training.train --prototype`. Runs 2 epochs on 100 images. Should converge to a plausible-looking loss curve in ~20 minutes on an M4 MacBook Air.
2. Full fine-tune: `python -m foveacast_training.train --config configs/v3-msinet-ueyes.yaml`. Expected wall-clock on M4 Air: 4–12 hours. On a CUDA T4 or similar: probably under an hour.
3. Evaluate: `python -m foveacast_training.eval --checkpoint runs/v3-msinet-ueyes/best.pt`. Produces CC / KLD / NSS scores on a held-out UEyes split and qualitative saliency maps for the committed benchmark screenshots.
4. Export: `python -m foveacast_training.export_onnx --checkpoint runs/v3-msinet-ueyes/best.pt --out releases/foveacast-v3.onnx`.

## Releases

Each trained model is tagged and released via GitHub Releases with the `.onnx` artefact attached. Foveacast consumes a specific release tag, recorded in Foveacast's CHANGELOG entry for the corresponding shipping version.

Releases so far: none yet.

## Why not train from scratch?

MSI-Net was trained on SALICON — 10,000 images with mouse-as-proxy-for-gaze data and 5,000 with real eye-tracking. That training gave the model a good prior for how humans look at images in general: contrast, edges, faces, centrality. Fine-tuning from those pretrained weights on 1,980 UI screenshots is cheap and effective; training from random initialisation on 1,980 screenshots alone would produce a weaker model because 1,980 is small.

The exception would be if we wanted to train an architecture for which no UI-trained checkpoint is available — which is an interesting research question but out of scope for a repo whose job is to produce something Foveacast can ship.

## Licences, in one place

This repo — MIT. Model architecture vendored / ported from MSI-Net — MIT. Training dataset — UEyes, CC BY 4.0, attribution to Jiang et al. 2023. Released model artefacts — MIT code, but downstream use must carry the UEyes citation per CC BY 4.0. Python dependencies — each under its own licence; PyTorch is BSD-3, numpy is BSD, Pillow is MIT-CMU.

Nothing here depends on closed-source or non-commercial components.
