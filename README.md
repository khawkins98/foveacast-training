# foveacast-training

Training pipeline for the saliency model that ships in [Foveacast](https://github.com/khawkins98/Foveacast).

This repo is the upstream producer; Foveacast is the downstream consumer. A training run here produces a single `.onnx` artefact that gets committed into Foveacast's `docs/models/` folder and loaded at runtime by `onnxruntime-web`. The split exists because Foveacast is a buildless static web app and model training is not — different languages, different dependencies, different release cadences, different concerns about dataset storage. Keeping them separate lets Foveacast stay shaped as "clone, `pnpm install`, `pnpm dev`" and lets this repo be shaped as "clone, set up a GPU or MPS environment, download a 12.9 GB dataset, train a saliency model."

## What we're trying to do, specifically

V1 of Foveacast shipped [MSI-Net](https://github.com/alexanderkroner/saliency) (Kroner et al., 2020) via TensorFlow.js. V2 spiked [UNISAL](https://github.com/rdroste/unisal) through ONNX Runtime Web. Both are SALICON-trained — which means both are primarily trained on natural photographs, not UI content. A benchmark against real eye-tracking ground truth from the UEyes study made that limitation visible: the models miss CTAs, don't weigh buttons differently from surrounding text, and produce diffuse centrality blobs on structured UI layouts.

The fix is not swapping to yet another natural-scene-trained model. The fix is training on UI content. This repo fine-tunes MSI-Net (permissively licensed, architecturally well-understood) on the [UEyes dataset](https://zenodo.org/records/8010312) (1,980 UI screenshots with real participant eye-tracking, permissively licensed). The goal is a saliency model that actually knows what a "Begin" button is and weighs it accordingly.

The longer-form history — V1 build, V2 ONNX spike, the ground-truth benchmark that exposed the SALICON limitation, the wider model survey, the pivot from a UMSI++ drop-in to this fine-tuning approach — lives in Foveacast's [`LEARNINGS.md`](https://github.com/khawkins98/Foveacast/blob/main/LEARNINGS.md). Read that first if the project's trajectory matters to what you're trying to do here.

For the end-to-end shape of the pipeline in *this* repo — what each module does, how the two repos fit together, and the contract on the `.onnx` artefact Foveacast consumes — see [`ARCHITECTURE.md`](ARCHITECTURE.md).

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
├── ARCHITECTURE.md        # end-to-end shape of the pipeline + .onnx contract with Foveacast
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
│   ├── msinet.py          # PyTorch port of MSI-Net (Phase 2, landed)
│   ├── ueyes_dataset.py   # PyTorch Dataset wrapping UEyes (Phase 3)
│   ├── train.py           # training loop (Phase 4/5)
│   ├── eval.py            # saliency metrics (CC, KLD, NSS) (Phase 6)
│   └── export_onnx.py     # produce the release artefact (Phase 8)
│
├── scripts/
│   └── import_msinet_weights.py  # one-time TF→PyTorch weight port
│
├── tests/
│   └── test_msinet_parity.py     # PyTorch-vs-TF numerical parity gate
│
├── weights/               # gitignored; imported pretrained weights (.pt)
└── runs/                  # gitignored; checkpoints + TensorBoard logs (Phase 4+)
```

`benchmark/screenshots/` (for Phase 7 qualitative eval) and `notebooks/` (optional Jupyter work) are created on demand by the phases that need them.

## Setup

You need Python 3.12, `uv`, and an Apple-Silicon Mac (MPS) or a CUDA GPU. The training pipeline uses PyTorch's automatic device detection so both work without code changes.

```sh
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
```

**Convention on invoking Python.** Every command in this README uses absolute paths into `.venv/bin/` (`.venv/bin/python`, `.venv/bin/pytest`) rather than `source .venv/bin/activate` first. This is deliberate: commands copy-paste cleanly regardless of shell state, and there is no ambient-activation footgun where a forgotten `deactivate` sends a command into the system Python. If you prefer activation, the commands still work — just drop the `.venv/bin/` prefix.

The extras groups are:

- `[dev]` — Ruff, Jupyter, pytest. Always installed; covers test-suite running.
- `[training]` — TensorBoard, tqdm, PyYAML. For the fine-tune loop (Phase 4+).
- `[eval]` — SciPy, scikit-image. For the saliency metrics (Phase 6).
- `[weights-import]` — TensorFlow (~500 MB), huggingface_hub. Only needed for the one-time weight port described in the next section. Disposable afterwards — `uv pip uninstall tensorflow` frees the disk space once you have `weights/msinet_salicon.pt`.

Install multiple at once by comma-separating: `.[dev,weights-import]`.

Fetch the UEyes dataset (12.9 GB zip). Instructions in [`data/README.md`](data/README.md). Only needed for Phase 3+ — not for running the Phase 2 parity test.

## Import the pretrained MSI-Net weights

A one-time step per contributor. Kroner's SALICON-pretrained MSI-Net ships as a TensorFlow SavedModel on [HuggingFace](https://huggingface.co/alexanderkroner/MSI-Net); `scripts/import_msinet_weights.py` downloads that deposit, walks the frozen inference graph to extract weights stored as named `Const` ops (see [`LEARNINGS.md`](LEARNINGS.md) 2026-04-16 Phase 2 for why this is harder than it sounds), transposes conv kernels from TF's HWIO layout to PyTorch's OIHW, and writes a PyTorch `state_dict` that `foveacast_training.msinet.MSINet` loads with `strict=True`.

**Cost to budget:** ~500 MB on disk for TensorFlow (uninstallable afterwards), ~100 MB network for the HF snapshot (cached at `~/.cache/huggingface/hub` by default — reused across re-runs), ~30 seconds wall-clock for the import itself. ~5–10 minutes total on a cold `uv venv` dominated by TF's install.

```sh
uv pip install --python .venv/bin/python -e '.[weights-import]'
.venv/bin/python scripts/import_msinet_weights.py
```

The importer is idempotent: re-running it re-uses the cached HF snapshot and over-writes the same `.pt` each time. Real output from a fresh run (2026-04-16):

```
→ Downloading HuggingFace SavedModel (alexanderkroner/MSI-Net)...
  local path: /Users/you/.cache/huggingface/hub/models--alexanderkroner--MSI-Net/snapshots/d950b35945db961ae63f84bc2b23f6bd578d0b8f
→ Walking inference graph to find weight Const tensors...
  collected 48 float32 Const tensors from pruned
→ Building PyTorch state_dict...
  built state_dict with 46 tensors (23 layers, one weight + optional bias each)
✓ Wrote weights/msinet_salicon.pt (99.8 MB)
→ Verifying by loading into MSINet with strict=True...
  loaded 24,934,209 parameters
  forward pass on (1, 3, 240, 320) → (1, 1, 240, 320), range [0.0000, 1.0000]
✓ Import complete. Run `pytest tests/test_msinet_parity.py` to confirm numerical parity with the reference forward pass.
```

If the variable ordering breaks on a newer HF revision (unlikely but possible — the deposit could be re-exported), pass `--discover` to dump the graph's float32 constants and diagnose. The script writes nothing in that mode:

```sh
.venv/bin/python scripts/import_msinet_weights.py --discover
```

After the import, the TensorFlow install is no longer needed for anything else in this repo. `uv pip uninstall tensorflow tf_keras` frees the disk space; the resulting `weights/msinet_salicon.pt` is all that subsequent phases depend on.

## Tests

```sh
.venv/bin/pytest                          # the whole suite
.venv/bin/pytest tests/test_msinet_parity.py -v -s    # just the parity test, verbose
```

Tests skip cleanly if their prerequisites aren't installed — the parity test skips when `[weights-import]` isn't available or `weights/msinet_salicon.pt` hasn't been imported. This is intentional: a minimal install shouldn't produce false failures. It does mean a green summary with skips is NOT the same as a passing gate — check the `-v` output to confirm the parity tests ran rather than were skipped.

The current gate-closing test is `tests/test_msinet_parity.py` — it runs a fixed-seed random input through both the PyTorch port and Kroner's reference TF SavedModel, then asserts outputs match within `atol=1e-5, rtol=1e-5`. Mean / max / p99 absolute error are printed regardless of pass or fail so a commit that barely passes shows a visible regression signal. First passing run on 2026-04-16 reported mean ~1.1e-7, max ~1.4e-6, p99 ~6.0e-7 — float32 machine-epsilon territory, three orders of magnitude below the tolerance.

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
