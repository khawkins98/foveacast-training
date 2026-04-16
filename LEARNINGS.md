# Learnings

Running log of what I found out while building `foveacast-training`. Prose, dated, informal. Author: Ken Hawkins.

Same convention as [Foveacast's `LEARNINGS.md`](https://github.com/khawkins98/Foveacast/blob/main/LEARNINGS.md) — this is not a changelog, it is the place where decisions, dead ends, and "oh, that's how that actually works" moments live. If you are trying to understand *why* the code in this repo is shaped the way it is, this file is more useful than the diff.

The *why for the repo existing at all* is in Foveacast's LEARNINGS, specifically the entries dated 2026-04-16 on the model survey, the UMSI++ correction, and the companion-repo split. This file picks up from there.

---

## 2026-04-16 — Why this repo exists

Foveacast V1 shipped [MSI-Net](https://github.com/alexanderkroner/saliency) via TensorFlow.js. V2 spiked [UNISAL](https://github.com/rdroste/unisal) through ONNX Runtime Web. Both are trained on SALICON (natural scenes). A benchmark against real eye-tracking ground truth from the [UEyes study](https://doi.org/10.1145/3544548.3581096) made the consequence visible: both models miss buttons, CTAs, and other UI-specific attention targets that real users actually look at. Neither is well-suited for the screenshots Foveacast users actually drop on the tool.

The fix is to train on UI content. The cleanest path turns out to be fine-tuning MSI-Net (MIT) on the UEyes dataset (CC BY 4.0) rather than chasing a drop-in UI-trained checkpoint elsewhere. Licence chain is clean end to end, compute is modest, architectures are well-understood.

That training work is a different shape of project than Foveacast itself — Python, PyTorch, 12.9 GB dataset, checkpoints, TensorBoard — so it gets its own repo. See Foveacast's LEARNINGS entry titled "Splitting the model-training work into a companion repo" for the longer argument.

## 2026-04-16 — Scaffolding choices

A few small decisions worth recording before the training code starts landing.

**Python 3.12 via `uv`, not conda.** Foveacast already uses `uv` for its UNISAL export script; keeping the same tool across both repos avoids a "wait, was this one conda or pip?" moment in six months. `uv` also handles Python-version installs itself, which means a contributor on any machine gets the same Python without fighting pyenv.

**PyTorch, not TensorFlow / Keras.** MSI-Net's original code is Keras/TF 1.x from 2020. That codebase still works but is increasingly hard to install cleanly on modern Python. Porting the architecture to PyTorch is maybe half a day of work and gets us onto the stack Foveacast's V2 ORT-export pipeline already uses. PyTorch also has first-class Apple Silicon support via MPS, which matters if the primary dev machine is a MacBook Air.

**Optional-dependency split in `pyproject.toml`.** Separate groups for `training`, `eval`, and `dev`. A downstream user who clones this repo only to re-export an existing checkpoint should not have to install TensorBoard, SciPy, and Jupyter. Lean core, opt-in extras.

**No CI yet.** For a research repo with one maintainer, a CI workflow would mostly be performative. Deferring until there is something specific worth testing automatically (e.g. the ONNX export step, which does have a contract Foveacast depends on).

**No LFS for the dataset.** UEyes is 12.9 GB zipped. Git LFS is the wrong tool for research datasets at this size — expensive, slow, not what LFS is designed for. The dataset gets fetched via `data/fetch.sh` from Zenodo, lives outside the repo in a gitignored folder, and the repo just carries the fetch instructions.

## 2026-04-16 — Phase 0 kickoff: scripted fetch and a CLAUDE.md

Starting on [issue #1](https://github.com/khawkins98/foveacast-training/issues/1)'s phased plan. First branch is narrow by design: add a repeatable UEyes fetcher, a CLAUDE.md so future sessions load the right conventions, and the scaffolding around it. No model code, no loader, no training — just the groundwork for closing Phase 0's gate ("we should know the exact shape of inputs and ground truths before writing a loader for them").

**Scripted fetch instead of manual curl.** The original `data/README.md` walked a human through `mkdir && curl && unzip`. Replacing that with `data/fetch.sh` costs a few lines and buys three things: resume support (`curl -C -`) on a 13 GB download that will get interrupted at least once; idempotent re-runs so a contributor doesn't have to remember whether they already unzipped; and a depth-2 tree dump at the end that Phase 0 can paste straight into the README's "Verified directory layout" section. The manual instructions stay in the README as a fallback.

**Why not fetch in CI?** The dataset is too large to put through CI even once, and this repo has no CI yet anyway. A local script is the right shape.

**Device target: MPS first.** Primary dev machine is an M4 MacBook Air. Training code will auto-detect (`mps → cuda → cpu`) but the proving ground is MPS. CUDA should work for anyone on a GPU box but is not the test bed. Captured as a convention in the new CLAUDE.md with the detection snippet inlined so it's not up for rediscussion every time.

**CLAUDE.md adapted from Foveacast's.** Foveacast's [`CLAUDE.md`](https://github.com/khawkins98/Foveacast/blob/main/CLAUDE.md) is several months of accumulated convention; forking it for a Python research repo was cheaper than writing from scratch. Dropped the JS/browser-specific sections (layer discipline, tfjs, heatmap.js, Vite, accessibility) and added the pieces this repo actually needs: attribution as a hard rule, MPS-first, extras-split dependency discipline, LEARNINGS-as-workflow, one-phase-per-PR. Humanizer, commit hygiene, co-author trailer, review-not-fix — all carried over unchanged.

**Open decisions parked.** Issue #1 flags four: port MSI-Net to PyTorch vs vendor Keras, input resolution, train/val/test split, device. None block Phase 0. All four will need answers before Phase 1 starts, which is why they live on a separate note rather than getting decided here.
