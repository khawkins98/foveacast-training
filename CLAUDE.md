# CLAUDE.md

Instructions for Claude Code (and other AI assistants following the same convention) working in this repo. Load this before other files.

If you are a human reader: this tells the assistant how we want it to behave on `foveacast-training`. The long-form equivalents for humans are [README.md](README.md) (what runs where), [LEARNINGS.md](LEARNINGS.md) (what we figured out along the way), and the phased plan in [issue #1](https://github.com/khawkins98/foveacast-training/issues/1).

This file is the sibling of Foveacast's [`CLAUDE.md`](https://github.com/khawkins98/Foveacast/blob/main/CLAUDE.md). Where conventions are shared, they match; where they differ, it is because this repo is a Python research codebase and Foveacast is a buildless JS web app.

---

## What this project is

`foveacast-training` is the upstream producer for [Foveacast](https://github.com/khawkins98/Foveacast). A training run here fine-tunes [MSI-Net](https://github.com/alexanderkroner/saliency) (Kroner et al. 2020, MIT) on the [UEyes dataset](https://zenodo.org/records/8010312) (Jiang et al. 2023, CC BY 4.0) and exports a single `.onnx` artefact that Foveacast ships via `onnxruntime-web`.

Read these before making substantive changes:

- [README.md](README.md) — what the repo is for, how to set it up, how to reproduce.
- [ARCHITECTURE.md](ARCHITECTURE.md) — end-to-end pipeline shape, module responsibilities, the `.onnx` contract with Foveacast, and the phase map. Start here if you are about to touch anything inside `src/foveacast_training/`.
- [CONTRIBUTING.md](CONTRIBUTING.md) — PR workflow and the documentation-lockstep checklist. The checklist is load-bearing; do not open a PR that adds a module without running through it.
- [LEARNINGS.md](LEARNINGS.md) — dated prose log of decisions and dead ends. Check this before re-deriving something.
- [issue #1](https://github.com/khawkins98/foveacast-training/issues/1) — phased plan from fetch-dataset to integration-PR. The handoff brief for anyone (human or LLM) starting work on this repo.
- Foveacast's [LEARNINGS.md](https://github.com/khawkins98/Foveacast/blob/main/LEARNINGS.md) — the upstream trajectory that explains *why this repo exists at all*. The 2026-04-16 entries on model survey, UMSI++ correction, and companion-repo split are the load-bearing ones.

---

## Project-specific overrides to your defaults

These override whatever your default guidance says.

### Write ample comments

Default to writing comments, not omitting them. Every non-obvious branch gets a `# why:` inline. Every module gets a header docstring explaining what lives there and — critically — which upstream work it borrows from (MSI-Net, UEyes, anything else).

This is the opposite of the common "self-documenting code" rule and it is intentional: this is a research codebase picked up by people who did not build it, often after the original context has decayed. Comments that capture *why* outlast diffs a reader would otherwise have to reverse-engineer.

What not to comment: obvious restatements of the code, changelog chatter, PR context, commented-out code.

### Attribution is non-negotiable

Every borrowed architecture, dataset, or code block gets named credit in:

1. The module docstring at the top of the file.
2. `README.md` under the attribution chain.
3. `CITATION.cff` if it belongs in a citation graph.

MSI-Net → Kroner et al. 2020 (MIT). UEyes → Jiang et al. 2023 (CC BY 4.0). If you add a new dataset or architecture, add it to all three places in the same commit.

### Humanizer pass on prose

Anything humans read gets a humanizer pass before committing. See [`CONTRIBUTING.md` §Prose register](CONTRIBUTING.md#prose-register) for the list of AI-register tells to strip and the reference voice in this repo. As an AI assistant your baseline output drifts into that register by default, so running the pass explicitly — not relying on it happening automatically — is the thing that matters.

### Device-agnostic code, MPS as the default test bed

Primary dev machine is an M4 MacBook Air, so MPS is where code gets proven. CUDA should work via PyTorch's auto-detection but is a secondary target. CPU fallback must exist for anyone running inference-only.

```python
# why: MPS on Apple Silicon, CUDA where available, CPU otherwise. All
# three are real paths — don't hard-code cuda or assume torch.cuda.is_available().
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
```

### Dependency discipline

Core install (`pip install -e .`) stays lean — enough to load a checkpoint and export ONNX. Heavy stuff (TensorBoard, SciPy, Jupyter) lives in the `[training]`, `[eval]`, `[dev]` extras. A downstream user who only wants to re-export an existing checkpoint should not have to install the whole training stack.

Before adding a new dependency, ask: does this belong in `dependencies` or in an extra? When in doubt, an extra.

### LEARNINGS.md is part of the workflow

Any decision that took more than an hour to figure out, any dead end worth documenting, any "oh, that's how that actually works" moment gets a dated prose entry. Not a changelog; not a spec; a running log.

If a decision is reversed later, the original entry stays and a new entry supersedes it. History is more useful than an always-current document.

### Commit-by-commit hygiene

See [`CONTRIBUTING.md` §Commit hygiene](CONTRIBUTING.md#commit-hygiene) for the Conventional Commits pattern, the one-conceptual-thing-per-commit rule, and the Claude co-author trailer. No AI-specific overrides here — the workflow is the same for humans and AI.

### One phase per PR

Issue #1 splits V3 into 10 phases with explicit gates. Each phase lands as its own small PR or tagged commit sequence. Do not bundle phases. The gate is there because the next phase's approach depends on what the previous phase actually produced (e.g. the dataset loader's input shape depends on what Phase 0 discovered about the UEyes directory layout).

---

## Workflow expectations

### Before starting substantive work

1. Read issue #1 for the phase you are about to work on.
2. Check `LEARNINGS.md` for related prior investigation.
3. If open decisions in issue #1 are unresolved for your phase, raise them with the maintainer before writing code.
4. If the work spans multiple files or phases, sketch the approach and confirm before implementing.

### While working

- Keep commits atomic.
- Run `ruff check .` before committing Python changes.
- When tests land in a later phase, run them after every commit — not only at the end.
- If something breaks, investigate the root cause. Do not `--no-verify` hooks. Do not skip tests to make a commit land.

### Documentation lockstep

Run the pre-PR checklist in [`CONTRIBUTING.md`](CONTRIBUTING.md#the-documentation-lockstep) before marking a PR ready. It is the single source of truth for which files to touch when — ARCHITECTURE, LEARNINGS, README, data/README, CLAUDE, CITATION, issue #1. Prior reviewer passes found real gaps on Phases 2 and 3 that the checklist now catches pre-merge.

### Risky actions — confirm first

See [`CONTRIBUTING.md` §Risky actions](CONTRIBUTING.md#risky-actions) for the list. Same list for humans and AI; duplicating it here invites drift.

### When asked to review

If the user asks for a "review", "audit", or "critique", they usually want findings written up, not code changes applied. Produce a review document and return a short summary. Don't start applying fixes until the user has seen the review.

---

## Common pitfalls (populated as they happen)

- **UEyes zip is 12.9 GB.** Don't start a download without telling the user first. `data/fetch.sh` is idempotent and resumable; a partial download can always be picked up.
- **`data/ueyes/` is gitignored.** Anything that depends on the dataset being present must be conditional, documented, or tested against a small committed fixture.
- **MPS ≠ CUDA in subtle ways.** Some ops fall back to CPU silently (slow) or raise (loud). When in doubt, run the same forward pass on CPU and diff outputs.
- **The MSI-Net HuggingFace SavedModel is exported frozen.** Weights live as named `Const` ops inside the inference graph, not as `tf.Variable`s. `tf.saved_model.load(path).variables` and `tf.keras.models.load_model(path).weights` both return empty lists. The working path is `tf.saved_model.load()` → walk two levels of `PartitionedCall` indirection → harvest Const tensors by name from `__inference_pruned_Y`. `scripts/import_msinet_weights.py` does this; `--discover` mode prints the graph's float32 constants if you need to debug a future re-export.
- **TF 1.x bilinear ≠ any PyTorch bilinear.** `ResizeBilinear(align_corners=False, half_pixel_centers=False)` in a frozen TF 1.x-style SavedModel uses `src = dst * (src_size / dst_size)` with duplicated-edge clamping. PyTorch's `F.interpolate(mode='bilinear', align_corners=False)` uses half-pixel centers; `align_corners=True` uses endpoint-aligned sampling. Neither matches. `src/foveacast_training/msinet.py:_tf1_bilinear_upsample` is the direct-implementation helper; use it whenever porting a frozen TF graph or reach-for-parity becomes the goal.
- **UEyes `image_types.csv` has a mixed `Block` column.** Most rows have plain integers (`"0"`, `"23"`), some have Excel scientific notation (`"0,00E+00"`) because the file was opened in Excel at some point. `int(block_str)` explodes on the latter. The loader keeps `Block` as a raw string; nothing downstream uses it. Don't re-introduce the `int()` parse.
- **UEyes `overlay_heatmaps_*` filenames carry an `overlay_` prefix.** The one naming irregularity across the deposit's 12 saliency-map + scanpath variants. `UEyesDataset` resolves the prefix at construction time and caches it on `self`. If you add a variant that also has this prefix pattern, update the logic in `ueyes_dataset.UEyesDataset.__init__`.
- **`KL(p || p)` is not exactly zero under Kroner's eps formulation.** Both maps get divided by `eps + sum`, and the log argument is `eps + p/(eps + p)`. For a 240×320 saliency map this produces a ~1e-4 residual that looks like a bug but is a mathematical property of the eps handling. `tests/test_losses.py` checks `abs(loss) < 1e-3` rather than strict zero for exactly this reason. Don't tighten that tolerance without understanding the origin.
- **`train.py`'s `FULL_CONFIG` is placeholder until Phase 5.** Running `.venv/bin/python -m foveacast_training.train` without `--prototype` prints a loud warning because the hyperparameters, best-checkpoint saving, LR schedule, early stopping, and gradient clipping are all Phase 5 deliverables. Do not kick off a 4-hour run against FULL_CONFIG before Phase 5 lands — you'll produce a weak checkpoint that looks plausible from stdout but wastes the fine-tune budget.
- **ONNX export parity.** Follow Foveacast's existing [`scripts/unisal-onnx-export.py`](https://github.com/khawkins98/Foveacast/blob/spike/unisal-onnx-research/scripts/unisal-onnx-export.py) pattern when Phase 8 lands: inline external data, validate PyTorch-vs-`onnxruntime` CPU outputs within float tolerance, target similar artefact size. `_tf1_bilinear_upsample` exports as `Gather + Mul + Add` rather than a single `Resize` — confirm `onnxruntime-web` handles this (expected to, all three are standard ops).

---

## Final note

If you are unsure whether something should be done, ask. The cost of asking is small; the cost of an un-done 12.9 GB download or a reversed architectural decision is not.
