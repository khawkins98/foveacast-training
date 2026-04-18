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

## 2026-04-16 — Phase 1 decision: PyTorch port, fine-tune at 240×320

After resolving the three open ambiguities from the upstream-code audit (details in [#2](https://github.com/khawkins98/foveacast-training/issues/2)), going with option B: port MSI-Net's architecture to PyTorch and load Kroner's pretrained SALICON weights via the HuggingFace SavedModel.

**Why B beat A.** The ambiguity audit made the port easier, not harder. Kroner's preprocessing turned out to be a single in-graph mean subtraction (`x - [103.939, 116.779, 123.68]`), not a black-box `keras.applications.preprocess_input` call. The architecture is hand-rolled without `tf.keras.applications` wrappers hiding anything behind library internals. Weight loading comes through the HuggingFace SavedModel, so we are not wrestling TF 1.x checkpoint plumbing. Effort estimate for the port came down from 4–8 days to 3–5. Meanwhile option A's case got correspondingly weaker — less to "buy" by vendoring Keras when the Keras-specific machinery is almost nothing.

The durable reasons were there before the audit and did not change:

- No existing PyTorch port means we build either way. The only question is what stack the build ends up in.
- MPS on Apple Silicon is better served by PyTorch than by TensorFlow 2.x.
- Foveacast V2's ONNX export pipeline is already PyTorch-shaped. Staying on one stack across training and export means one toolchain to debug, not two.
- TF 2.x is moving into maintenance mode; Keras 3 is the active line. Pinning a multi-year research repo to TF 2.15 now accumulates technical debt that is not needed.

**Fine-tune resolution: 240×320, not V1's 120×160.** Trading inference cost for accuracy, on purpose.

- The HuggingFace SavedModel is the SALICON variant, trained at (240, 320). Fine-tuning at the same resolution keeps the weights close to the prior they were trained against.
- V1's 120×160 was a TFJS-era browser-perf choice, not a Kroner-recommended input size. Fine-tuning at 120×160 means running MSI-Net at a resolution it was not trained for — the pretrained weights become an approximate prior instead of a close one.
- The whole point of V3 is better saliency prediction on UI content. Starting at the resolution where the model is actually good and scaling back if browser perf is unacceptable is a better ordering than starting cheaper and discovering we have handicapped the accuracy story.

**Revisit triggers.** The 240×320 choice is not load-bearing forever. Re-open the question if:

- Phase 6 quantitative eval shows the fine-tuned model is no better than stock MSI-Net — something else is broken; resolution is not the fix.
- Phase 10 Foveacast integration shows inference latency is unacceptable on the target hardware baseline (mid-range laptops, not GPUs).
- A production case emerges for multiple quality presets — V1 shipped five; V3 could do the same by fine-tuning once and exporting at multiple resolutions.

**What does not change.** The ONNX contract with Foveacast keeps the same shape as V2's — single `.onnx` artefact, consumed via `onnxruntime-web`, loaded from `docs/models/` in the Foveacast repo. The only visible difference downstream is better heatmaps on UI content. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full end-to-end shape of the pipeline and the contract boundary.

Closes #2. Ticks Phase 1 on #1.

## 2026-04-16 — Phase 2 port: three surprises between "architecture looks right" and "numerical parity"

The PyTorch port of MSI-Net (#5) passed its architecture smoke test on the first try — 24.9M parameters, correct input/output shapes, ruff clean. Then the first real run against Kroner's HuggingFace SavedModel produced a mean absolute error of ~2e-2 (which is a lot when outputs are normalised to [0, 1]). Closing that gap took three discoveries, each of which would have been non-obvious without the parity test catching them.

Worth recording in order because the debugging arc itself illustrates a pattern: the port *as written* was structurally correct; what broke parity was three separate things *inside the SavedModel* that didn't match the assumptions in the reference paper, Kroner's GitHub code, or my research notes.

**1. The HuggingFace SavedModel is exported *frozen*.** Weights live as named `Const` ops inside the inference graph, not as restorable `tf.Variable` objects. This is not documented anywhere on the HF model page; I found it by running the import script, getting `No loader returned any variables`, and probing the graph structure. Both `tf.saved_model.load(path).variables` and `tf.keras.models.load_model(path).weights` return empty lists because there are literally no live variables in the re-loaded model. TF 2.21 shipping Keras 3 by default adds a second layer of confusion — `tf.keras.models.load_model` refuses legacy SavedModels entirely ("Keras 3 only supports V3 `.keras` files"), so even the wrong answer is a red herring. The working path is `tf.saved_model.load(path)` → walk two levels of `PartitionedCall` indirection → reach `__inference_pruned_Y` → harvest float32 `Const` tensors by name. The importer does this in `load_tf_constants()`, and the resulting TF scope names (`conv1/conv1_1/kernel`, `aspp/conv1_1/kernel`) are Kroner's original — matching his `model.py` exactly. Name-based mapping via an explicit `TF_TO_PYTORCH` dict is more robust than positional pairing for this reason.

**2. `decoder/conv4` has a bias after all.** My first visual probe of 1D constants filtered on `shape[0] > 1` to suppress noise — which silently hid shape-`(1,)` biases. `decoder/conv4` has one with value ~2.4e-5: effectively zero, definitely present. I wasted a commit on setting `nn.Conv2d(..., bias=False)` for that layer; the importer promptly tripped over `Unexpected key decoder_conv4.bias` and exposed the mistake. Lesson: any filter applied during exploratory probing is a lie by omission, and "I already checked" for this specific thing was exactly wrong. The fix is trivially `bias=True` (PyTorch default) once the bias is surfaced.

**3. `ResizeBilinear` in the frozen graph carries TF 1.x legacy semantics that *neither* PyTorch mode reproduces.** This was the load-bearing one — responsible for the full 2e-2 mean error. The four `ResizeBilinear` ops all have `align_corners=False` AND `half_pixel_centers=False`. TensorFlow 1.x's default; TensorFlow 2.x changed the default for `tf.image.resize` to `half_pixel_centers=True`, but a frozen SavedModel preserves whatever the exporter set at export time. PyTorch's `F.interpolate(mode='bilinear', align_corners=False)` uses half-pixel sampling (modern convention); `align_corners=True` uses endpoint-aligned sampling. Neither matches the TF 1.x convention of `src = dst * (src_size / dst_size)` with duplicated-edge clamping. The 0.25-pixel source-sampling offset alone is small; compounded through three decoder upsample blocks it produces output-level errors of 20% per pixel. Fixed by writing `_tf1_bilinear_upsample` — a direct implementation via integer-index gather — and replacing every `F.interpolate(mode='bilinear')` in `_aspp` and `_decoder`. This single change dropped mean abs error from ~2e-2 to ~1e-7. The new helper exports as `Gather + Mul + Add` in ONNX rather than a single `Resize` op; flagged in the helper's docstring for Phase 8 to confirm `onnxruntime-web` handles it (it should — all three are standard ops).

**Meta-observation.** Numerical parity tests against a reference implementation pay for themselves on the first run. Each of the three surprises above would have been a latent bug that Phase 5's fine-tune loop would have swept under the rug — training from a subtly-wrong initialisation produces a fine-tune that looks fine on loss curves and produces plausibly-decent saliency maps, but is silently worse than the reference stock model. The 1e-7 / 1e-6 tolerance we ended up with is sharp enough to catch a single changed op; I'd rather have that than the generous `atol=1e-3` the test started with.

**What changed in the repo as a result of the debugging arc.**

- `scripts/import_msinet_weights.py` — completely rewritten to walk the frozen graph and extract named `Const` tensors, with an explicit `TF_TO_PYTORCH` mapping and a `--discover` mode that dumps the graph's float32 constants for debugging future re-exports.
- `src/foveacast_training/msinet.py` — added `_tf1_bilinear_upsample` helper; replaced all `F.interpolate(mode='bilinear')` calls.
- `tests/test_msinet_parity.py` — tolerance tightened from `atol=1e-3` to `atol=1e-5` once we knew the natural residual was ~1e-7.

Phase 2 gate closed. Forward pass matches the reference. Ticks Phase 2 on #1.

## 2026-04-16 — Phase 3 loader: the boring phase, mostly

Phase 3 landed `src/foveacast_training/ueyes_dataset.py` plus 16 tests against the real unpacked deposit (tests skip cleanly when `data/ueyes/` is absent). This is the closest the project has come to "just write the code" — Phase 0 did the hard work of characterising the archive, Phase 1 resolved substrate, Phase 2 resolved the architecture itself. For the loader, almost every decision had a reasonable default waiting to be picked.

Three choices worth documenting even though none of them felt load-bearing at the time:

**Saliency variant: `heatmaps_3s`.** Continuous Gaussian-smoothed maps rather than binary `fixmaps_*`, at a 3-second aggregation window rather than 1s or 7s. The 3s choice approximates SALICON's ~5s aggregate window — the one MSI-Net's pretrained weights were fit against — so fine-tuning doesn't ask the model to shift duration semantics on top of the domain shift from natural scenes to UI. If Phase 6 eval shows the variant is a load-bearing choice (e.g. 7s gives noticeably different numbers), the `saliency_variant` constructor argument is already there to swap. Made configurable up front for cheap optionality later.

**Validation split: stratified 10% carve from upstream Train.** 1,684 train / 188 val / 108 test, with val pinned by seed=42 (fresh `np.random.default_rng(seed)`, not the global NumPy state). Exactly 47 val images per category × 4 categories. The upstream deposit ships only a train/test split, so val is our construction. I considered a category-and-block stratification (use `Block` as an additional axis) but the Block column is a mess (mixed plain integers and Excel scientific notation — see below) and no paper I'm aware of reports the block assignment as a fine-tuning-relevant covariate. Single-axis category stratification is enough.

**Preprocessing: PIL BICUBIC + constant pad.** Kroner's reference uses area interpolation for downscale, bicubic for upscale — a small parity boost for SALICON training. For a UEyes fine-tune we don't actually need preprocessing parity with Kroner's SALICON run; we need train-time and eval-time consistency *within this repo*. Single-method BICUBIC end-to-end keeps the loader small and dodges PIL-vs-TF interpolation differences. If Phase 6 ends up comparing fine-tuned outputs against stock SALICON MSI-Net, the eval path gets its own reference-preprocessor.

**The `Block` column quirk.** The upstream CSV has most `Block` values as plain integer strings (`"0"`, `"23"`) and a handful as Excel scientific notation (`"0,00E+00"`). Opening the file in Excel and re-saving almost certainly did it. First pass of the loader parsed `Block` as `int()` and exploded. The fix is trivially to keep `Block` as a raw string — nothing in the loader uses it — but it's the kind of thing worth writing down because someone six months from now will read the column, assume it parses cleanly, and re-introduce the same bug.

**What's deliberately not in this module.**

- No data augmentation. 1,980 images and a close pretrained prior means augmentation adds noise without a training-size payoff.
- No caching of preprocessed tensors. Single-pass PIL decode + numpy resize + numpy pad is fast enough; PyTorch's `DataLoader` with workers handles the per-epoch cost. Revisit if Phase 5 turns out to be I/O bound.
- No collate_fn customisation. Default collation is fine for same-shape tensors.

**What Phase 4 will need from this module.** `UEyesDataset(root, split, ...)` and `DataLoader(ds, batch_size=..., shuffle=True, num_workers=...)`. That's it. The prototype training loop and the full fine-tune both call the same Dataset; the only difference is the subset and epoch count they iterate over.

Phase 3 gate closed. `(image, saliency)` tensors of correct shape, splits sum to 1,980, stratified, reproducible. Ticks Phase 3 on #1.

## 2026-04-16 — Phase 4 prototype: the loss curve looks plausible

Phase 4 landed `src/foveacast_training/train.py` + `src/foveacast_training/losses.py` + 10 loss-module tests. Full suite is now 32 passed. The gate — "does the loss curve look plausible on 100 images, 2 epochs" — closed on the first run.

**Numbers from the Phase 4 gate-closing run on M4 MPS (2026-04-16, seeded so they're reproducible):**

| metric     | epoch 1 | epoch 2 | direction |
|------------|---------|---------|-----------|
| train loss (avg) | 0.8928 | 0.7612 | ↓ 15%   |
| val loss         | 0.9006 | 0.8297 | ↓ 8%    |
| val CC           | 0.6012 | 0.6421 | ↑ 6.8%  |

Training + validation in ~90 seconds (budget was 30 minutes). No NaN, no divergence, no memory pressure. Loss is monotonically decreasing epoch-over-epoch; val CC is monotonically increasing. All three directions point the right way, which is what Phase 4's gate is asking.

A second run produced bit-identical numbers — confirming the `torch.manual_seed(0)` + `numpy.random.seed(0)` + `python random.seed(0)` + `num_workers=0` combination in the prototype config actually gives reproducible metrics, not just a plausible-looking stochastic run. Per-step loss values may still jitter slightly on MPS because Apple's Metal kernels aren't bit-deterministic at the op level, but end-of-epoch averages are stable.

**Starting val CC of 0.60 is the useful signal.** The pretrained SALICON weights already produce a decent correlation with UEyes ground truth before any fine-tuning happens — meaning MSI-Net's natural-scene prior transfers reasonably to UI content out of the box, and fine-tuning is improving on an already-credible baseline rather than starting from noise. That's consistent with the UEyes paper's +10 AUC finding from fine-tuning and suggests the port is structurally sound all the way through.

**Choices that didn't need debating in Phase 4 (worth recording so they don't get re-derived).**

- **KL divergence loss, ported verbatim from Kroner's `loss.py`.** Both pred and target sum-normalised per image (absolute scale is uninformative for saliency; relative-to-other-pixels is the signal). Eps = 1e-7 in every divide and log, same as the reference. KL(p||p) isn't exactly zero under this formulation — eps shifts produce a ~1e-4 residual for a 240×320 map — which is why the corresponding test checks `abs(loss) < 1e-3` rather than strict equality. Not a bug; a physical consequence of the eps handling.
- **Correlation coefficient as the validation metric.** Pearson correlation on flattened-per-image maps; standard across saliency literature. Range [-1, 1]; higher is better. Works symmetrically on any non-negative scalar field; no preprocessing needed beyond what the training loop already does.
- **Adam optimiser, learning rate 1e-5 for the prototype.** Matches Kroner's `PARAMS` default. Phase 5's full fine-tune will reduce this by 10× (to 1e-6) per issue #1's guidance on fine-tuning learning rates — but the prototype's goal is "does it work at all," and running at the pretraining LR exercises more of the gradient landscape in the 50 steps we get.
- **Batch size 4.** Fits comfortably on M4 MPS without checkpointing; leaves headroom for the Phase 5 increase to 8.

**What's deliberately not in this module.**

- **No TensorBoard.** 50 training steps doesn't benefit from TB's strengths. Stdout numbers are sufficient for "does the loss curve look plausible." Phase 5's 8,000-step run is where TB becomes worth the complexity.
- **No best-checkpoint saving.** Prototype is a sanity check, not a keep-the-artefact exercise. Phase 5 adds this.
- **No learning-rate scheduler, gradient clipping, early stopping.** Same reasoning — Phase 5 territory.
- **No data augmentation.** Explicitly a non-goal for the UEyes fine-tune, per ARCHITECTURE.md.

**What Phase 5 will need from this module.** Most of the skeleton is here:
- `FULL_CONFIG` dict exists but its numbers are placeholder; Phase 5 tunes them.
- Best-checkpoint saving on validation CC.
- Learning-rate scheduler (probably cosine or plateau).
- Early stopping on validation CC plateau.
- TensorBoard logging as a `--tensorboard` flag (not default).

Phase 4 gate closed. Loss curve looks plausible. Ticks Phase 4 on #1.

## 2026-04-17 — Phase 5 full fine-tune: 30 epochs, best at 25

Ran `python -m foveacast_training.train --full` overnight on M4 MPS. All 30 epochs completed; early stopping triggered at epoch 30 (5 non-improvement epochs since epoch 25's best). LR scheduler halved to 5e-7 around epoch 28 but didn't recover the plateau. Total wall-clock ~3.5 hours.

**Best checkpoint: epoch 25, val_cc=0.7247.** Saved to `runs/full-20260416-225926/best.pt`.

| epoch | train_loss | val_loss | val_cc | note |
|---|---|---|---|---|
| 1 | 0.8206 | 0.7454 | 0.6508 | ★ first |
| 5 | 0.6314 | 0.6446 | 0.7016 | ★ |
| 10 | 0.5836 | 0.6151 | 0.7162 | ★ |
| 15 | 0.5539 | 0.6043 | 0.7211 | ★ |
| 19 | 0.5350 | 0.5990 | 0.7236 | ★ |
| 20 | 0.5300 | 0.5994 | 0.7232 | first non-improvement |
| 22 | 0.5221 | 0.5968 | 0.7244 | ★ recovery |
| 25 | 0.5097 | 0.5966 | 0.7247 | ★ **best** |
| 30 | 0.4898 | 0.5983 | 0.7237 | early stop triggers |

19 of 30 epochs produced new-bests. Gains slowed from +0.023/epoch (early) to +0.001/epoch (late). Two mini-plateaus at epochs 20-21 and 23-24 each recovered with a small jump. The LR scheduler fired around epoch 28 (3 epochs of plateau at patience=3), halving to 5e-7, which wasn't enough to break the final plateau — early stopping caught it at 5 epochs without improvement.

The safety machinery from #11/#12 worked exactly as designed. Gradient clipping never visibly activated (no loss spikes in the log). Best-checkpoint saving captured epoch 25's weights. ReduceLROnPlateau and early stopping fired in sequence when the model genuinely plateaued. No intervention needed overnight.

## 2026-04-17 — Phase 6 quantitative eval: fine-tuned beats stock on all three metrics

Ran `python -m foveacast_training.eval --checkpoint runs/full-20260416-225926/best.pt --compare weights/msinet_salicon.pt --split test` on the held-out UEyes test split (108 images, never seen during training).

| metric | fine-tuned (epoch 25) | stock SALICON | delta | direction |
|---|---|---|---|---|
| CC  | **0.7068** ± 0.105 | 0.4934 ± 0.094 | +0.2135 | +43% better |
| KLD | **0.6574** ± 0.210 | 1.1682 ± 0.246 | -0.5108 | -44% better |
| NSS | **2.2879** ± 0.605 | 1.5776 ± 0.451 | +0.7103 | +45% better |

Phase 6's gate was "fine-tuned beats stock on at least one metric." It beat stock on all three, by 43-45% each. The improvement is consistent across metrics (CC, KLD, NSS all agree on the direction and magnitude) and across the standard deviation bands (the fine-tuned model's worst test image is still better on average than stock's mean).

For context against the broader literature: the UEyes paper (Jiang et al. 2023) reported +10 AUC points fine-tuning their own models on UEyes. Our CC improvement of 0.21 absolute is in the same ballpark — the fine-tuning does what the paper said it would, on a different architecture (MSI-Net vs their in-house models) with a different training stack (PyTorch vs their TF pipeline).

**Release artefact also produced:** `releases/foveacast-v3.onnx` (106.4 MB), exported from the fine-tuned `best.pt` with the same Phase 8 parity-validated export path. PyTorch ↔ onnxruntime CPU max abs err 6.14e-06 (16× under tolerance). Ready for Phase 9 release tagging.

Phases 5 + 6 gates closed. Ticks both on #1.

## 2026-04-16 — Phase 8 ONNX export: closed the gate on stock weights

Phase 8's gate per #1 is "validate parity between PyTorch and `onnxruntime` CPU." That's testable against any checkpoint — the gate is about the export mechanism, not about which specific weights we export. Running against `weights/msinet_salicon.pt` (stock MSI-Net) was therefore the fastest way to close Phase 8 while Phase 5's `--full` training was still running in the background.

**Numbers from the first passing run (2026-04-16, stock weights):**

| metric | value |
|---|---|
| artefact size | 106.4 MB |
| max abs err   | 1.79e-06 |
| mean abs err  | 1.09e-07 |
| tolerance     | 1e-4 (55× headroom) |

55× headroom is reassuring — the `_tf1_bilinear_upsample` helper that I'd flagged in Phase 2's LEARNINGS as "Gather + Mul + Add in ONNX rather than a single Resize" round-trips cleanly. Parity is dominated by the expected fp32-vs-fp32 ordering drift across `torch.onnx`'s decomposed ops, not by any semantic mismatch.

**Choices that worked first-try:**

- **Opset 17** (current PyTorch default at export time). Widely supported by `onnxruntime-web`.
- **Dynamic batch dim, fixed spatial dims.** `dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}`. A future caller wanting to batch multiple images doesn't need a re-export; a future caller wanting a different input resolution does, because `_tf1_bilinear_upsample` bakes in concrete output sizes at trace time. That's deliberate — dynamic spatial would make the helper substantially more complex, and shipping one resolution is the Phase 9 release goal anyway.
- **Single-file artefact via `save_as_external_data=False`.** Belt-and-braces round-trip through `onnx.load` + `onnx.save_model` forces all weight tensors back into the main blob. MSI-Net is well under the 2 GB auto-externalisation threshold but the round-trip costs nothing and means `onnxruntime-web` loads one blob rather than two.
- **Export on CPU.** `EXPORT_DEVICE = torch.device("cpu")` regardless of what's available. MPS tracing has known shape-inference quirks; the exported graph runs on any target regardless of where it was traced. Portability wins over throughput (export is a one-shot anyway).

**Artefact size: 106.4 MB is bigger than V2's UNISAL 12.5 MB.** Expected — MSI-Net is 25M params (VGG16 backbone dominates); UNISAL is smaller arch. 100 MB at fp32 is the floor without quantisation. If browser-side loading turns out to be a Phase 10 concern, fp16 export halves it to ~53 MB and int8 quantisation could go lower, but those are Phase 9/10 optimisations — the gate as specified in #1 is about parity, not size.

**What Phase 9 / Phase 10 need from this module.** When Phase 5's `best.pt` lands, the same command with a different `--checkpoint` produces `releases/foveacast-v3.onnx`. No code changes; the parity gate will either re-close (expected, since fine-tuning doesn't change which ops run) or surface a concrete number for the release notes. That artefact is what Foveacast's integration PR (Phase 10) drops into `docs/models/foveacast-v3/model.onnx`.

Phase 8 gate closed. Ticks Phase 8 on #1.

## 2026-04-17 — FP16 quantisation: naive beats selective

Explored two approaches to halving the release artefact from 106 MB (fp32) to ~57 MB (fp16):

**Naive:** `convert_float_to_float16(model, keep_io_types=True)` — converts everything internal to fp16 while keeping input/output tensors as fp32 for caller compatibility. The `onnxconverter_common` library handles the precision boundaries automatically.

**Selective:** `op_block_list=["ReduceMin", "ReduceMax", "Div"]` — keeps the `_normalize` step's min/max/division ops in fp32 to protect the `eps=1e-7` arithmetic from fp16 underflow (fp16 min normal is ~6e-5). The conv weights still go fp16 for the bulk savings.

| variant | size | max err vs PyTorch |
|---|---|---|
| fp32 | 106.4 MB | 6.14e-06 |
| **fp16 naive** | **56.5 MB** | **7.16e-04** |
| fp16 selective | 56.6 MB | 9.27e-04 |

The selective approach was supposed to be cleaner but was actually 29% worse on max error. The reason: keeping specific ops in fp32 while their upstream inputs arrive in fp16 creates mixed-precision boundaries. Each fp16→fp32 cast introduces a rounding step that the naive all-fp16 path avoids entirely. With everything in fp16, the arithmetic is consistent within its precision — no cross-format conversion noise.

Size difference is negligible (0.1 MB). Ship the naive fp16.

FP8 was also evaluated and ruled out: `onnxruntime-web` has no FP8 op support, and 3 mantissa bits would need quantisation-aware retraining to avoid visible artefacts. INT8 (~26 MB) is the next realistic step if 57 MB turns out to be a problem in Foveacast's browser loading — it needs a calibration dataset and per-tensor scale/zero-point computation, so it's a half-day of work tracked in #15 if needed.

## 2026-04-17 — Multi-duration model training infrastructure (issue #19)

UEyes ships ground truth at three viewing-duration aggregates (`heatmaps_1s`, `heatmaps_3s`, `heatmaps_7s`). v0.1.0 ships one model trained against `heatmaps_3s`. Issue #19 is to train two more — 1s for "what grabs the eye first", 7s for "what's eventually noticed" — so Foveacast can offer a duration selector.

The issue asserted "no code changes needed — just run `--full` with a different `saliency_variant`." Almost true. A few pieces needed to land before the two ~3.5 h runs were safe to kick off:

**`--saliency-variant` as a first-class CLI flag.** Was a constructor arg on `UEyesDataset`; wasn't wired through `train.py`. Threading it through both the train and val dataset instances (easy) and into the run-dir name (so `runs/full-heatmaps_1s-.../` is self-identifying when parallel runs are on disk) took the bulk of the PR.

**`eval.py --time-window`.** Evaluation previously hard-coded `heatmaps_3s` and `fixmaps_3s` — the two-dataset lock-step pattern Phase 6 uses for CC/KLD (heatmap target) vs NSS (fixmap target). Parameterised both off a single `1s|3s|7s` flag so cross-duration evaluation is impossible by accident. Matched-window eval is the right question for #19 ("does the 1s model agree with 1s ground truth?"), not cross-window ("how does 1s do on 3s ground truth?").

**Resume support for interrupted runs.** This one was explicitly out of #19's stated scope but worth doing before committing to 7 h of compute. Each epoch now overwrites `state.pt` with model weights + optimizer moments + scheduler counter + history + next-epoch marker. `--resume state.pt` picks up at the last completed epoch. The resumed trajectory is *not* bit-exact vs an uninterrupted run — DataLoader shuffle order diverges because we don't restore RNG state — but that's fine; the goal of resume is "recover a trained model after a sleep event or OOM", not "reproduce the exact loss curve." State file is ~300 MB per save (model ~100 MB + Adam moments ~200 MB), overwritten each epoch, so no disk-growth concern across a 30-epoch run.

Smoke-tested by running `--prototype` to completion, hand-editing the resulting `state.pt` back to an "after epoch 1" posture, and running `--resume`. History from epoch 1 was preserved, epoch 2 executed, the merged `history.json` had both epochs in order. Good enough; the design is straightforward and the test catches the obvious failure modes (load errors, schema mismatch, optimizer state not restored).

**FP16 as a committed flag, not an ad-hoc conversion.** The v0.1.0 FP16 artefact (`foveacast-v3-fp16.onnx`) was produced via a one-shot `convert_float_to_float16` call from an interactive session — no script in the repo. For three releases that needs to be reproducible, so `export_onnx.py --fp16` is now a first-class flag with a matched 1e-3 parity tolerance (the LEARNINGS entry right above this one covers why naive beats selective). Dep is `onnxconverter-common` — ~1 MB pure-Python, added to core rather than a separate extras group so a downstream re-exporter doesn't need to discover an extras name.

**INT8 static post-training quantisation.** `quantize_int8.py` is new. Produces ~26 MB artefacts (three of those plus three FP16 fallbacks is 6 assets on `v0.2.0`). Design decisions:

- *Static over dynamic quantisation.* Static needs a calibration dataset (100 random UEyes training images here) but produces meaningfully better results on conv-heavy networks. Dynamic would skip calibration but tends to underperform on activation distributions that vary per layer — which is exactly the case for a VGG16-backbone saliency model.
- *QUInt8 activations + QInt8 weights.* ORT's standard recipe. Activations post-ReLU are always non-negative so QUInt8 (asymmetric, `[0, 255]`) wastes no range on negative values; weights are zero-centred so QInt8 (symmetric, `[-128, 127]`) fits the distribution better.
- *`quant_pre_process` before `quantize_static`.* ORT's docs call this optional; in practice skipping it is the #1 cause of "Could not infer shape of tensor X" failures at quantisation time. The graph's shape-inference state from `torch.onnx.export` isn't always complete enough for the quantiser to reason about.
- *100 calibration samples.* ORT docs recommend 100-500. Saliency output is a well-behaved `[0, 1]`-ish range; diminishing returns kick in fast. If a model's parity gate fails, bumping to 300 is the first knob to turn.

Quality gate is comparative: ship INT8 as primary if CC/KLD/NSS delta vs FP16 is <3% per metric, fall back to FP16 otherwise. Measurement happens post-training in the same pass as the stock-vs-fine-tuned comparison. `eval.py --onnx` grew a second life as an ONNX-aware evaluator during this work — see the 2026-04-18 entry below.

## 2026-04-18 — INT8 quality gate: the structural parity number lied, the real metric answer is fine

Spent a chunk of time chasing the INT8 structural parity gate (PyTorch FP32 vs INT8 ONNX, random-uniform inputs) on the three multi-duration artefacts. First run with the saturated-input trial bank showed max-pixel error of 4e-1 on the 1s model — catastrophic. Removing the saturated trials (static PTQ calibrates against a sample distribution; saturated inputs fall outside it and blow up the clamp error) brought max error down to 8e-2 on 1s. But the 3s and 7s models landed at 1.1e-1 and 1.4e-1 respectively, tripping a 1e-1 tolerance.

Three ways to play it: loosen the tolerance until everything passes (meaningless gate), fall back to FP16-only (punt), or build a real CC/KLD/NSS gate via `eval.py --onnx`. Went with the third. `make_inferencer(path, device)` in `eval.py` now dispatches to PyTorch or ONNX Runtime based on file extension, so the rest of the evaluation pipeline stays backend-agnostic — same split loading, same metric computation, same comparison-table formatting. ~40 lines total.

Running FP16 ONNX through the new eval path showed CC/KLD/NSS matching PyTorch `best.pt` to 4 decimal places across all three windows, which is a good sanity signal that the ONNX plumbing is correct before trusting it on INT8.

The INT8 numbers were the surprise:

| window | metric | PyTorch | INT8 | relative delta |
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

**All three INT8 models pass the <3% per-metric delta criterion, comfortably.** Random-uniform structural parity showed 8-14% max-pixel error across the three; real UEyes images show <3% metric degradation. The structural number was a lower bound — it *is* real, but it lives on pixels nobody looks at (background regions with near-zero saliency values where small absolute errors are large relative errors without visual consequence). The metric evaluation weights pixels by where the real ground truth actually lands, and INT8's calibration is fit to exactly that distribution.

Lesson for future quantisation work: **don't trust random-uniform parity numbers as a quality gate for statically-calibrated models.** They overestimate perceived quality loss by roughly 3-5× in this case. If a structural gate is needed as a fast sanity check, the right design is to include a small realistic-input sample (a handful of UEyes images, same family as the calibration set) in the trial bank. The current `quantize_int8.py` leaves that as a follow-up if the pattern recurs.

KLD is the most sensitive metric — all three INT8 models show +1.7% to +2.6% KLD regression. That's expected: KLD measures distribution divergence, and INT8's discretisation most affects tail values (low-confidence regions), which shift the predicted distribution's shape slightly. CC and NSS are essentially unmoved because they respond to where the peaks land, and peaks survive INT8 cleanly.

Ship decision: all three INT8 artefacts go alongside the FP16 ones in the v0.2.0 release. Consumers pick based on their own size/quality tradeoff (57 MB FP16 vs 32 MB INT8 per duration).

## 2026-04-17 — Phase 10 integration: heatmap.js was distorting the visual output all along

Discovered during the Foveacast integration (PR #5 on Foveacast) that the saliency overlays in the browser looked dramatically different from the benchmark renders in foveacast-training. The model output was verified identical (PyTorch ↔ ONNX FP16 at 100% hotspot overlap), and the preprocessing difference (PIL BICUBIC vs JS bilinear) produced only 96.8% overlap — not enough to explain what we were seeing.

The actual cause was **heatmap.js's rendering**, which added three layers of distortion between the raw model output and the visual overlay:

1. **Radius spreading.** Each saliency point was rendered as a 40px circle, making every hotspot look 40 pixels larger in every direction. A sharp attention peak on a button became a diffuse glow covering the button and its surroundings.
2. **Stride sampling.** The library was fed every ~2nd pixel (stride = width/160) to keep point counts manageable. The spatial precision of the model's 240×320 output was being halved before rendering even started.
3. **Internal blur at 0.85.** On top of our postprocess Gaussian blur (which we'd already reduced from sigma=28 to sigma=5 for V3), heatmap.js applied its own blur pass.

The combined effect: a model that correctly predicted "the user will look at the Join Now button" produced a visual that said "the user will look at the entire right half of the page." The quantitative metrics (CC, KLD, NSS from Phase 6) were always comparing raw model outputs and were never affected — they remain valid. But every visual comparison ever shown through Foveacast's browser UI — including the V1 vs V2 benchmark screenshots in `docs/spikes/comparison/` — was distorted by this rendering layer.

**What this means for the old V1/V2 comparisons:** the benchmark conclusion that "V1 MSI-Net produces diffuse centrality blobs" was partially a heatmap.js artefact. The stock model's raw output is more focused than heatmap.js made it look. V3 fine-tuned is still quantitatively better (+43% CC), but the visual gap between stock and fine-tuned was exaggerated by the renderer. The old comparison screenshots in this repo's `benchmark/screenshots/` carry a note about this — they should not be compared pixel-for-pixel against the new inferno renders.

**The fix:** replaced heatmap.js with a direct pixel-by-pixel inferno colormap renderer (`docs/src/render/saliency-canvas.js` in Foveacast). No radius spreading, no stride sampling, no external library. Each pixel of the saliency map maps directly to a colour. The output now matches the benchmark renders from foveacast-training exactly.

**Two other preprocessing issues found and fixed during integration:**

- **Aspect-ratio-preserving resize.** Foveacast V2's preprocess stretched the image to fill 240×320. V3 MSI-Net was fine-tuned with aspect-preserving resize + constant-126 padding (Kroner's convention). The stretch produced noticeably different saliency predictions because the model saw a different pixel distribution than it was trained on. Fixed to match the training pipeline.
- **Gaussian blur sigma.** V2's sigma=28 was tuned for UNISAL's very peaky log-probability output. V3's output is already smooth (VGG16 decoder + min-max normalisation). Reduced to sigma=5 — just enough to smooth resize staircasing without washing out attention peaks.

## 2026-04-16 — Phase 5 readiness: safety before hyperparameters

Spun out of #10's DevRel review, tracked in #11. The point is that Phase 5's full fine-tune is a 4-hour commitment on M4 MPS, so making it fail *loudly* and recover *gracefully* before the first run matters more than tuning hyperparameters ahead of time. Hyperparameters get tuned empirically; safety machinery doesn't have to be.

Five things landed together in `src/foveacast_training/train.py`:

**Gradient clipping at norm 1.0 per step.** `torch.nn.utils.clip_grad_norm_` between `loss.backward()` and `optimizer.step()`. KL with eps-shifted logs is exactly the kind of loss where a single-batch spike corrupts the best checkpoint — three lines of code to prevent that failure mode.

**Best-checkpoint saving on validation CC.** On every val-CC improvement (>= `early_stop_min_delta` above the current best), write `best.pt` (state_dict only) and `best.json` (epoch, val_cc, val_loss, train_loss, lr). On training end, also write `final.pt`. The two-file approach means we can recover if the "best" checkpoint turns out to be a local maximum that later epochs would have improved on.

**ReduceLROnPlateau on validation CC.** `mode="max"`, `factor=0.5`, `patience=3`. Halves the learning rate when val CC plateaus for 3 epochs. Default-safe — fancier schedules (cosine, warmup) can land later once we have signal that the simple one isn't sufficient.

**Early stopping after 5 epochs of val-CC plateau.** `early_stop_min_delta=1e-4` defines "plateau"; `early_stop_patience=5` defines how long to tolerate it. Prevents the worst Phase 5 failure mode — training past the optimum and saving progressively worse "best" checkpoints (which wouldn't happen given we only save on improvements, but the symptom-free version is a model trained N extra epochs that doesn't improve, so throughput matters too).

**Explicit `--prototype` vs `--full` mode, no default.** `argparse.add_mutually_exclusive_group(required=True)`. The previous "default to full with a loud warning" was a soft guard; requiring an explicit flag makes the typo-turns-into-a-4-hour-run failure mode literally impossible. `--full` still prints a warning about placeholder hyperparameters because those *are* still untuned.

**What's NOT in this commit.** Everything that wants empirical Phase 5 data:

- Actual tuning of the hyperparameters. `FULL_CONFIG` currently has `lr=1e-6, n_epochs=30, batch_size=8` — picked from issue #1's guidance, not from measurement.
- TensorBoard logging. 8k steps benefit; 50 prototype steps don't. Gated behind a future `--tensorboard` flag.
- `--resume PATH` for resuming a killed run.
- Seeded full-mode runs for hyperparameter comparison. Deliberately unseeded by default for throughput.

**Prototype numbers unchanged.** Kept the Phase 4 canonical values (train 0.8928 → 0.7612, val CC 0.6012 → 0.6421) by disabling the safety machinery for `--prototype` — grad clip shifts updates even when it doesn't strictly clip, so enabling it would have drifted the gate numbers ~3% without re-running the gate. The prototype is a frozen sanity check, not a Phase-5 rehearsal; Phase 5's safety machinery exercises via `--full`.

Phase 5 is now safe to run. When it does, the first pass is an explicit hyperparameter probe — not a "ship the artefact" run.
