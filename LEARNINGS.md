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
