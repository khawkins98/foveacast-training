# Dataset — UEyes

This folder is where the UEyes dataset goes after you fetch it. The data itself is not committed to this repo; at 12.9 GB zipped it belongs on Zenodo, not in git.

## Fetching

The UEyes dataset is hosted on Zenodo at [record 8010312](https://zenodo.org/records/8010312) under Creative Commons Attribution 4.0 International. A single `UEyes_dataset.zip` file.

Use the helper script from the repo root:

```sh
bash data/fetch.sh
```

The script downloads the zip into `data/ueyes/`, verifies the archive with `unzip -t`, extracts in place, and prints a depth-2 directory listing you can paste back if `data/README.md` needs updating. It is idempotent and resumable — a partial download picks up from where it stopped; a completed extraction is a no-op.

Useful environment variables:

- `FETCH_SKIP_UNZIP=1` — download only, leave the zip on disk. Handy if you want to archive the raw deposit somewhere before extracting.
- `FETCH_DEST=/some/path` — extract somewhere other than `data/ueyes/` (useful if your repo lives on a small SSD and the dataset lives on external storage).

Prerequisites: `curl` and `unzip` on `PATH`. Both ship with macOS and every mainstream Linux.

If you would rather fetch by hand — network restrictions, a download manager, whatever — the equivalent manual steps are:

```sh
mkdir -p data/ueyes
cd data/ueyes
curl -L -C - -o UEyes_dataset.zip \
  "https://zenodo.org/records/8010312/files/UEyes_dataset.zip?download=1"
unzip UEyes_dataset.zip
```

## What's in the dataset

From the Zenodo record and the [CHI 2023 paper](https://doi.org/10.1145/3544548.3581096):

- 1,980 UI screenshots spanning four types: webpage, desktop UI, mobile UI, and poster.
- Eye-tracking data from 62 participants.
- Ground-truth saliency maps derived from participant fixations.

## Verified directory layout

Observed on 2026-04-16 after unpacking `UEyes_dataset.zip` (Zenodo record 8010312, sha of the deposit as shipped on that date). `UEyes_dataset/` is the single top-level folder; `__MACOSX/` and `.DS_Store` next to it are macOS zip cruft and are safe to delete (they are inside `data/ueyes/` which is gitignored anyway).

```
UEyes_dataset/
├── README.md                         # upstream-authored; refers to info.csv — see note below
├── image_types.csv                   # 1,980 rows, CRLF line endings, ';' delimiter
├── images/                           # 1,980 UI screenshots, mixed .png/.jpg/.jpeg
├── eyetracker_logs/                  # 554 raw Gazepoint fixation CSVs
├── saliency_maps/
│   ├── fixmaps_1s/                   # 1,980 — binary fixation maps at 1s duration
│   ├── fixmaps_3s/                   # 1,980 — …at 3s
│   ├── fixmaps_7s/                   # 1,980 — …at 7s
│   ├── heatmaps_1s/                  # 1,980 — Gaussian-smoothed saliency heatmaps
│   ├── heatmaps_3s/                  # 1,980
│   ├── heatmaps_7s/                  # 1,980
│   ├── overlay_heatmaps_1s/          # 1,980 — heatmap composited over the source image;
│   ├── overlay_heatmaps_3s/          #          filenames prefixed 'overlay_' (irregular!)
│   └── overlay_heatmaps_7s/          # 1,980
└── scanpaths/
    ├── paths_1s/                     # 1,980 per-image subfolders
    ├── paths_3s/                     #   each contains N.png for participant N
    └── paths_7s/                     #   (3–24 participants per image)
```

### Counts and shapes

- **1,980 source images** across four equally-sized categories:

  | Category | Count |
  |----------|------:|
  | desktop  | 495   |
  | mobile   | 495   |
  | poster   | 495   |
  | web      | 495   |

  Note the category label is `web`, not `webpage` as the paper's abstract phrases it.

- **File formats are mixed in all ground-truth folders:** 1,283 `.png`, 695 `.jpg`, 2 `.jpeg`. Saliency-map filenames exactly mirror the source image filenames — an image `foo.png` has `saliency_maps/heatmaps_1s/foo.png`, `saliency_maps/fixmaps_1s/foo.png`, etc. **Exception:** the `overlay_heatmaps_*` subfolders prefix every filename with `overlay_`, so `foo.png` there is `overlay_foo.png`. The Dataset loader needs to know this.
- **Image dimensions are wildly heterogeneous.** Observed range 237×260 up to 5,636×5,130; mean around 1,089×1,104. Aspect ratios are all over the place. A tiny number of images load as `mode=L` (grayscale) rather than `RGB` — the loader has to coerce.
- **Per-image participant coverage in `scanpaths/` ranges 3–24** with mode 7–12. This is long-tailed — one pass of eye-tracking per participant per image, attrition accounts for the low end, extra blocks for the high end.
- **`eyetracker_logs/` contains 554 raw Gazepoint fixation CSVs** named `{BB}_kh{PPP}_fixations.csv` (block × participant). The column format is documented upstream at https://www.gazept.com/dl/Gazepoint_API_v2.0.pdf — we probably don't need these for MSI-Net fine-tuning (the aggregated `saliency_maps/` is the training target), but they're available if Phase 5 or a later phase wants finer-grained supervision.

### Train / test split

`image_types.csv` gives an upstream-defined split:

- **Train:** 1,872 images (~94.5%)
- **Test:** 108 images (~5.5%)

Columns: `Image Name;Category;Block;Train/Test`. Header line present. CRLF line endings — load with `newline=''` or strip `\r` on read. The 1,980 filenames in the CSV exactly match the files in `images/` (zero orphans either direction, verified with `comm`).

**No validation set is provided.** Carving a validation split from the 1,872 train images is a Phase 3 loader responsibility. Recommended: hold out ~10% of train (~187 images), stratified by category so all four UI types are represented; use a fixed random seed so the split is reproducible across runs.

### Upstream documentation drift

The upstream `UEyes_dataset/README.md` refers to a file called `info.csv`. The file that actually ships in the deposit is `image_types.csv` — same columns, just a different name. Not a problem, but the loader code should key on `image_types.csv` and a comment should note the rename so future readers don't wonder.

### Choices resolved in Phase 3

Phase 3 closed the three open choices from Phase 0. Decisions and one-line rationale captured here; the long form lives in [`LEARNINGS.md`](../LEARNINGS.md) 2026-04-16 Phase 3.

- **Training saliency target: `heatmaps_3s`.** Gaussian-smoothed continuous maps at a 3-second aggregate viewing window. Continuous signal gives gradients to fine-tune against (vs binary `fixmaps_*`); 3s is the closest analogue to SALICON's ~5s aggregation, which is what MSI-Net's pretrained weights were fit against. `overlay_heatmaps_*` stays out of scope for training. The `saliency_variant` argument on `UEyesDataset` lets Phase 6 compare variants without code changes.
- **Validation set: stratified 10% hold-out from upstream Train, fixed seed 42.** 1,872 upstream Train → 1,684 train + 188 val (47 per category × 4). Upstream Test (108) stays untouched. Fresh `np.random.default_rng(seed)` per carve — no dependence on global NumPy state.
- **Preprocessing: aspect-preserving PIL BICUBIC resize + constant pad to (240, 320).** Stimuli padded with 126 (mid-grey, Kroner's convention), saliency maps padded with 0. Stimulus tensors emerge as `(3, 240, 320)` float32 RGB in `[0, 255]`; saliency tensors as `(1, 240, 320)` float32 in `[0, 1]`.

### Upstream CSV quirk

The `Block` column in `image_types.csv` is mixed: most rows store the integer plainly (`"0"`, `"23"`), but a handful are in Excel scientific notation (`"0,00E+00"`) — the file was clearly opened in Excel at some point. The loader keeps `Block` as a raw string; nothing downstream uses it. Recorded here so the next contributor to parse it as `int()` isn't surprised.

### Disk usage note

After a successful run, `data/ueyes/` holds:
- ~13 GB for `UEyes_dataset.zip` (kept in place so `data/fetch.sh` re-runs are no-ops; safe to delete after confirming extraction worked)
- ~12 GB for the unpacked `UEyes_dataset/` tree

Roughly 25 GB on disk. If space is tight, delete the zip after verification:

```sh
rm data/ueyes/UEyes_dataset.zip
```

A re-run of `data/fetch.sh` is a fast no-op if the unpacked tree is present (sentinel file: `UEyes_dataset/image_types.csv`). If the tree is missing but the zip is, the script skips the download and goes straight to extraction. Set `FETCH_FORCE=1` to force a clean redownload and re-extract.

## Licence and attribution

Creative Commons Attribution 4.0 International (CC BY 4.0). Commercial use permitted; attribution required.

If you use this dataset — whether directly, via a model trained with it, or via Foveacast's shipped artefact — cite:

> Yue Jiang, Luis A. Leiva, Hamed Rezazadegan Tavakoli, Paul R. B. Houssel, Julia Kylmälä, and Antti Oulasvirta. 2023. UEyes: Understanding Visual Saliency across User Interface Types. In *Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems (CHI '23)*. Association for Computing Machinery, New York, NY, USA, Article 285, 1–21. https://doi.org/10.1145/3544548.3581096

Foveacast carries this citation in its attribution footer once a model trained on UEyes ships. If you build something else on top of this repo's output, you carry the citation too.

## What does not belong here

- Trained checkpoints (`checkpoints/` or `runs/` at the repo root — gitignored).
- Model weights from elsewhere (different project, different attribution story, keep them separate).
- Any dataset that is not UEyes. If future work uses additional datasets, they get their own subfolder with their own README and their own licence note.
