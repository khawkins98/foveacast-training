# Dataset — UEyes

This folder is where the UEyes dataset goes after you fetch it. The data itself is not committed to this repo; at 12.9 GB zipped it belongs on Zenodo, not in git.

## Fetching

The UEyes dataset is hosted on Zenodo at [record 8010312](https://zenodo.org/records/8010312) under Creative Commons Attribution 4.0 International. A single `UEyes_dataset.zip` file.

A convenience script will live at `data/fetch.sh` when the training code lands. Until then, fetch manually:

```sh
# From the repo root:
mkdir -p data/ueyes
cd data/ueyes
curl -L -o UEyes_dataset.zip "https://zenodo.org/records/8010312/files/UEyes_dataset.zip?download=1"
unzip UEyes_dataset.zip
```

The unzipped layout and any preprocessing specifics will be documented here once the `ueyes_dataset.py` loader lands and I know exactly what shape the files come in.

## What's in the dataset

From the Zenodo record and the [CHI 2023 paper](https://doi.org/10.1145/3544548.3581096):

- 1,980 UI screenshots spanning four types: webpage, desktop UI, mobile UI, and poster.
- Eye-tracking data from 62 participants.
- Ground-truth saliency maps derived from participant fixations.

## Licence and attribution

Creative Commons Attribution 4.0 International (CC BY 4.0). Commercial use permitted; attribution required.

If you use this dataset — whether directly, via a model trained with it, or via Foveacast's shipped artefact — cite:

> Yue Jiang, Luis A. Leiva, Hamed Rezazadegan Tavakoli, Paul R. B. Houssel, Julia Kylmälä, and Antti Oulasvirta. 2023. UEyes: Understanding Visual Saliency across User Interface Types. In *Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems (CHI '23)*. Association for Computing Machinery, New York, NY, USA, Article 285, 1–21. https://doi.org/10.1145/3544548.3581096

Foveacast carries this citation in its attribution footer once a model trained on UEyes ships. If you build something else on top of this repo's output, you carry the citation too.

## What does not belong here

- Trained checkpoints (`checkpoints/` or `runs/` at the repo root — gitignored).
- Model weights from elsewhere (different project, different attribution story, keep them separate).
- Any dataset that is not UEyes. If future work uses additional datasets, they get their own subfolder with their own README and their own licence note.
