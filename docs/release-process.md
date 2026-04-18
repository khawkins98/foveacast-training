# Release process

How to cut a new release of `foveacast-training`. Written as a checklist you can follow top-to-bottom. Derived from the v0.2.0 walkthrough on 2026-04-18 — the first time we shipped a multi-artefact release and discovered the process was only in someone's head.

## Context

A release has four outputs, and the process is "complicated" because three of them live outside git:

1. A version bump in `pyproject.toml` (in git).
2. A git tag on `main`, e.g. `v0.2.0` (in git).
3. A GitHub Release created against that tag (in GitHub, not the repo).
4. `.onnx` artefacts uploaded to the release (outside git — `releases/` is gitignored, so artefacts only exist on the machine that built them).

Foveacast (the downstream consumer) fetches `.onnx` files by URL from the GitHub Release. Artefact names are part of the API; renaming them breaks downstream fetch URLs.

## Version numbering

SemVer-ish: `0.MAJOR.MINOR`. Bump MINOR for new trained models, new artefact formats, or breaking CLI changes. A patch-level release would be for corrections (bad artefact, parity regression caught post-release) that don't add new deliverables.

## Artefact naming convention

`foveacast-v{arch-version}-{window}-{precision}.onnx`

- `{arch-version}` — the architecture generation. Currently `v3` (MSI-Net fine-tuned on UEyes). The leading `v` is literal.
- `{window}` — viewing-duration window: `1s`, `3s`, or `7s`.
- `{precision}` — `fp32`, `fp16`, or `int8`. FP32 isn't shipped in releases (too large); FP16 and INT8 are the user-facing choices.

Example: `foveacast-v3-3s-fp16.onnx`. Each `.onnx` ships with a `.parity.json` sibling (PyTorch ↔ ONNX Runtime forward-pass numerics). INT8 artefacts additionally ship a `.quality.json` sibling (CC/KLD/NSS on test split vs PyTorch FP32).

## Pre-release checklist (before merging the release PR)

Code + data:

- [ ] All training runs completed; `best.pt` checkpoints under `runs/full-{variant}-*/`.
- [ ] `eval.py` results vs stock at matched `--time-window` for each new model, saved to `runs/*/eval-vs-stock-*.json`.
- [ ] `export_onnx.py --fp16` produces an FP16 artefact per model in `releases/`.
- [ ] `quantize_int8.py` produces an INT8 artefact per model in `releases/`.
- [ ] Each INT8 artefact passes CC/KLD/NSS delta <3% per metric vs PyTorch (`eval.py --checkpoint releases/*-int8.onnx --compare runs/*/best.pt --time-window <w>`).
- [ ] `pytest` passes; `ruff check src/foveacast_training/` is clean.

Docs:

- [ ] `docs/training-guide.md` experiment log has bold rows for the new models plus stock-at-window rows for the baselines.
- [ ] `docs/training-guide.md` "Precision variants and quality" section has per-model INT8 numbers.
- [ ] `README.md` banner updated to reflect the new version (`**[vX.Y.Z released]...**`).
- [ ] `README.md` "Practical notes on quality" reflects the shipped artefacts' measured numbers if they moved materially.
- [ ] `LEARNINGS.md` dated entry for anything non-obvious that came up during training or quantisation.
- [ ] `pyproject.toml` version bumped.
- [ ] `CITATION.cff` version bumped if that file gains a version field (it currently doesn't).

## Release sequence

Assumes the release PR is open and approved.

1. Squash-merge the release PR into `main`:

   ```sh
   gh pr merge <PR_NUMBER> --squash \
     --subject "feat(...): <short summary> (#<issue>)" \
     --body "<release-style summary of the merged commits>"
   ```

2. Sync local `main`:

   ```sh
   git checkout main && git pull origin main
   ```

3. Tag the merge commit and push the tag:

   ```sh
   git tag vX.Y.Z -m "vX.Y.Z: <short description>"
   git push origin vX.Y.Z
   ```

4. Create the GitHub Release with artefacts attached. `gh release create` takes the tag, a title, a markdown body (use a heredoc), and a positional list of files to upload:

   ```sh
   gh release create vX.Y.Z \
     --title "vX.Y.Z — <short title>" \
     --notes "$(cat <<'EOF'
   <release body in markdown>
   EOF
   )" \
     releases/foveacast-v3-1s-fp16.onnx \
     releases/foveacast-v3-1s-fp16.parity.json \
     releases/foveacast-v3-1s-int8.onnx \
     releases/foveacast-v3-1s-int8.parity.json \
     releases/foveacast-v3-1s-int8.quality.json \
     ... continue for each duration ...
   ```

   For v0.2.0 this was 15 files (3 durations × 2 precisions, FP16 has a `.parity.json`, INT8 has `.parity.json` + `.quality.json`) totalling ~270 MB. Upload takes a few minutes on residential broadband.

5. Verify the release page. Tag, title, body, and all artefacts should be visible at `https://github.com/khawkins98/foveacast-training/releases/tag/vX.Y.Z`.

6. Coordinate with Foveacast if artefact names changed. That's a separate PR on the consumer repo updating fetch URLs.

## Pitfalls seen in practice

- **`releases/` is gitignored.** Artefacts live only on the machine that built them. Running `gh release create vX.Y.Z` from a fresh clone without re-exporting would ship a release with zero artefacts.
- **INT8 structural parity (random-uniform inputs) is not a quality gate.** It overestimates perceived quality loss by ~3–5×. Always measure CC/KLD/NSS on the UEyes test split via `eval.py --checkpoint <int8.onnx>` for the ship decision. See [`LEARNINGS.md`](../LEARNINGS.md) 2026-04-18 for the narrative.
- **Artefact naming drives downstream breakage.** v0.1.0 shipped `foveacast-v3-fp16.onnx`; v0.2.0 renamed it to `foveacast-v3-3s-fp16.onnx` for consistency with the new `-1s-` and `-7s-` siblings. Coordinate renames with Foveacast, or users 404 on fetch.
- **Squash-merge loses individual commit context in `main`.** For release PRs this is fine (the PR and issue carry the detail), but if the PR has genuinely distinct conceptual commits you want to preserve, use `--merge` instead of `--squash`.

## Where things live

- This doc: `docs/release-process.md`.
- Training guide (experiment log + precision variants): `docs/training-guide.md`.
- Dated decision log: `LEARNINGS.md`.
- README §How to cite this work for BibTeX when citing the release.
- `CITATION.cff` for structured citation metadata.
