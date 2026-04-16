#!/usr/bin/env bash
# data/fetch.sh — fetch the UEyes dataset from Zenodo.
#
# The UEyes dataset (Jiang et al. 2023, CHI '23) is a single ~12.9 GB zip
# hosted on Zenodo record 8010312 under CC BY 4.0. It is not committed to
# this repo — per data/README.md it lives in gitignored data/ueyes/ and is
# fetched on demand by contributors who are actually going to train.
#
# This script is idempotent and resumable: re-running it after a partial
# download picks up where curl left off, and re-running it after a full
# extraction is a no-op (the zip is skipped, unzip is skipped).
#
# Usage:
#   bash data/fetch.sh                # download + verify + extract
#   FETCH_SKIP_UNZIP=1 bash ...       # just download the zip
#   FETCH_DEST=/some/path bash ...    # extract somewhere other than data/ueyes
#
# Attribution: if you use this dataset — directly or via a model trained on
# it — cite Jiang et al. 2023. See data/README.md for the full citation.

set -euo pipefail

# why: pin Zenodo record and filename as constants. If UEyes ever re-deposits
# under a new record id, bumping these two values is the entire diff.
ZENODO_RECORD="8010312"
ZIP_NAME="UEyes_dataset.zip"
URL="https://zenodo.org/records/${ZENODO_RECORD}/files/${ZIP_NAME}?download=1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${FETCH_DEST:-${SCRIPT_DIR}/ueyes}"

# --- Preflight --------------------------------------------------------------

for bin in curl unzip; do
  if ! command -v "${bin}" >/dev/null 2>&1; then
    echo "error: '${bin}' is required but not on PATH" >&2
    exit 1
  fi
done

mkdir -p "${DEST}"
cd "${DEST}"

# --- Download ---------------------------------------------------------------

echo "→ Fetching ${ZIP_NAME} from Zenodo record ${ZENODO_RECORD}"
echo "  destination: ${DEST}/${ZIP_NAME}"
echo "  (12.9 GB — expect 15–60+ min depending on bandwidth)"

# why: -L follows the 302 to Zenodo's storage backend; -C - resumes a
# partial file so an interrupted run doesn't restart from zero; --fail
# turns HTTP 4xx/5xx into non-zero exits instead of writing an error
# page to disk; --retry handles transient blips without human babysitting.
curl -L -C - --fail --retry 3 --retry-delay 5 \
  -o "${ZIP_NAME}" "${URL}"

# --- Verify -----------------------------------------------------------------

echo "→ Verifying archive integrity"
# why: unzip -t walks the central directory and checks CRCs without
# extracting. If the file is truncated or corrupt, this fails now rather
# than 8 GB into an extraction.
unzip -tq "${ZIP_NAME}"

# --- Extract ----------------------------------------------------------------

if [[ "${FETCH_SKIP_UNZIP:-0}" != "0" ]]; then
  echo "✓ FETCH_SKIP_UNZIP set — leaving zip un-extracted at ${DEST}/${ZIP_NAME}"
  exit 0
fi

echo "→ Extracting into ${DEST}"
# why: -o overwrites without prompting so a re-run doesn't block on
# "replace? [y]es/[n]o/..." for every existing file. Extraction is
# idempotent by construction — the same zip produces the same tree.
unzip -q -o "${ZIP_NAME}"

# --- Report -----------------------------------------------------------------

echo ""
echo "✓ Done."
echo ""
echo "Top-level contents of ${DEST}:"
ls -la "${DEST}" | sed 's/^/    /'

echo ""
echo "Directory tree, depth 2 (paste this back to the maintainer / LLM so"
echo "data/README.md can be updated with the real layout):"
echo ""
echo "--- BEGIN UEYES TREE ---"
# why: depth 2 is enough to see the shape (top-level folders + one level
# of their contents) without spamming the terminal with thousands of image
# filenames. Sorting keeps the output stable across machines.
find "${DEST}" -maxdepth 2 -not -path "${DEST}" | sort
echo "--- END UEYES TREE ---"
