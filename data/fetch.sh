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

# --- Short-circuit if already unpacked --------------------------------------

# why: re-running after a successful extract should be a fast no-op, not a
# re-extract of 12 GB. image_types.csv is a cheap sentinel — it's small,
# present in every complete extraction, and missing from a partial one.
SENTINEL="${DEST}/UEyes_dataset/image_types.csv"
if [[ -f "${SENTINEL}" && "${FETCH_FORCE:-0}" == "0" ]]; then
  echo "✓ UEyes already unpacked at ${DEST}/UEyes_dataset/"
  echo "  (sentinel: ${SENTINEL})"
  echo "  set FETCH_FORCE=1 to redownload and re-extract anyway"
  exit 0
fi

# --- Download ---------------------------------------------------------------

echo "→ Fetching ${ZIP_NAME} from Zenodo record ${ZENODO_RECORD}"
echo "  destination: ${DEST}/${ZIP_NAME}"

# Preflight: ask Zenodo the total size so a resumed run can print a clearer
# "resuming from X GB of Y GB" message. Non-fatal if this fails (offline, HEAD
# not supported, etc.) — the download itself is the source of truth.
#
# why: curl's own progress meter resets its percent counter after -C -, which
# reads as "1% of 3.6 GB remaining" even when we're 70% through the 12.9 GB
# file. A one-line preflight up front is cheaper than explaining curl's meter.
TOTAL_BYTES=$(curl -sIL --max-time 15 "${URL}" 2>/dev/null \
  | grep -i '^content-length:' | tail -1 | awk '{print $2}' | tr -d '\r\n' \
  || true)
CURRENT_BYTES=0
if [[ -f "${ZIP_NAME}" ]]; then
  CURRENT_BYTES=$(wc -c < "${ZIP_NAME}" | tr -d ' ')
fi

if [[ -n "${TOTAL_BYTES}" && "${TOTAL_BYTES}" -gt 0 ]]; then
  GB_TOTAL=$(( TOTAL_BYTES / 1000000000 ))
  if [[ "${CURRENT_BYTES}" -gt 0 ]]; then
    PCT=$(( CURRENT_BYTES * 100 / TOTAL_BYTES ))
    GB_CURRENT=$(( CURRENT_BYTES / 1000000000 ))
    echo "  resuming from ${GB_CURRENT} GB / ${GB_TOTAL} GB (${PCT}%)"
  else
    echo "  total size: ${GB_TOTAL} GB (expect 15–60+ min depending on bandwidth)"
  fi
else
  echo "  (expect 15–60+ min depending on bandwidth)"
fi

# why: combined, these flags turn "stalls silently forever" into "retries
# transparently." The flag that actually matters for Zenodo is
# --retry-all-errors — Zenodo stalls look like transport-level failures, not
# HTTP 5xx, so curl's default --retry doesn't fire on them. Without
# --speed-limit/--speed-time, a stalled connection sits at 0 B/s until the
# user notices and Ctrl-C's. With them, curl fails fast and the retry chain
# kicks in.
#   -L                  — follow Zenodo's 302 to the storage backend.
#   -C -                — resume from the partial file if present.
#   --fail              — HTTP 4xx/5xx becomes a non-zero exit instead of
#                         writing an error page over the zip.
#   --retry 5           — curl-level retries before the outer loop takes over.
#   --retry-all-errors  — retry on transport errors too, not only 5xx.
#   --retry-delay 10    — 10s between curl-level retries.
#   --speed-limit 1024  — if throughput drops below 1 KB/s ...
#   --speed-time 60     — ... for 60 consecutive seconds, abort the transfer
#                         so --retry can resume from the partial file.
#
# The surrounding `until` loop is the outermost safety net: if curl exhausts
# its five retries, we restart it from scratch. The download is still
# idempotent thanks to -C -.
max_tries=8
tries=0
until curl -L -C - --fail \
        --retry 5 --retry-delay 10 --retry-all-errors \
        --speed-limit 1024 --speed-time 60 \
        -o "${ZIP_NAME}" "${URL}"; do
  tries=$((tries + 1))
  if [[ ${tries} -ge ${max_tries} ]]; then
    echo "" >&2
    echo "error: curl failed ${tries} consecutive times, giving up" >&2
    echo "       partial file preserved at ${DEST}/${ZIP_NAME} — rerun when ready" >&2
    exit 1
  fi
  echo "" >&2
  echo "  curl exited non-zero (outer attempt ${tries}/${max_tries}) —" \
       "sleeping 15s then resuming" >&2
  sleep 15
done

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
