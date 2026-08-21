#!/usr/bin/env bash
# Download the pinned fixture model if it is missing, verify its checksum,
# and print the model path on stdout.
#
# Usage: scripts/fetch-model.sh [--dest DIR]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=potluck-model.sh
source "${REPO}/scripts/potluck-model.sh"

dest="$(potluck_model_dir)"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest)
            [[ $# -ge 2 ]] || { echo "fetch-model: --dest needs a value" >&2; exit 2; }
            dest="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | cut -c3-; exit 0 ;;
        *)
            echo "fetch-model: unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ -n "${POTLUCK_TEST_MODEL:-}" ]]; then
    if [[ -f "${POTLUCK_TEST_MODEL}" ]]; then
        printf '%s\n' "${POTLUCK_TEST_MODEL}"
        exit 0
    fi
    printf 'fetch-model: POTLUCK_TEST_MODEL is set but missing: %s\n' \
        "${POTLUCK_TEST_MODEL}" >&2
    exit 1
fi

path="${dest%/}/${POTLUCK_MODEL_FILE}"
if [[ -f "${path}" ]]; then
    printf '%s\n' "${path}"
    exit 0
fi

mkdir -p "${dest}"
url="$(potluck_model_url)"
printf 'fetch-model: downloading %s\n' "${url}" >&2
tmp="${path}.part"
curl -L --fail --progress-bar -o "${tmp}" "${url}"

if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "${tmp}" | cut -d' ' -f1)"
elif command -v shasum >/dev/null 2>&1; then
    actual="$(shasum -a 256 "${tmp}" | cut -d' ' -f1)"
else
    printf 'fetch-model: need sha256sum or shasum to verify the download\n' >&2
    rm -f "${tmp}"
    exit 1
fi
if [[ "${actual}" != "${POTLUCK_MODEL_SHA256}" ]]; then
    printf 'fetch-model: checksum mismatch for %s\n  expected %s\n  actual   %s\n' \
        "${POTLUCK_MODEL_FILE}" "${POTLUCK_MODEL_SHA256}" "${actual}" >&2
    rm -f "${tmp}"
    exit 1
fi

mv "${tmp}" "${path}"
printf '%s\n' "${path}"
