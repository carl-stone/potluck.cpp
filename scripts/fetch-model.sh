#!/usr/bin/env bash
# Download the selected pinned model if it is missing, verify its checksum,
# and print the model path on stdout.
#
# Usage: scripts/fetch-model.sh [--large] [--dest DIR]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=potluck-model.sh
source "${REPO}/scripts/potluck-model.sh"

large=0
dest="$(potluck_model_dir)"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --large)
            large=1; shift ;;
        --dest)
            [[ $# -ge 2 ]] || { echo "fetch-model: --dest needs a value" >&2; exit 2; }
            dest="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | cut -c3-; exit 0 ;;
        *)
            echo "fetch-model: unknown option: $1" >&2; exit 2 ;;
    esac
done

if (( large )); then
    hf_repo="${POTLUCK_LARGE_HF_REPO}"
    file="${POTLUCK_LARGE_FILE}"
    sha256="${POTLUCK_LARGE_SHA256}"
else
    hf_repo="${POTLUCK_MODEL_HF_REPO}"
    file="${POTLUCK_MODEL_FILE}"
    sha256="${POTLUCK_MODEL_SHA256}"
fi

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | cut -d' ' -f1
    else
        printf 'fetch-model: need sha256sum or shasum to verify the download\n' >&2
        return 1
    fi
}

if (( ! large )) && [[ -n "${POTLUCK_TEST_MODEL:-}" ]]; then
    if [[ -f "${POTLUCK_TEST_MODEL}" ]]; then
        printf '%s\n' "${POTLUCK_TEST_MODEL}"
        exit 0
    fi
    printf 'fetch-model: POTLUCK_TEST_MODEL is set but missing: %s\n' \
        "${POTLUCK_TEST_MODEL}" >&2
    exit 1
fi

path="${dest%/}/${file}"
if [[ -f "${path}" ]]; then
    if actual="$(sha256_file "${path}")" && [[ "${actual}" == "${sha256}" ]]; then
        printf '%s\n' "${path}"
        exit 0
    fi
fi

mkdir -p "${dest}"
url="https://huggingface.co/${hf_repo}/resolve/main/${file}"
printf 'fetch-model: downloading %s\n' "${url}" >&2
tmp="${path}.part"
if curl -L --fail --progress-bar -o "${tmp}" "${url}"; then
    :
else
    status=$?
    rm -f "${tmp}"
    exit "${status}"
fi

actual="$(sha256_file "${tmp}")" || {
    rm -f "${tmp}"
    exit 1
}
if [[ "${actual}" != "${sha256}" ]]; then
    printf 'fetch-model: checksum mismatch for %s\n  expected %s\n  actual   %s\n' \
        "${file}" "${sha256}" "${actual}" >&2
    rm -f "${tmp}"
    exit 1
fi

mv "${tmp}" "${path}"
printf '%s\n' "${path}"
