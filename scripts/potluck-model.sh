#!/usr/bin/env bash
# Pinned engineering fixture model for Potluck checks and smokes.
# Source this file; do not execute it directly.
#
# Overrides:
#   POTLUCK_MODEL_HF_REPO   Hugging Face repo id
#   POTLUCK_MODEL_FILE      file name inside the repo
#   POTLUCK_MODEL_SHA256    expected checksum of the file
#   POTLUCK_MODEL_DIR       download/cache directory (default: <repo>/models)
#   POTLUCK_TEST_MODEL      explicit fixture path; used by the default flow

POTLUCK_MODEL_HF_REPO="${POTLUCK_MODEL_HF_REPO:-ggml-org/Qwen3.5-0.8B-GGUF}"
POTLUCK_MODEL_FILE="${POTLUCK_MODEL_FILE:-Qwen3.5-0.8B-Q4_0.gguf}"
POTLUCK_MODEL_SHA256="${POTLUCK_MODEL_SHA256:-57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf}"

POTLUCK_LARGE_HF_REPO="${POTLUCK_LARGE_HF_REPO:-ggml-org/gemma-3-27b-it-GGUF}"
POTLUCK_LARGE_FILE="${POTLUCK_LARGE_FILE:-gemma-3-27b-it-Q4_K_M.gguf}"
POTLUCK_LARGE_SHA256="${POTLUCK_LARGE_SHA256:-edc9aff4d811a285b9157618130b08688b0768d94ee5355b02dc0cb713012e15}"

potluck_model_repo_root() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    (cd "${script_dir}/.." && pwd)
}

potluck_model_url() {
    printf 'https://huggingface.co/%s/resolve/main/%s\n' \
        "${POTLUCK_MODEL_HF_REPO}" "${POTLUCK_MODEL_FILE}"
}

potluck_model_dir() {
    printf '%s\n' "${POTLUCK_MODEL_DIR:-$(potluck_model_repo_root)/models}"
}

potluck_model_path() {
    if [[ -n "${POTLUCK_TEST_MODEL:-}" ]]; then
        printf '%s\n' "${POTLUCK_TEST_MODEL}"
    else
        printf '%s/%s\n' "$(potluck_model_dir)" "${POTLUCK_MODEL_FILE}"
    fi
}

potluck_large_path() {
    printf '%s/%s\n' "$(potluck_model_dir)" "${POTLUCK_LARGE_FILE}"
}
