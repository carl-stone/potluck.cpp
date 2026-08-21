#!/usr/bin/env bash
# Red/green smoke test for Qwen3.5-family support in potluck.cpp.
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LLAMA_CLI="${LLAMA_CLI:-${REPO}/build/bin/llama-cli}"
# shellcheck source=../../scripts/potluck-model.sh
source "${REPO}/scripts/potluck-model.sh"
MODEL="${MODEL:-$(potluck_model_path)}"

if [[ ! -x "${LLAMA_CLI}" ]]; then
    printf 'missing llama-cli (build it first): %s\n' "${LLAMA_CLI}" >&2
    exit 2
fi
if [[ ! -f "${MODEL}" ]]; then
    printf 'missing Qwen3.5 fixture: %s\n' "${MODEL}" >&2
    exit 2
fi

log="$(mktemp -t potluck-qwen35-smoke.XXXXXX)"
trap 'rm -f "${log}"' EXIT

if ! "${LLAMA_CLI}" \
    -m "${MODEL}" \
    -c 256 \
    -n 8 \
    -p 'Say hello in one short sentence.' \
    >"${log}" 2>&1; then
    cat "${log}" >&2
    exit 1
fi

if grep -Eiq 'unknown architecture|unsupported.*qwen|qwen35.*not supported|not currently supported' "${log}"; then
    cat "${log}" >&2
    printf 'Qwen3.5 smoke test reached an unsupported-model path\n' >&2
    exit 1
fi

if ! grep -Eiq 'hello|model|sampler|llama' "${log}"; then
    cat "${log}" >&2
    printf 'Qwen3.5 smoke test produced no recognizable inference output\n' >&2
    exit 1
fi

cat "${log}"
printf 'Qwen3.5 potluck smoke test passed\n'
