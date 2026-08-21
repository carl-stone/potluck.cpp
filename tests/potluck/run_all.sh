#!/usr/bin/env bash
# Run the retained direct-ring checks and integrated local server gate.
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
BIN="${BIN:-${REPO}/build/bin}"
if [[ -z "${MODEL:-}" ]]; then
    MODEL="$(bash "${REPO}/scripts/fetch-model.sh")"
fi
export REPO BIN MODEL

if [[ ! -f "${MODEL}" ]]; then
    printf 'missing model: %s\n' "${MODEL}" >&2
    exit 2
fi

for test_bin in test-potluck-discovery test-potluck-protocol test-potluck-transport test-potluck-qwen35-stages test-potluck-run; do
    if [[ ! -x "${BIN}/${test_bin}" ]]; then
        printf 'missing test binary: %s\n' "${BIN}/${test_bin}" >&2
        exit 2
    fi
done

for test_bin in test-potluck-discovery test-potluck-protocol test-potluck-transport; do
    printf '== %s ==\n' "${test_bin}"
    "${BIN}/${test_bin}"
done

printf '== test-potluck-qwen35-stages ==\n'
"${BIN}/test-potluck-qwen35-stages" "${MODEL}"

printf '== test-potluck-run ==\n'
"${BIN}/test-potluck-run" "${MODEL}"

printf '== integrated potluck-server ==\n'
bash "${REPO}/tests/potluck/test_server.sh"

printf 'DIRECT-RING SUITE PASSED\n'
