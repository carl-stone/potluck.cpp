#!/usr/bin/env bash
# Run the retained direct-ring checks and integrated local server gate.
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
source "${REPO}/scripts/potluck-safe.sh"
BIN="${BIN:-${REPO}/build/bin}"
if [[ -z "${MODEL:-}" ]]; then
    MODEL="$(bash "${REPO}/scripts/fetch-model.sh")"
fi
export REPO BIN MODEL

if [[ ! -f "${MODEL}" ]]; then
    printf 'missing model: %s\n' "${MODEL}" >&2
    exit 2
fi

for test_bin in test-potluck-discovery test-potluck-protocol test-potluck-refresh test-potluck-probe test-potluck-transport test-potluck-halda test-potluck-qwen35-stages test-potluck-run; do
    if [[ ! -x "${BIN}/${test_bin}" ]]; then
        printf 'missing test binary: %s\n' "${BIN}/${test_bin}" >&2
        exit 2
    fi
done

potluck_require_memory_for_model "${MODEL}"

for test_bin in test-potluck-discovery test-potluck-protocol test-potluck-refresh test-potluck-transport test-potluck-halda; do
    printf '== %s ==\n' "${test_bin}"
    "${BIN}/${test_bin}"
done
printf '== test-potluck-probe ==\n'
"${BIN}/test-potluck-probe" "${BIN}/potluck-worker"

printf '== test-potluck-qwen35-stages ==\n'
"${BIN}/test-potluck-qwen35-stages" "${MODEL}"

printf '== test-potluck-run ==\n'
"${BIN}/test-potluck-run" "${MODEL}"

printf '== integrated potluck-server ==\n'
POTLUCK_TEST_WORKER_LOSS=1 bash "${REPO}/tests/potluck/test_server.sh" \
    "${POTLUCK_TEST_WORKERS:-2}" "${N_PREDICT:-8}" "${HOST:-127.0.0.1}"
printf '== speculative server ==\n'
bash "${REPO}/tests/potluck/test_speculative.sh"
printf '== workload overrides ==\n'
bash "${REPO}/tests/potluck/test_overrides.sh"

printf 'DIRECT-RING SUITE PASSED\n'
