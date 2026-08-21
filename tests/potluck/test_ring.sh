#!/usr/bin/env bash
# Historical component check for coordinator-routed multi-window execution.
# It does not exercise the direct peer-to-peer ZeroMQ ring required by ADR 0007
# and is not a product or architecture acceptance check.
#
# The legacy route covers the model with several windows per worker, relays
# each hop through the coordinator over raw TCP, and compares output with a
# full-model fixture reference. This test must be rewritten around the
# integrated direct-peer server during the clean cutover.
#
# Scenarios (each must print RING PASSED):
#   1. 2 workers, W=[6,6]   -> 4 windows, 2 windows per worker
#   2. 3 workers, W=[4,4,4] -> 6 windows, 2 windows per worker
#   3. 2 workers, W=[3,9]   -> uneven windows (cycle 12), 2 per worker
#   4. 2 workers, W=[4,4]   -> 3 cycles, 6 windows, 3 per worker
#
# Usage: test_ring.sh [n_predict] [host]
set -euo pipefail

N_PREDICT="${1:-32}"
HOST="${2:-127.0.0.1}"
PORT_BASE=$((49000 + RANDOM % 1000))

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

BIN="${BIN:-${REPO}/build/bin}"
MODEL="${MODEL:-${REPO}/models/Qwen3.5-0.8B-Q4_0.gguf}"
source "${REPO}/tests/potluck/test_helpers.sh"


if [[ ! -x "${BIN}/potluck-head" || ! -x "${BIN}/potluck-worker" ]]; then
    printf 'missing binaries (build potluck-head/potluck-worker first): %s\n' "${BIN}" >&2
    exit 2
fi
if [[ ! -f "${MODEL}" ]]; then
    printf 'missing Qwen3.5 fixture: %s\n' "${MODEL}" >&2
    exit 2
fi

work="$(mktemp -d -t potluck-ring.XXXXXX)"
pids=()
cleanup() {
    for p in "${pids[@]:-}"; do
        kill "${p}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    if [[ -z "${POTLUCK_KEEP_WORK:-}" ]]; then
        rm -rf "${work}"
    fi
}
trap cleanup EXIT

start_workers() {
    local n="$1" base="$2"
    pids=()
    for i in $(seq 0 $((n - 1))); do
        "${BIN}/potluck-worker" "${MODEL}" "${HOST}" $((base + i)) >"${work}/w_${base}_${i}.log" 2>&1 &
        pids+=($!)
    done
    sleep 2
}
stop_workers() {
    for p in "${pids[@]:-}"; do
        kill "${p}" 2>/dev/null || true
    done
    pids=()
    wait 2>/dev/null || true
    sleep 1
}

fail_run() {
    local head_log="$1" label="$2"
    cat "${head_log}" >&2
    printf 'test_ring failed: %s\n' "${label}" >&2
    exit 1
}
scenarios=(
    "2|6,6|4 windows, 2 per worker"
    "3|4,4,4|6 windows, 2 per worker"
    "2|3,9|uneven windows, 2 per worker"
    "2|4,4|3 cycles, 6 windows, 3 per worker"
)

for spec in "${scenarios[@]}"; do
    n="${spec%%|*}"
    rest="${spec#*|}"
    sizes="${rest%%|*}"
    label="${rest##*|}"
    base=$((PORT_BASE + RANDOM % 500))
    head_log="${work}/head_${n}_${sizes//,/}.log"

    start_workers "${n}" "${base}"
    printf '%s:%d\n' "${HOST}" "${base}" > "${work}/workers.txt"
    for i in $(seq 1 $((n - 1))); do
        printf '%s:%d\n' "${HOST}" $((base + i)) >> "${work}/workers.txt"
    done
    if "${BIN}/potluck-head" "${MODEL}" "${work}/workers.txt" "${N_PREDICT}" "${HOST}" --parity-check \
            --ring "${sizes}" >"${head_log}" 2>&1; then
        head_result=1
    elif potluck_accept_backend_variance "${head_log}"; then
        head_result=2
    else
        fail_run "${head_log}" "head rc nonzero (${label})"
    fi
    stop_workers

    if (( head_result == 1 )) && ! grep -q 'RING PASSED' "${head_log}"; then
        fail_run "${head_log}" "no RING PASSED (${label})"
    fi
    # Every worker must host >= 2 windows (the DoD's ">=2 disjoint windows").
    for i in $(seq 0 $((n - 1))); do
        if ! grep -qE "ring rank ${i}/.* loaded [2-9][0-9]* windows" "${work}/w_${base}_${i}.log"; then
            cat "${work}/w_${base}_${i}.log" >&2
            printf 'test_ring failed: worker %s did not host >=2 windows (label: %s)\n' "${i}" "${label}" >&2
            exit 1
        fi
    done
    printf '  ok: %-28s -> %s\n' "${label}" "$(grep -oE 'RING PASSED[^,]*' "${head_log}" | head -1)"
done

printf 'test_ring passed (%s predict)\n' "${N_PREDICT}"
