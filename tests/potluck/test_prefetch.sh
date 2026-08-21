#!/usr/bin/env bash
# End-to-end acceptance for feature-parity row 8 (layer-weight prefetch).
#
# potluck-worker --prefetch warms the model file into the OS page cache in a
# background thread, overlapping that disk I/O with the downstream wiring and
# ready handshake, so the first chain decode does not stall on cold-cache page
# faults. Correctness must be unchanged (CHAIN PASSED) and the prefetch path
# must be measurable (the worker prints the warmed byte count and time).
#
# Usage: test_prefetch.sh [n_workers] [n_predict] [host]
set -euo pipefail

N_WORKERS="${1:-2}"
N_PREDICT="${2:-32}"
HOST="${3:-127.0.0.1}"
PORT_BASE=$((47000 + RANDOM % 2000))

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

BIN="${BIN:-${REPO}/build/bin}"
MODEL="${MODEL:-${REPO}/models/Qwen3.5-0.8B-Q4_0.gguf}"

if [[ ! -x "${BIN}/potluck-head" || ! -x "${BIN}/potluck-worker" ]]; then
    printf 'missing binaries (build potluck-head/potluck-worker first): %s\n' "${BIN}" >&2
    exit 2
fi
if [[ ! -f "${MODEL}" ]]; then
    printf 'missing Qwen3.5 fixture: %s\n' "${MODEL}" >&2
    exit 2
fi

work="$(mktemp -d -t potluck-prefetch.XXXXXX)"
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

workers="${work}/workers.txt"
for i in $(seq 0 $((N_WORKERS - 1))); do
    printf '%s:%d\n' "${HOST}" $((PORT_BASE + i)) >> "${workers}"
done

for i in $(seq 0 $((N_WORKERS - 1))); do
    "${BIN}/potluck-worker" "${MODEL}" "${HOST}" $((PORT_BASE + i)) --prefetch >"${work}/worker_${i}.log" 2>&1 &
    pids+=($!)
done
sleep 2

head_log="${work}/head.log"
if ! "${BIN}/potluck-head" "${MODEL}" "${workers}" "${N_PREDICT}" "${HOST}" --parity-check >"${head_log}" 2>&1; then
    cat "${head_log}" >&2
    printf 'test_prefetch failed (head rc nonzero)\n' >&2
    exit 1
fi

if ! grep -q 'CHAIN PASSED' "${head_log}"; then
    cat "${head_log}" >&2
    printf 'test_prefetch failed (no CHAIN PASSED)\n' >&2
    exit 1
fi

if ! grep -qE 'WORKER prefetch: warmed [1-9][0-9]* bytes' "${work}"/worker_*.log; then
    cat "${work}"/worker_*.log >&2
    printf 'test_prefetch failed (no measurable prefetch line)\n' >&2
    exit 1
fi

prefetch_line="$(grep -hoE 'WORKER prefetch: warmed [0-9]+ bytes in [0-9.]+s' "${work}"/worker_*.log | head -1)"
printf '  ok: %s\n' "${prefetch_line}"
printf 'test_prefetch passed (%s workers, %s predict, correctness unchanged)\n' "${N_WORKERS}" "${N_PREDICT}"
