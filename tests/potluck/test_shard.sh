#!/usr/bin/env bash
# Validate per-window GGUF creation, loading, parity, and mismatch errors.
# Usage: test_shard.sh [host]
set -euo pipefail

HOST="${1:-127.0.0.1}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BIN="${BIN:-${REPO}/build/bin}"
MODEL="${MODEL:-${REPO}/models/Qwen3.5-0.8B-Q4_0.gguf}"

[[ -x "${BIN}/potluck-shard" ]] || { printf 'missing potluck-shard in %s\n' "${BIN}" >&2; exit 2; }
[[ -x "${BIN}/potluck-head" && -x "${BIN}/potluck-worker" ]] || {
    printf 'missing potluck-head/potluck-worker in %s\n' "${BIN}" >&2
    exit 2
}
[[ -f "${MODEL}" ]] || { printf 'missing model: %s\n' "${MODEL}" >&2; exit 2; }

WORK=$(mktemp -d "${TMPDIR:-/tmp}/potluck-shard.XXXXXX")
pids=()
cleanup() {
    for pid in "${pids[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    if [[ -z "${POTLUCK_KEEP_WORK:-}" ]]; then
        rm -rf "${WORK}"
    else
        printf 'kept work directory: %s\n' "${WORK}" >&2
    fi
}
trap cleanup EXIT

mkdir -p "${WORK}/shards"
"${BIN}/potluck-shard" "${MODEL}" --parts 4 -o "${WORK}/shards" >"${WORK}/shard.log"
SHARDS=("${WORK}"/shards/*.gguf)
[[ "${#SHARDS[@]}" -eq 4 ]] || { printf 'expected 4 shards\n' >&2; exit 1; }
for shard in "${SHARDS[@]}"; do
    [[ -s "${shard}" ]] || { printf 'empty shard: %s\n' "${shard}" >&2; exit 1; }
done

# The same short chain test is run once from the generated windows. It
# compares the shard-backed chain against the full model and checks tensor
# bytes, metadata, and global layer names without extending the known
# multi-worker near-tie window on this Metal fixture.
models_csv=$(IFS=,; printf '%s' "${SHARDS[*]}")
POTLUCK_WORKER_MODELS="${models_csv}" POTLUCK_BOUNDS=0,6,12,18,24 \
    bash "${REPO}/tests/potluck/test_chain.sh" 4 2 "${HOST}"

# Deliberately give stage 1 stage 0's shard. The worker must reject the window
# and the coordinator must exit instead of waiting forever for a ready message.
PORT_BASE=$((47000 + RANDOM % 1000))
workers_file="${WORK}/workers.txt"
printf '%s:%s\n%s:%s\n%s:%s\n%s:%s\n' \
    "${HOST}" "${PORT_BASE}" \
    "${HOST}" "$((PORT_BASE + 1))" \
    "${HOST}" "$((PORT_BASE + 2))" \
    "${HOST}" "$((PORT_BASE + 3))" >"${workers_file}"
"${BIN}/potluck-worker" "${SHARDS[0]}" "${HOST}" "${PORT_BASE}" >"${WORK}/worker0.log" 2>&1 &
pids+=("$!")
"${BIN}/potluck-worker" "${SHARDS[0]}" "${HOST}" "$((PORT_BASE + 1))" >"${WORK}/worker1.log" 2>&1 &
pids+=("$!")
"${BIN}/potluck-worker" "${SHARDS[2]}" "${HOST}" "$((PORT_BASE + 2))" >"${WORK}/worker2.log" 2>&1 &
pids+=("$!")
"${BIN}/potluck-worker" "${SHARDS[3]}" "${HOST}" "$((PORT_BASE + 3))" >"${WORK}/worker3.log" 2>&1 &
pids+=("$!")
sleep 2
set +e
POTLUCK_TIMEOUT_HANDSHAKE_S=2 POTLUCK_TIMEOUT_DECODE_S=2 \
    "${BIN}/potluck-head" "${MODEL}" "${workers_file}" 4 "${HOST}" \
    --bounds 0,6,12,18,24 >"${WORK}/mismatch-head.log" 2>&1 &
head_pid=$!
set -e
for _ in $(seq 1 80); do
    if ! kill -0 "${head_pid}" 2>/dev/null; then
        break
    fi
    sleep 0.25
done
if kill -0 "${head_pid}" 2>/dev/null; then
    kill "${head_pid}" 2>/dev/null || true
    wait "${head_pid}" 2>/dev/null || true
    printf 'mismatched shard test hung\n' >&2
    exit 1
fi
if wait "${head_pid}"; then
    head_rc=0
else
    head_rc=$?
fi
if [[ "${head_rc}" -eq 0 ]]; then
    printf 'mismatched shard unexpectedly succeeded\n' >&2
    exit 1
fi
if ! grep -q "assigned layers \[6,12) but shard file .* holds \[0,6)" "${WORK}/worker1.log"; then
    printf '%s\n' '--- worker0 ---' >&2
    cat "${WORK}/worker0.log" >&2 || true
    printf '%s\n' '--- worker1 ---' >&2
    cat "${WORK}/worker1.log" >&2 || true
    printf '%s\n' '--- head ---' >&2
    cat "${WORK}/mismatch-head.log" >&2 || true
    printf 'shard mismatch error did not name both windows\n' >&2
    exit 1
fi

printf 'POTLUCK-SHARD TEST PASSED: 4 shards, chain parity, mismatch rejection\n'
