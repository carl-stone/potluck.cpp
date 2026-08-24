#!/usr/bin/env bash
# Behavioral coverage for manual layer-window overrides.
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BIN="${BIN:-${REPO}/build/bin}"
HOST="${HOST:-${POTLUCK_TEST_HOST:-127.0.0.1}}"
STARTUP_TIMEOUT="${POTLUCK_TEST_STARTUP_TIMEOUT:-180}"
NEXT_PORT="${POTLUCK_TEST_PORT:-$((8100 + RANDOM % 400))}"

# shellcheck source=../../scripts/potluck-model.sh
source "${REPO}/scripts/potluck-model.sh"
MODEL="${MODEL:-$(potluck_model_path)}"
# shellcheck source=./test_helpers.sh
source "${REPO}/tests/potluck/test_helpers.sh"

if [[ ! -x "${BIN}/potluck-server" || ! -x "${BIN}/potluck-worker" ]]; then
    printf 'missing potluck-server or potluck-worker (build them first): %s\n' "${BIN}" >&2
    exit 2
fi
if [[ ! -f "${MODEL}" ]]; then
    printf 'missing model: %s\n' "${MODEL}" >&2
    exit 2
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/potluck-overrides.XXXXXX")"
SRV=""
SRV_LOG=""
LAST_LOG=""
CURRENT_PORT=""

wait_pid_bounded() {
    local pid="$1"
    local timeout="$2"
    local deadline=$((SECONDS + timeout))
    while kill -0 "${pid}" 2>/dev/null; do
        if (( SECONDS >= deadline )); then
            return 124
        fi
        sleep 0.1
    done
    wait "${pid}" 2>/dev/null || true
}

wait_pid_exit_bounded() {
    local rc=0
    wait_pid_bounded "$1" "$2" || rc=$?
    [[ "${rc}" -ne 124 ]]
}

terminate_pid_bounded() {
    local pid="$1"
    local timeout="$2"
    if ! kill -0 "${pid}" 2>/dev/null; then
        wait "${pid}" 2>/dev/null || true
        return 0
    fi
    kill "${pid}" 2>/dev/null || true
    if wait_pid_exit_bounded "${pid}" "${timeout}"; then
        return 0
    fi
    kill -KILL "${pid}" 2>/dev/null || true
    wait_pid_exit_bounded "${pid}" 5
}

stop_server() {
    local server_pid="${SRV}"
    local worker_pids=""
    local status=0
    if [[ -z "${server_pid}" ]]; then
        return 0
    fi
    worker_pids="$(pgrep -P "${server_pid}" -f '[p]otluck-worker' 2>/dev/null || true)"
    if ! terminate_pid_bounded "${server_pid}" 10; then
        status=1
    fi
    for worker in ${worker_pids}; do
        if ! terminate_pid_bounded "${worker}" 10; then
            status=1
        fi
    done
    if (( status != 0 )); then
        printf 'server or worker did not exit within the bounded shutdown wait\n' >&2
    else
        SRV=""
    fi
    return "${status}"
}

cleanup() {
    local rc=$?
    if [[ -n "${SRV}" ]]; then
        stop_server || true
    fi
    if (( rc != 0 )) && [[ -n "${LAST_LOG}" && -f "${LAST_LOG}" ]]; then
        printf '%s\n' '--- last server log ---' >&2
        cat "${LAST_LOG}" >&2
    fi
    rm -rf "${WORK}" 2>/dev/null || true
    return "${rc}"
}
trap cleanup EXIT

next_port() {
    CURRENT_PORT="${NEXT_PORT}"
    NEXT_PORT=$((NEXT_PORT + 1))
}

start_server() {
    local name="$1"
    shift
    next_port
    SRV_LOG="${WORK}/${name}.log"
    LAST_LOG="${SRV_LOG}"
    "${BIN}/potluck-server" -m "${MODEL}" --workers 2 --slots 2 --ubatch 1 \
        --prefetch advise --host "${HOST}" --port "${CURRENT_PORT}" "$@" \
        >"${SRV_LOG}" 2>&1 &
    SRV=$!
    local deadline=$((SECONDS + STARTUP_TIMEOUT))
    while (( SECONDS < deadline )); do
        if grep -Fq 'listening on http' "${SRV_LOG}" 2>/dev/null; then
            return 0
        fi
        if ! kill -0 "${SRV}" 2>/dev/null; then
            wait "${SRV}" 2>/dev/null || true
            return 1
        fi
        sleep 1
    done
    return 1
}

run_expected_failure() {
    local name="$1"
    local expected="$2"
    shift 2
    next_port
    local log="${WORK}/${name}.log"
    local pid=""
    local worker_pids=""
    local rc=0
    local deadline=0
    LAST_LOG="${log}"
    "${BIN}/potluck-server" -m "${MODEL}" --workers 2 --slots 2 --ubatch 1 \
        --prefetch advise --host "${HOST}" --port "${CURRENT_PORT}" "$@" \
        >"${log}" 2>&1 &
    pid=$!
    SRV="${pid}"
    SRV_LOG="${log}"
    deadline=$((SECONDS + STARTUP_TIMEOUT))
    while kill -0 "${pid}" 2>/dev/null; do
        if (( SECONDS >= deadline )); then
            printf '%s did not fail within the bounded wait\n' "${name}" >&2
            stop_server || true
            return 1
        fi
        sleep 0.1
    done
    worker_pids="$(pgrep -P "${pid}" -f '[p]otluck-worker' 2>/dev/null || true)"
    wait "${pid}" 2>/dev/null || rc=$?
    SRV=""
    for worker in ${worker_pids}; do
        terminate_pid_bounded "${worker}" 10 || true
    done
    if (( rc == 0 )); then
        printf '%s unexpectedly succeeded\n' "${name}" >&2
        return 1
    fi
    if ! grep -Fq -- "${expected}" "${log}"; then
        printf '%s did not report: %s\n' "${name}" "${expected}" >&2
        return 1
    fi
}

extract_layer_count() {
    python3 - "$1" <<'PY'
import re
import sys

ranges = []
pattern = re.compile(r"potluck-server: ring window \d+ owner=\d+ layers=\[(\d+),(\d+)\)")
with open(sys.argv[1], encoding="utf-8", errors="replace") as stream:
    for line in stream:
        if line.startswith("potluck-server: ring ready "):
            ranges = []
        match = pattern.search(line)
        if match:
            ranges.append((int(match.group(1)), int(match.group(2))))
if not ranges:
    raise SystemExit("no ring window log lines")
layer_count = max(end for _, end in ranges)
if layer_count <= 0:
    raise SystemExit("ring window log has no positive layer count")
print(layer_count)
PY
}

assert_windows() {
    python3 - "$1" "$2" "$3" <<'PY'
import re
import sys

log_path = sys.argv[1]
mode = sys.argv[2]
layer_count = int(sys.argv[3])
pattern = re.compile(
    r"potluck-server: ring window \d+ owner=(\d+) layers=\[(\d+),(\d+)\)"
)
ready_pattern = re.compile(r"potluck-server: ring ready \((\d+) workers,")
ranges = []
worker_count = None
with open(log_path, encoding="utf-8", errors="replace") as stream:
    for line in stream:
        if line.startswith("potluck-server: ring ready "):
            ranges = []
            ready = ready_pattern.search(line)
            worker_count = int(ready.group(1)) if ready else None
        match = pattern.search(line)
        if match:
            ranges.append((int(match.group(1)), int(match.group(2)), int(match.group(3))))
if not ranges:
    raise SystemExit("no ring window log lines")
if worker_count is None or worker_count < 1:
    raise SystemExit("no ring worker count in ready log")

if mode == "override":
    half = layer_count // 2
    expected = [(0, half), (half, layer_count)]
    actual = [(start, end) for _, start, end in ranges]
    if actual != expected:
        raise SystemExit(f"override windows were {actual}, expected {expected}")
else:
    cursor = 0
    for _, start, end in ranges:
        if start != cursor or end <= start:
            raise SystemExit(f"automatic windows do not tile at [{start},{end}) after {cursor}")
        cursor = end
    if cursor != layer_count:
        raise SystemExit(f"automatic windows end at {cursor}, expected {layer_count}")

owners = {owner for owner, _, _ in ranges}
expected_owners = set(range(worker_count))
if owners != expected_owners:
    raise SystemExit(
        f"{mode} windows use owners {sorted(owners)}, expected {sorted(expected_owners)}"
    )
PY
}


if ! start_server baseline; then
    printf 'baseline server did not become ready\n' >&2
    exit 1
fi
BASE_LOG="${SRV_LOG}"
L="$(extract_layer_count "${BASE_LOG}")"
if (( L < 4 || L % 2 != 0 )); then
    printf 'pinned model layer count must be even and at least four: %s\n' "${L}" >&2
    exit 1
fi
assert_windows "${BASE_LOG}" automatic "${L}"
if ! stop_server; then
    exit 1
fi

TOO_MANY="1"
for _ in $(seq 2 33); do
    TOO_MANY+=",1"
done
run_expected_failure zero-layer-window \
    '--layer-window expects a positive integer' -lw 0,4
run_expected_failure too-many-layer-windows \
    '--layer-window accepts at most 32 entries' -lw "${TOO_MANY}"
run_expected_failure nondividing-layer-window \
    '--layer-window sum must divide the model layer count exactly' -lw "$((L + 1))"
run_expected_failure one-layer-window \
    'HALDA fixed window widths must have one entry per device' -lw "${L}"

HALF=$((L / 2))
if ! start_server override -lw "${HALF},${HALF}"; then
    printf 'layer-window override server did not become ready\n' >&2
    exit 1
fi
OVERRIDE_LOG="${SRV_LOG}"
grep -Fq 'layer-window=' "${OVERRIDE_LOG}"
assert_windows "${OVERRIDE_LOG}" override "${L}"
curl -fsS -H 'Content-Type: application/json' \
    -d '{"prompt":"The capital of France is","n_predict":4,"temperature":0}' \
    "http://${HOST}:${CURRENT_PORT}/completion" >"${WORK}/override-completion.json"
python3 - "${WORK}/override-completion.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    response = json.load(stream)
content = response.get("content")
if not isinstance(content, str) or not content:
    raise SystemExit("override completion returned no content")
PY
if ! stop_server; then
    exit 1
fi

printf 'POTLUCK OVERRIDE TEST PASSED: L=%s automatic and manual windows\n' "${L}"
