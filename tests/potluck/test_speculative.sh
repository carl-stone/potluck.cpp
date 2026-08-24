#!/usr/bin/env bash
# End-to-end speculative decoding check for the Potluck server.
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BIN="${BIN:-${REPO}/build/bin}"
# shellcheck source=../../scripts/potluck-model.sh
source "${REPO}/scripts/potluck-model.sh"
MODEL="${MODEL:-$(potluck_model_path)}"

# Keep this source limited to the shared numerical-variance helper.
source "${REPO}/tests/potluck/test_helpers.sh"

if [[ ! -x "${BIN}/potluck-server" ]]; then
    printf 'missing potluck-server (build it first): %s\n' "${BIN}/potluck-server" >&2
    exit 2
fi
if [[ ! -f "${MODEL}" ]]; then
    printf 'missing model: %s\n' "${MODEL}" >&2
    exit 2
fi

WORK=$(mktemp -d "${TMPDIR:-/tmp}/potluck-speculative.XXXXXX")
SERVER_PID=""

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
    wait "${pid}" 2>/dev/null
}
wait_pid_exit_bounded() {
    local rc=0
    wait_pid_bounded "$1" "$2" || rc=$?
    [[ "${rc}" -ne 124 ]]
}
terminate_pid_bounded() {
    local pid="$1"
    local timeout="$2"
    kill "${pid}" 2>/dev/null || true
    if wait_pid_exit_bounded "${pid}" "${timeout}"; then
        return 0
    fi
    kill -KILL "${pid}" 2>/dev/null || true
    wait_pid_exit_bounded "${pid}" 5
}
stop_server() {
    local server_pid="${SERVER_PID}"
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
        if ! wait_pid_exit_bounded "${worker}" 10; then
            if ! terminate_pid_bounded "${worker}" 5; then
                status=1
            fi
        fi
    done
    if (( status != 0 )); then
        printf 'server or worker did not exit within the bounded shutdown wait\n' >&2
    else
        SERVER_PID=""
    fi
    return "${status}"
}
cleanup() {
    local rc=$?
    if (( rc != 0 )); then
        for log in "${WORK}/plain.log" "${WORK}/speculative.log" \
                   "${WORK}/ngram.log"; do
            if [[ -f "${log}" ]]; then
                printf '%s\n' "--- ${log} ---" >&2
                cat "${log}" >&2
            fi
        done
    fi
    stop_server || true
    rm -rf "${WORK}" 2>/dev/null || true
    return "${rc}"
}
trap cleanup EXIT

start_server() {
    local log="$1"
    local port="$2"
    shift 2
    "${BIN}/potluck-server" -m "${MODEL}" --workers 2 --slots 1 --ubatch 32 \
        --host 127.0.0.1 --port "${port}" "$@" >"${log}" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 120); do
        if grep -q 'listening on http' "${log}" 2>/dev/null; then
            return 0
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            cat "${log}" >&2
            printf 'server exited during startup\n' >&2
            return 1
        fi
        sleep 1
    done
    cat "${log}" >&2
    printf 'server did not become ready\n' >&2
    return 1
}

PORT="${PORT:-$((8100 + RANDOM % 400))}"
SPEC_PORT="${SPEC_PORT:-$((PORT + 1))}"
NGRAM_PORT="${NGRAM_PORT:-$((PORT + 2))}"
REQUEST='{"prompt":"The capital of France is","n_predict":16,"temperature":0,"seed":1}'
NGRAM_REQUEST='{"prompt":"alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu alpha beta gamma delta epsilon zeta eta theta iota kappa lambda","n_predict":16,"temperature":0,"seed":1}'
AUTH_HEADER=(-H 'Content-Type: application/json')

start_server "${WORK}/plain.log" "${PORT}"
curl -fsS "${AUTH_HEADER[@]}" -d "${REQUEST}" \
    "http://127.0.0.1:${PORT}/completion" >"${WORK}/plain.json"
python3 - "${WORK}/plain.json" "${WORK}/plain.content" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    result = json.load(handle)
content = result.get("content")
assert isinstance(content, str) and content
with open(sys.argv[2], "w") as handle:
    handle.write(content)
PY
curl -fsS "${AUTH_HEADER[@]}" -d "${NGRAM_REQUEST}" \
    "http://127.0.0.1:${PORT}/completion" >"${WORK}/ngram-plain.json"
python3 - "${WORK}/ngram-plain.json" "${WORK}/ngram-plain.content" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    result = json.load(handle)
content = result.get("content")
assert isinstance(content, str) and content
with open(sys.argv[2], "w") as handle:
    handle.write(content)
PY
stop_server

start_server "${WORK}/speculative.log" "${SPEC_PORT}" \
    --spec-draft-model "${MODEL}" --spec-type draft-simple --spec-draft-n-max 4
curl -fsS "${AUTH_HEADER[@]}" -d "${REQUEST}" \
    "http://127.0.0.1:${SPEC_PORT}/completion" >"${WORK}/speculative.json"
python3 - "${WORK}/speculative.json" "${WORK}/speculative.content" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    result = json.load(handle)
content = result.get("content")
assert isinstance(content, str) and content
with open(sys.argv[2], "w") as handle:
    handle.write(content)
PY
stop_server

if ! cmp -s "${WORK}/plain.content" "${WORK}/speculative.content"; then
    printf 'plain and speculative completion content differs\n' >&2
    diff -u "${WORK}/plain.content" "${WORK}/speculative.content" >&2 || true
    exit 1
fi
python3 - "${WORK}/speculative.log" <<'PY'
import re
import sys

pattern = re.compile(
    r"potluck-server: speculative drafted=(\d+) accepted=(\d+) "
    r"accept-rate=([0-9]+\.[0-9]{3})"
)
with open(sys.argv[1]) as handle:
    matches = pattern.findall(handle.read())
assert matches, "missing speculative shutdown metrics"
drafted, accepted, rate_text = matches[-1]
drafted = int(drafted)
accepted = int(accepted)
rate = float(rate_text)
assert drafted > 0, drafted
assert 0 < accepted <= drafted, (accepted, drafted)
assert abs(rate - accepted / drafted) <= 0.001, (rate, accepted, drafted)
print(f"speculative drafted={drafted} accepted={accepted} accept-rate={rate_text}")
PY

start_server "${WORK}/ngram.log" "${NGRAM_PORT}" \
    --spec-type ngram-simple --spec-draft-n-max 4
curl -fsS "${AUTH_HEADER[@]}" -d "${NGRAM_REQUEST}" \
    "http://127.0.0.1:${NGRAM_PORT}/completion" >"${WORK}/ngram.json"
python3 - "${WORK}/ngram.json" "${WORK}/ngram.content" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    result = json.load(handle)
content = result.get("content")
assert isinstance(content, str) and content
with open(sys.argv[2], "w") as handle:
    handle.write(content)
PY
stop_server

if ! cmp -s "${WORK}/ngram-plain.content" "${WORK}/ngram.content"; then
    printf 'plain and ngram completion content differs\n' >&2
    diff -u "${WORK}/ngram-plain.content" "${WORK}/ngram.content" >&2 || true
    exit 1
fi
python3 - "${WORK}/ngram.log" <<'PY'
import re
import sys

pattern = re.compile(
    r"potluck-server: speculative drafted=(\d+) accepted=(\d+) "
    r"accept-rate=([0-9]+\.[0-9]{3})"
)
with open(sys.argv[1]) as handle:
    matches = pattern.findall(handle.read())
assert matches, "missing ngram shutdown metrics"
drafted, accepted, rate_text = matches[-1]
drafted = int(drafted)
accepted = int(accepted)
rate = float(rate_text)
assert drafted > 0, drafted
assert 0 < accepted <= drafted, (accepted, drafted)
assert abs(rate - accepted / drafted) <= 0.001, (rate, drafted, accepted)
print(f"ngram speculative drafted={drafted} accepted={accepted} accept-rate={rate_text}")
PY

printf 'SPECULATIVE TEST PASSED\n'
