#!/usr/bin/env bash
# Behavioral HTTP test for potluck-server.
# Usage: test_server.sh [n_workers] [n_predict] [host] [port]
set -euo pipefail

N_WORKERS="${1:-2}"
N_PREDICT="${2:-8}"
HOST="${3:-127.0.0.1}"
PORT="${4:-$((8100 + RANDOM % 400))}"

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BIN="${BIN:-${REPO}/build/bin}"
# shellcheck source=../../scripts/potluck-model.sh
source "${REPO}/scripts/potluck-model.sh"
MODEL="${MODEL:-$(potluck_model_path)}"

source "${REPO}/tests/potluck/test_helpers.sh"
if [[ ! -x "${BIN}/potluck-server" || ! -x "${BIN}/potluck-worker" || ! -x "${BIN}/llama-cli" ]]; then
    printf 'missing binaries (build potluck-server, potluck-worker, llama-cli first): %s\n' "${BIN}" >&2
    exit 2
fi
if [[ ! -f "${MODEL}" ]]; then
    printf 'missing model: %s\n' "${MODEL}" >&2
    exit 2
fi

WORK=$(mktemp -d "${TMPDIR:-/tmp}/potluck-server.XXXXXX")
SRV=""
SHUTDOWN_CURL_PID=""
SHUTDOWN_READER_PID=""

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
        if ! wait_pid_exit_bounded "${worker}" 10; then
            if ! terminate_pid_bounded "${worker}" 5; then
                status=1
            fi
        fi
    done
    if (( status != 0 )); then
        printf 'server or worker did not exit within the bounded shutdown wait\n' >&2
    else
        SRV=""
    fi
    return "${status}"
}
cleanup_background_requests() {
    local pid=""
    for pid in "${SHUTDOWN_CURL_PID}" "${SHUTDOWN_READER_PID}"; do
        if [[ -n "${pid}" ]]; then
            terminate_pid_bounded "${pid}" 5 || true
        fi
    done
    SHUTDOWN_CURL_PID=""
    SHUTDOWN_READER_PID=""
}
cleanup() {
    local rc=$?
    if (( rc != 0 )) && [[ -f "${WORK}/server.log" ]]; then
        printf '%s\n' '--- server.log ---' >&2
        cat "${WORK}/server.log" >&2
    fi
    cleanup_background_requests || true
    stop_server || true
    rm -rf "${WORK}" 2>/dev/null || true
    return "${rc}"
}
trap cleanup EXIT

POTLUCK_TRACE_PREFETCH=1 "${BIN}/potluck-server" -m "${MODEL}" --workers "${N_WORKERS}" --slots 2 --ubatch 1 --port "${PORT}" \
    >"${WORK}/server.log" 2>&1 &
SRV=$!
for _ in $(seq 1 120); do
    grep -q 'listening on http' "${WORK}/server.log" 2>/dev/null && break
    if ! kill -0 "${SRV}" 2>/dev/null; then
        sed -n '$p' "${WORK}/server.log" >&2
        printf 'server exited during startup\n' >&2
        exit 1
    fi
    sleep 1
done
grep -q 'listening on http' "${WORK}/server.log" || {
    sed -n '$p' "${WORK}/server.log" >&2
    printf 'server never became ready\n' >&2
    exit 1
}

auth_header=(-H 'Content-Type: application/json')
FRANCE=$(curl -fsS "${auth_header[@]}" -d '{"prompt":"The capital of France is","n_predict":8}' \
    "http://${HOST}:${PORT}/completion")
EIFFEL=$(curl -fsS "${auth_header[@]}" -d '{"prompt":"The Eiffel Tower is in","n_predict":8}' \
    "http://${HOST}:${PORT}/completion")
python3 - "${FRANCE}" "${EIFFEL}" <<'PY'
import json, sys
first, second = map(json.loads, sys.argv[1:])
for result in (first, second):
    assert isinstance(result.get("content"), str) and result["content"]
    assert result["finish_reason"] in {"stop", "length"}
    assert isinstance(result["n_predict"], int) and result["n_predict"] > 0
assert first["content"] != second["content"], (first, second)
PY

CHAT_REQ='{"messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":8,"reasoning_effort":"none"}'
CHAT=$(curl -fsS "${auth_header[@]}" -d "${CHAT_REQ}" "http://${HOST}:${PORT}/v1/chat/completions")
python3 - "${CHAT}" <<'PY'
import json, sys
result = json.loads(sys.argv[1])
assert result["object"] == "chat.completion"
assert result["choices"][0]["message"]["role"] == "assistant"
assert result["choices"][0]["message"]["content"]
assert result["choices"][0]["finish_reason"] in {"stop", "length"}
usage = result["usage"]
assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
PY
OMP_CHAT=$(curl -fsS "${auth_header[@]}" -d \
    '{"messages":[{"role":"user","content":"Say hello"}],"max_tokens":2,"reasoning_effort":"none","preserve_thinking":true,"chat_template_kwargs":{"preserve_thinking":true}}' \
    "http://${HOST}:${PORT}/v1/chat/completions")
python3 - "${OMP_CHAT}" <<'PY'
import json, sys
result = json.loads(sys.argv[1])
assert result["object"] == "chat.completion"
assert result["choices"][0]["message"]["role"] == "assistant"
PY
V1_COMPLETION=$(curl -fsS "${auth_header[@]}" \
    -d "{\"model\":\"$(basename "${MODEL}")\",\"prompt\":\"A short fact:\",\"max_completion_tokens\":4,\"n\":2,\"stop\":[\"\\n\"],\"temperature\":0,\"top_p\":1,\"top_k\":1,\"min_p\":0,\"presence_penalty\":0,\"frequency_penalty\":0,\"repeat_penalty\":1,\"repeat_last_n\":64,\"logprobs\":1}" \
    "http://${HOST}:${PORT}/v1/completions")
python3 - "${V1_COMPLETION}" <<'PY'
import json, sys
result = json.loads(sys.argv[1])
assert result["object"] == "text_completion"
assert len(result["choices"]) == 2
assert result["choices"][0]["finish_reason"] in {"stop", "length"}
assert isinstance(result["choices"][0]["text"], str)
logprobs = result["choices"][0]["logprobs"]
assert set(logprobs) == {"tokens", "token_logprobs", "top_logprobs", "text_offset"}
assert len(logprobs["tokens"]) == len(logprobs["token_logprobs"])
assert len(logprobs["tokens"]) == len(logprobs["top_logprobs"])
assert len(logprobs["tokens"]) == len(logprobs["text_offset"])
usage = result["usage"]
assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
PY
STOP_REGRESSION=$(curl -fsS "${auth_header[@]}" \
    -d "{\"model\":\"$(basename "${MODEL}")\",\"prompt\":\"The capital of France is\",\"max_completion_tokens\":64,\"stop\":[\"Paris\"],\"temperature\":0,\"top_p\":1,\"top_k\":1,\"logprobs\":2}" \
    "http://${HOST}:${PORT}/v1/completions")
python3 - "${STOP_REGRESSION}" <<'PY'
import json, sys
result = json.loads(sys.argv[1])
choice = result["choices"][0]
assert choice["finish_reason"] == "stop"
assert "Paris" not in choice["text"]
usage = result["usage"]
assert usage["completion_tokens"] > 0
logprobs = choice["logprobs"]
assert len(logprobs["tokens"]) <= usage["completion_tokens"]
assert usage["completion_tokens"] < 64
assert len(logprobs["tokens"]) == len(logprobs["token_logprobs"])
assert len(logprobs["tokens"]) == len(logprobs["top_logprobs"])
assert len(logprobs["tokens"]) == len(logprobs["text_offset"])
PY
PROPS=$(curl -fsS "http://${HOST}:${PORT}/props")
python3 - "${PROPS}" <<'PY'
import json, sys
props = json.loads(sys.argv[1])
assert props["model"]
assert props["ready"] is True
assert props["rebuilding"] is False
assert props["slot_count"] == 2
PY

TOOL_CHAT=$(curl -fsS "${auth_header[@]}" -d \
    '{"messages":[{"role":"user","content":"Say hello"}],"max_tokens":2,"temperature":0,"tools":[{"type":"function","function":{"name":"lookup","description":"Look up a value","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}}}],"tool_choice":"none","parallel_tool_calls":false}' \
    "http://${HOST}:${PORT}/v1/chat/completions")
python3 - "${TOOL_CHAT}" <<'PY'
import json, sys
result = json.loads(sys.argv[1])
assert result["object"] == "chat.completion"
assert result["choices"][0]["message"]["role"] == "assistant"
assert "usage" in result
PY

TOOL_REQUIRED=$(curl -fsS "${auth_header[@]}" -d \
    '{"messages":[{"role":"user","content":"What is the weather in Paris? Use the weather tool."}],"max_tokens":48,"temperature":0,"reasoning_effort":"none","tools":[{"type":"function","function":{"name":"lookup","description":"Look up a value","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}}},{"type":"function","function":{"name":"weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"tool_choice":{"type":"function","function":{"name":"weather"}}}' \
    "http://${HOST}:${PORT}/v1/chat/completions")
TOOL_CONTINUATION=$(python3 - "${TOOL_REQUIRED}" <<'PY'
import json, sys
result = json.loads(sys.argv[1])
choice = result["choices"][0]
assert choice["finish_reason"] == "tool_calls"
call = choice["message"]["tool_calls"][0]
assert call["id"].startswith("call_")
assert call["function"]["name"] == "weather"
assert json.loads(call["function"]["arguments"])["city"] == "Paris"
print(json.dumps({
    "messages": [
        {"role": "user", "content": "What is the weather in Paris? Use the weather tool."},
        choice["message"],
        {"role": "tool", "tool_call_id": call["id"], "content": "Sunny, 24 C"},
    ],
    "max_tokens": 32,
    "temperature": 0,
    "reasoning_effort": "none",
    "tools": [{
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }],
    "tool_choice": "none",
}))
PY
)
TOOL_FINAL=$(curl -fsS "${auth_header[@]}" -d "${TOOL_CONTINUATION}" \
    "http://${HOST}:${PORT}/v1/chat/completions")
python3 - "${TOOL_FINAL}" <<'PY'
import json, sys
choice = json.loads(sys.argv[1])["choices"][0]
assert choice["finish_reason"] == "stop"
assert "sunny" in choice["message"]["content"].lower()
PY

TOOL_STREAM=$(curl -fsS -N "${auth_header[@]}" -d \
    '{"messages":[{"role":"user","content":"What is the weather in Paris? Use the weather tool."}],"max_tokens":48,"temperature":0,"reasoning_effort":"none","stream":true,"tools":[{"type":"function","function":{"name":"weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"tool_choice":"required"}' \
    "http://${HOST}:${PORT}/v1/chat/completions")
python3 - "${TOOL_STREAM}" <<'PY'
import json, sys
lines = [
    line[6:]
    for line in sys.argv[1].splitlines()
    if line.startswith("data: ")
]
assert lines and lines[-1] == "[DONE]"
events = [json.loads(line) for line in lines[:-1]]
assert events
assert events[0]["choices"][0]["delta"].get("role") == "assistant"
finish_index = next(
    i
    for i, event in enumerate(events)
    for choice in event["choices"]
    if choice["finish_reason"] == "tool_calls"
)
tool_chunks = [
    (event_index, choice)
    for event_index, event in enumerate(events)
    for choice in event["choices"]
    if choice["delta"].get("tool_calls")
]
assert len(tool_chunks) > 1
assert any(
    choice["delta"]["tool_calls"][0].get("function", {}).get("arguments")
    for event_index, choice in tool_chunks
    if event_index < finish_index
)
call_id = None
name = None
arguments = ""
for _, choice in tool_chunks:
    for call in choice["delta"]["tool_calls"]:
        assert call["index"] == 0
        if "id" in call:
            call_id = call_id or call["id"]
            assert call["id"] == call_id
        function = call.get("function", {})
        if "name" in function:
            name = name or function["name"]
            assert function["name"] == name
        arguments += function.get("arguments", "")
assert call_id and call_id.startswith("call_")
assert name == "weather"
assert json.loads(arguments)["city"] == "Paris"
PY

TOOL_TRUNCATED_STREAM=$(curl -fsS -N "${auth_header[@]}" -d \
    '{"messages":[{"role":"user","content":"Call weather for Paris"}],"max_tokens":12,"temperature":0,"reasoning_effort":"none","stream":true,"tools":[{"type":"function","function":{"name":"weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"tool_choice":"required"}' \
    "http://${HOST}:${PORT}/v1/chat/completions")
python3 - "${TOOL_TRUNCATED_STREAM}" <<'PY'
import json, sys
lines = [
    line[6:]
    for line in sys.argv[1].splitlines()
    if line.startswith("data: ")
]
assert lines and lines[-1] == "[DONE]"
events = [json.loads(line) for line in lines[:-1]]
assert any(
    call.get("function", {}).get("name") == "weather"
    for event in events
    for choice in event["choices"]
    for call in choice["delta"].get("tool_calls", [])
)
assert any(
    choice["finish_reason"] == "tool_calls"
    for event in events
    for choice in event["choices"]
)
PY
TOOL_AUTO_STREAM=$(curl -fsS -N "${auth_header[@]}" -d \
    '{"messages":[{"role":"user","content":"Say hello without calling a tool."}],"max_tokens":8,"temperature":0,"reasoning_effort":"none","stream":true,"tools":[{"type":"function","function":{"name":"weather","description":"Get current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"tool_choice":"auto"}' \
    "http://${HOST}:${PORT}/v1/chat/completions")
python3 - "${TOOL_AUTO_STREAM}" <<'PY'
import json, sys
lines = [
    line[6:]
    for line in sys.argv[1].splitlines()
    if line.startswith("data: ")
]
assert lines and lines[-1] == "[DONE]"
events = [json.loads(line) for line in lines[:-1]]
assert events
assert events[0]["choices"][0]["delta"].get("role") == "assistant"
finish_index = next(
    i
    for i, event in enumerate(events)
    for choice in event["choices"]
    if choice["finish_reason"] in {"stop", "length"}
)
content_deltas = [
    choice["delta"].get("content", "")
    for event_index, event in enumerate(events)
    for choice in event["choices"]
    if event_index < finish_index
]
assert any(content_deltas)
assert "".join(content_deltas)
PY


TOO_MANY_STOPS_STATUS=$(curl -sS -o "${WORK}/too-many-stops.json" -w '%{http_code}' \
    "${auth_header[@]}" -d '{"prompt":"hello","stop":["1","2","3","4","5"]}' \
    "http://${HOST}:${PORT}/completion")
[[ "${TOO_MANY_STOPS_STATUS}" == 400 ]]
python3 - "${WORK}/too-many-stops.json" <<'PY'
import json, sys
error = json.load(open(sys.argv[1]))["error"]
assert error["type"] == "invalid_request_error"
assert error["message"].startswith("invalid stop:")
PY

N_TOO_LARGE_STATUS=$(curl -sS -o "${WORK}/n-too-large.json" -w '%{http_code}' \
    "${auth_header[@]}" -d '{"prompt":"hello","n":3}' "http://${HOST}:${PORT}/completion")
[[ "${N_TOO_LARGE_STATUS}" == 400 ]]
STREAM_N_STATUS=$(curl -sS -o "${WORK}/stream-n.json" -w '%{http_code}' \
    "${auth_header[@]}" -d '{"prompt":"hello","n":2,"stream":true}' \
    "http://${HOST}:${PORT}/v1/completions")
[[ "${STREAM_N_STATUS}" == 400 ]]
STREAM_LOGPROBS_STATUS=$(curl -sS -o "${WORK}/stream-logprobs.json" -w '%{http_code}' \
    "${auth_header[@]}" -d '{"prompt":"hello","stream":true,"logprobs":true}' \
    "http://${HOST}:${PORT}/v1/completions")
[[ "${STREAM_LOGPROBS_STATUS}" == 400 ]]

curl -fsS "${auth_header[@]}" \
    -d '{"messages":[{"role":"user","content":"A short fact about the Moon is"}],"max_tokens":4,"temperature":0,"seed":1,"reasoning_effort":"none"}' \
    "http://${HOST}:${PORT}/v1/chat/completions" >"${WORK}/concurrent-a.json" &
CONCURRENT_A=$!
curl -fsS "${auth_header[@]}" \
    -d '{"messages":[{"role":"user","content":"A short fact about Mars is"}],"max_tokens":4,"temperature":1,"seed":2,"reasoning_effort":"none"}' \
    "http://${HOST}:${PORT}/v1/chat/completions" >"${WORK}/concurrent-b.json" &
CONCURRENT_B=$!
wait "${CONCURRENT_A}"
wait "${CONCURRENT_B}"
python3 - "${WORK}/concurrent-a.json" "${WORK}/concurrent-b.json" <<'PY'
import json, sys
for path in sys.argv[1:]:
    result = json.load(open(path))
    assert result["object"] == "chat.completion"
    choice = result["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"]
    assert choice["finish_reason"] in {"stop", "length"}
    usage = result["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
PY

MISSING_STATUS=$(curl -sS -o "${WORK}/missing.json" -w '%{http_code}' \
    "${auth_header[@]}" -d '{}' "http://${HOST}:${PORT}/completion")
[[ "${MISSING_STATUS}" == 400 ]]
python3 - "${WORK}/missing.json" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
assert body["error"]["message"] == "missing prompt"
assert body["error"]["type"] == "invalid_request_error"
PY

BAD_JSON_STATUS=$(curl -sS -o "${WORK}/bad-json.json" -w '%{http_code}' \
    "${auth_header[@]}" --data-binary '{' "http://${HOST}:${PORT}/completion")
[[ "${BAD_JSON_STATUS}" == 400 ]]
python3 - "${WORK}/bad-json.json" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
assert body["error"]["message"].startswith("invalid JSON:")
assert body["error"]["type"] == "invalid_request_error"
PY

INVALID_NUMBER_STATUS=$(curl -sS -o "${WORK}/invalid-number.json" -w '%{http_code}' \
    "${auth_header[@]}" -d '{"prompt":"hello","n_predict":"eight"}' \
    "http://${HOST}:${PORT}/completion")
[[ "${INVALID_NUMBER_STATUS}" == 400 ]]
python3 - "${WORK}/invalid-number.json" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
assert body["error"]["message"].startswith("invalid n_predict:")
assert body["error"]["type"] == "invalid_request_error"
PY

INVALID_NEGATIVE_STATUS=$(curl -sS -o "${WORK}/negative-number.json" -w '%{http_code}' \
    "${auth_header[@]}" -d '{"prompt":"hello","n_predict":-1}' \
    "http://${HOST}:${PORT}/completion")
[[ "${INVALID_NEGATIVE_STATUS}" == 400 ]]
python3 - "${WORK}/negative-number.json" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
assert body["error"]["message"].startswith("invalid n_predict:")
assert body["error"]["type"] == "invalid_request_error"
PY

INVALID_LARGE_STATUS=$(curl -sS -o "${WORK}/large-number.json" -w '%{http_code}' \
    "${auth_header[@]}" -d '{"prompt":"hello","n_predict":4294967296}' \
    "http://${HOST}:${PORT}/completion")
[[ "${INVALID_LARGE_STATUS}" == 400 ]]
python3 - "${WORK}/large-number.json" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
assert body["error"]["message"].startswith("invalid n_predict:")
assert body["error"]["type"] == "invalid_request_error"
PY

INVALID_STREAM_STATUS=$(curl -sS -o "${WORK}/invalid-stream.json" -w '%{http_code}' \
    "${auth_header[@]}" -d '{"prompt":"hello","stream":"yes"}' \
    "http://${HOST}:${PORT}/completion")
[[ "${INVALID_STREAM_STATUS}" == 400 ]]
python3 - "${WORK}/invalid-stream.json" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
assert body["error"]["message"].startswith("invalid stream:")
assert body["error"]["type"] == "invalid_request_error"
PY
NOT_FOUND_STATUS=$(curl -sS -o "${WORK}/not-found.json" -w '%{http_code}' \
    "http://${HOST}:${PORT}/not-found")
[[ "${NOT_FOUND_STATUS}" == 404 ]]
python3 - "${WORK}/not-found.json" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
assert body["error"]["message"] == "not found"
assert body["error"]["type"] == "invalid_request_error"
PY

OPTIONS_STATUS=$(curl -sS -o /dev/null -w '%{http_code}' -X OPTIONS \
    "${auth_header[@]}" "http://${HOST}:${PORT}/completion")
[[ "${OPTIONS_STATUS}" == 204 ]]

HEALTH=$(curl -fsS "http://${HOST}:${PORT}/health")
python3 - "${HEALTH}" "${N_WORKERS}" <<'PY'
import json, sys
health, expected = json.loads(sys.argv[1]), int(sys.argv[2])
assert health["status"] == "ok"
assert health["loading"] is False
assert health["ready"] is True
assert health["rebuilding"] is False
actual = health["workers"]
assert 1 <= actual <= expected
assert len(health["slots"]) == 2
windows = sorted(health["windows"], key=lambda w: w["start"])
assert windows and windows[0]["start"] == 0
for previous, current in zip(windows, windows[1:]):
    assert current["start"] == previous["end"]
assert all(w["start"] < w["end"] for w in windows)
assert all(0 <= w["owner"] < actual for w in windows)
assert [w["index"] for w in windows] == list(range(len(windows)))
PY

MODELS=$(curl -fsS "http://${HOST}:${PORT}/v1/models")
python3 - "${MODELS}" "$(basename "${MODEL}")" <<'PY'
import json, sys
models, model = json.loads(sys.argv[1]), sys.argv[2]
assert models["object"] == "list"
assert models["data"] and models["data"][0]["id"] == model
PY
python3 - "${WORK}/server.log" <<'PY'
import re, sys
prefetched = {}
computed = []
for line_number, line in enumerate(open(sys.argv[1])):
    match = re.search(r"WORKER rank (\d+) prefetched window (\d+) \((\d+) bytes\)", line)
    if match:
        rank, window, byte_count = match.groups()
        assert int(byte_count) > 0, f"window {(rank, window)} reported zero prefetch bytes"
        prefetched.setdefault((rank, window), line_number)
    match = re.search(r"WORKER rank (\d+) computing window (\d+)", line)
    if match:
        computed.append((match.groups(), line_number))
assert computed, "no worker compute trace"
for window, line_number in computed:
    assert window in prefetched, f"window {window} computed without prefetch"
    assert prefetched[window] < line_number, f"window {window} prefetch followed compute"
PY


if [[ "${POTLUCK_TEST_WORKER_LOSS:-0}" == 1 ]]; then
    WORKER_PID="$(pgrep -P "${SRV}" -f '[p]otluck-worker' | sed -n '1p')"
    [[ -n "${WORKER_PID}" ]] || {
        printf 'worker-loss gate could not find a child worker\n' >&2
        exit 1
    }
    kill -STOP "${WORKER_PID}"
    RECOVERY_BODY="${WORK}/recovery.json"
    RECOVERY_STATUS=$(curl -sS -o "${RECOVERY_BODY}" -w '%{http_code}' \
        "${auth_header[@]}" -d '{"prompt":"recovery probe","n_predict":8}' \
        "http://${HOST}:${PORT}/completion")
    [[ "${RECOVERY_STATUS}" == 503 ]]
    python3 - "${RECOVERY_BODY}" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
error = body["error"]["message"]
assert body["error"]["type"] == "server_error"
assert "retry" in error.lower(), error
PY
    for _ in $(seq 1 60); do
        grep -q 'ring rebuild succeeded' "${WORK}/server.log" && break
        sleep 0.5
    done
    grep -q 'ring rebuild succeeded' "${WORK}/server.log"
    RECOVERY_HEALTH=$(curl -fsS "http://${HOST}:${PORT}/health")
    python3 - "${RECOVERY_HEALTH}" <<'PY'
import json, sys
health = json.loads(sys.argv[1])
assert health["status"] == "ok", health
assert health["workers"] >= 1, health
PY
fi

SSE=$(curl -fsS -N "${auth_header[@]}" -d \
    '{"messages":[{"role":"user","content":"Say hi"}],"max_tokens":3,"stream":true,"stream_options":{"include_usage":true},"reasoning_effort":"none"}' \
    "http://${HOST}:${PORT}/v1/chat/completions")
python3 - "${SSE}" <<'PY'
import json, sys
lines = [line[6:] for line in sys.argv[1].splitlines() if line.startswith("data: ")]
assert lines and lines[-1] == "[DONE]"
chunks = [json.loads(line) for line in lines[:-1]]
assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
finish = next(c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"] in {"stop", "length"})
assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
usage = chunks[-1]["usage"]
assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
PY
RAW_SSE=$(curl -fsS -N "${auth_header[@]}" -d \
    '{"prompt":"Say hi","n_predict":3,"stream":true}' \
    "http://${HOST}:${PORT}/completion")
python3 - "${RAW_SSE}" <<'PY'
import json, sys
lines = [line[6:] for line in sys.argv[1].splitlines() if line.startswith("data: ")]
assert lines and lines[-1] == "[DONE]"
chunks = [json.loads(line) for line in lines[:-1]]
assert chunks and "".join(c.get("content", "") for c in chunks)
PY


SHUTDOWN_PIPE="${WORK}/shutdown.pipe"
SHUTDOWN_READY="${WORK}/shutdown.ready"
SHUTDOWN_OUTPUT="${WORK}/shutdown.sse"
mkfifo "${SHUTDOWN_PIPE}" "${SHUTDOWN_READY}"
: >"${SHUTDOWN_OUTPUT}"
(
    first=1
    while IFS= read -r line; do
        printf '%s\n' "${line}" >>"${SHUTDOWN_OUTPUT}"
        if (( first )); then
            printf '%s\n' "${line}" >"${SHUTDOWN_READY}"
            first=0
        fi
    done
) <"${SHUTDOWN_PIPE}" &
SHUTDOWN_READER_PID=$!
curl -fsS -N "${auth_header[@]}" -d \
    '{"messages":[{"role":"user","content":"Write a long sentence about the Moon"}],"max_tokens":64,"stream":true,"reasoning_effort":"none"}' \
    "http://${HOST}:${PORT}/v1/chat/completions" >"${SHUTDOWN_PIPE}" &
SHUTDOWN_CURL_PID=$!
SHUTDOWN_FIRST=""
if ! IFS= read -r -t 30 SHUTDOWN_FIRST <"${SHUTDOWN_READY}"; then
    printf 'shutdown regression request never became active\n' >&2
    exit 1
fi
if [[ "${SHUTDOWN_FIRST}" != data:* ]]; then
    printf 'shutdown regression did not receive an SSE event\n' >&2
    exit 1
fi
if ! stop_server; then
    exit 1
fi
if ! wait_pid_exit_bounded "${SHUTDOWN_CURL_PID}" 15; then
    printf 'streaming request did not exit after server shutdown\n' >&2
    exit 1
fi
SHUTDOWN_CURL_PID=""
if ! wait_pid_exit_bounded "${SHUTDOWN_READER_PID}" 15; then
    printf 'shutdown stream reader did not exit\n' >&2
    exit 1
fi
SHUTDOWN_READER_PID=""
rm -f "${SHUTDOWN_PIPE}" "${SHUTDOWN_READY}" "${SHUTDOWN_OUTPUT}"
if [[ "${POTLUCK_SKIP_CLI_PARITY:-0}" == 1 ]]; then
    python3 - "${CHAT}" <<'PY'
import json, sys
chat_text = json.loads(sys.argv[1])["choices"][0]["message"]["content"]
assert chat_text
PY
    printf '  llama-cli text parity skipped (POTLUCK_SKIP_CLI_PARITY=1)\n'
else
    PROMPT='The capital of France is'
    CLI_OUT=$("${BIN}/llama-cli" -m "${MODEL}" -p "${PROMPT}" -n 8 --temp 0 --seed 1 -no-cnv -st \
        --no-display-prompt --chat-template-kwargs '{"enable_thinking": false}' --log-disable </dev/null 2>/dev/null)
    python3 - "${CHAT}" "${CLI_OUT}" "${PROMPT}" <<'PY'
import json, sys
chat, cli, prompt = sys.argv[1:]
chat_text = json.loads(chat)["choices"][0]["message"]["content"]
lines = cli.splitlines()
marker = next(i for i, line in enumerate(lines) if line.strip() == f"> {prompt}")
generated = []
for line in lines[marker + 1:]:
    if line.startswith("[ Prompt:"):
        break
    generated.append(line)
while generated and not generated[-1]:
    generated.pop()
expected = "\n".join(generated)
assert chat_text == expected, (chat_text, expected)
PY
fi

printf 'POTLUCK-SERVER TEST PASSED: %s workers, prompt/error/health/models/parity/SSE checks\n' "${N_WORKERS}"
