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
if [[ ! -x "${BIN}/potluck-server" || ! -x "${BIN}/potluck-worker" ||
      ! -x "${BIN}/llama-cli" ]]; then
    printf 'missing binaries (build potluck-server, potluck-worker, llama-cli first): %s\n' "${BIN}" >&2
    exit 2
fi
if [[ ! -f "${MODEL}" ]]; then
    printf 'missing model: %s\n' "${MODEL}" >&2
    exit 2
fi

WORK=$(mktemp -d "${TMPDIR:-/tmp}/potluck-server.XXXXXX")
SRV=""
FIXED_SPLIT_ACTIVE=0
SHUTDOWN_CURL_PID=""
SHUTDOWN_READER_PID=""
PREEMPT_PID=""
STREAM_ABORT_PID=""
NONSTREAM_ABORT_PID=""

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
    for pid in "${SHUTDOWN_CURL_PID}" "${SHUTDOWN_READER_PID}" \
               "${PREEMPT_PID}" "${STREAM_ABORT_PID}" "${NONSTREAM_ABORT_PID}"; do
        if [[ -n "${pid}" ]]; then
            terminate_pid_bounded "${pid}" 5 || true
        fi
    done
    SHUTDOWN_CURL_PID=""
    SHUTDOWN_READER_PID=""
    PREEMPT_PID=""
    STREAM_ABORT_PID=""
    NONSTREAM_ABORT_PID=""
}
cleanup() {
    local rc=$?
    if (( rc != 0 )) && [[ -f "${WORK}/server.log" ]]; then
        printf '%s\n' '--- server.log ---' >&2
        cat "${WORK}/server.log" >&2
    fi
    if (( rc != 0 )) && [[ -f "${WORK}/auth-server.log" ]]; then
        printf '%s\n' '--- auth-server.log ---' >&2
        cat "${WORK}/auth-server.log" >&2
    fi
    cleanup_background_requests || true
    stop_server || true
    rm -rf "${WORK}" 2>/dev/null || true
    return "${rc}"
}
trap cleanup EXIT

"${BIN}/potluck-server" -m "${MODEL}" --workers "${N_WORKERS}" --slots 4 --ubatch 1 --prefetch advise --port "${PORT}" \
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
grep -q "HTTP bound while loading ring at http://0.0.0.0:${PORT}" "${WORK}/server.log" || {
    sed -n '$p' "${WORK}/server.log" >&2
    printf 'server did not use the default LAN-visible bind address\n' >&2
    exit 1
}
grep -q 'listening on http' "${WORK}/server.log" || {
    sed -n '$p' "${WORK}/server.log" >&2
    printf 'server never became ready\n' >&2
    exit 1
}

collect_worker_pids() {
    WORKER_PIDS=()
    for child_pid in $(pgrep -P "${SRV}" 2>/dev/null || true); do
        worker_name="$(ps -o comm= -p "${child_pid}" 2>/dev/null || true)"
        if [[ "${worker_name}" == *potluck-worker* ]]; then
            WORKER_PIDS+=("${child_pid}")
        fi
    done
}

collect_worker_pids
if (( N_WORKERS > 1 && ${#WORKER_PIDS[@]} < 2 )); then
    FIXED_SPLIT_ACTIVE=1
    LAYER_WINDOW="$(python3 - "${WORK}/server.log" "${N_WORKERS}" <<'PY'
import re
import sys

log_path, worker_count = sys.argv[1], int(sys.argv[2])
ends = []
with open(log_path, encoding="utf-8", errors="replace") as stream:
    for line in stream:
        match = re.search(r"layers=\[\d+,(\d+)\)", line)
        if match:
            ends.append(int(match.group(1)))
if not ends:
    raise SystemExit("cannot derive model layer count from server log")
layer_count = max(ends)
if layer_count < worker_count or layer_count % worker_count:
    raise SystemExit(
        f"model layer count {layer_count} cannot split across {worker_count} workers"
    )
print(",".join([str(layer_count // worker_count)] * worker_count))
PY
)"
    stop_server
    "${BIN}/potluck-server" -m "${MODEL}" --workers "${N_WORKERS}" --slots 4 --ubatch 1 \
        --prefetch advise --layer-window "${LAYER_WINDOW}" --port "${PORT}" \
        >"${WORK}/server.log" 2>&1 &
    SRV=$!
    for _ in $(seq 1 120); do
        grep -q 'listening on http' "${WORK}/server.log" 2>/dev/null && break
        if ! kill -0 "${SRV}" 2>/dev/null; then
            sed -n '$p' "${WORK}/server.log" >&2
            printf 'server exited during split startup\n' >&2
            exit 1
        fi
        sleep 1
    done
    grep -q "HTTP bound while loading ring at http://0.0.0.0:${PORT}" "${WORK}/server.log" || {
        sed -n '$p' "${WORK}/server.log" >&2
        printf 'split server did not use the default LAN-visible bind address\n' >&2
        exit 1
    }
    grep -q 'listening on http' "${WORK}/server.log" || {
        sed -n '$p' "${WORK}/server.log" >&2
        printf 'split server never became ready\n' >&2
        exit 1
    }
    collect_worker_pids
fi

INSTALL_PREFIX="$(cd "${BIN}/.." && pwd)"
for root in "${INSTALL_PREFIX}" "${REPO}"; do
    SHARD_DIR="$(find "${root}" -type d -name '.potluck-shards' -print -quit)"
    if [[ -n "${SHARD_DIR}" ]]; then
        printf 'unexpected shard directory: %s\n' "${SHARD_DIR}" >&2
        exit 1
    fi
done

if (( ${#WORKER_PIDS[@]} == 0 )); then
    printf 'ring startup found no direct potluck-worker child\n' >&2
    exit 1
fi
if (( ${#WORKER_PIDS[@]} < 2 )); then
    printf 'ring startup found fewer than two potluck-worker children: %s\n' \
        "${#WORKER_PIDS[@]}" >&2
    exit 1
fi
case "$(uname -s)" in
    Darwin)
        MODEL_BYTES="$(stat -f %z "${MODEL}")"
        ;;
    *)
        MODEL_BYTES="$(stat -c %s "${MODEL}")"
        ;;
esac
[[ "${MODEL_BYTES}" =~ ^[0-9]+$ && "${MODEL_BYTES}" -gt 0 ]] || {
    printf 'could not read model size: %s\n' "${MODEL}" >&2
    exit 1
}

for worker_pid in "${WORKER_PIDS[@]}"; do
    MODEL_RESIDENT_BYTES="$(python3 - "${worker_pid}" "${MODEL}" <<'PY'
import re
import subprocess
import sys

pid, model = sys.argv[1], sys.argv[2]


def bytes_from(value, unit):
    scale = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}
    return int(float(value) * scale[unit])


if sys.platform == "darwin":
    output = subprocess.check_output(
        ["vmmap", pid],
        stderr=subprocess.DEVNULL,
        text=True,
    )
    total = 0
    for line in output.splitlines():
        if model not in line:
            continue
        match = re.search(r"\[\s*[0-9.]+[KMG]\s+([0-9.]+)([KMG])", line)
        if match:
            total += bytes_from(match.group(1), match.group(2))
else:
    header = re.compile(r"^[0-9a-f]+-[0-9a-f]+")
    total = 0
    in_model = False
    with open(f"/proc/{pid}/smaps", encoding="ascii") as stream:
        for line in stream:
            if header.match(line):
                in_model = line.rstrip().endswith(model)
            elif in_model:
                match = re.match(r"Rss:\s+([0-9]+)\s+kB", line)
                if match:
                    total += int(match.group(1)) * 1024

if total == 0:
    raise SystemExit(f"could not measure resident model mapping for worker {pid}")
print(total)
PY
)"
    [[ "${MODEL_RESIDENT_BYTES}" =~ ^[0-9]+$ ]] || {
        printf 'could not read resident model mapping for potluck-worker %s\n' \
            "${worker_pid}" >&2
        exit 1
    }
    if (( MODEL_RESIDENT_BYTES * 100 >= MODEL_BYTES * 75 )); then
        printf 'potluck-worker %s resident model mapping %s bytes is not below 75%% of model %s bytes\n' \
            "${worker_pid}" "${MODEL_RESIDENT_BYTES}" "${MODEL_BYTES}" >&2
        exit 1
    fi
done


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
ANTHROPIC=$(curl -fsS "${auth_header[@]}" -d \
    "{\"model\":\"$(basename "${MODEL}")\",\"max_tokens\":3,\"messages\":[{\"role\":\"user\",\"content\":\"Say hello\"}]}" \
    "http://${HOST}:${PORT}/v1/messages")
python3 - "${ANTHROPIC}" <<'PY'
import json, sys
result = json.loads(sys.argv[1])
assert result["type"] == "message"
assert result["role"] == "assistant"
assert isinstance(result["content"], list) and result["content"]
assert all(block["type"] in {"text", "thinking", "tool_use"} for block in result["content"])
assert result["stop_reason"] in {"end_turn", "max_tokens", "stop_sequence", "tool_use"}
assert result["usage"]["input_tokens"] > 0
assert result["usage"]["output_tokens"] > 0
PY
ANTHROPIC_BAD_STATUS=$(curl -sS -o "${WORK}/anthropic-invalid.json" -w '%{http_code}' \
    "${auth_header[@]}" -d \
    "{\"model\":\"$(basename "${MODEL}")\",\"max_tokens\":\"three\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello\"}]}" \
    "http://${HOST}:${PORT}/v1/messages")
[[ "${ANTHROPIC_BAD_STATUS}" == 400 ]]
python3 - "${WORK}/anthropic-invalid.json" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
assert body["error"]["type"] == "invalid_request_error", body
PY
ANTHROPIC_TOOL=$(curl -fsS "${auth_header[@]}" -d \
    "{\"model\":\"$(basename "${MODEL}")\",\"max_tokens\":48,\"temperature\":0,\"thinking\":{\"type\":\"disabled\"},\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris? Use the weather tool.\"}],\"tools\":[{\"name\":\"weather\",\"description\":\"Get current weather for a city\",\"input_schema\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}],\"tool_choice\":{\"type\":\"tool\",\"name\":\"weather\"}}" \
    "http://${HOST}:${PORT}/v1/messages")
python3 - "${ANTHROPIC_TOOL}" <<'PY'
import json, sys
result = json.loads(sys.argv[1])
assert result["stop_reason"] == "tool_use", result
tool = next(block for block in result["content"] if block["type"] == "tool_use")
assert tool["name"] == "weather"
assert tool["input"]["city"] == "Paris"
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
assert props["slot_count"] == 4
PY
ALPHA_ID="chat.alpha"
BETA_ID="chat.beta"
ALPHA_MARKER="POTLUCK_ALPHA_MARKER_7F3A"
BETA_MARKER="POTLUCK_BETA_MARKER_91C2"
ALPHA_FOLLOWUP_MARKER="POTLUCK_ALPHA_FOLLOWUP_4A9D"
BETA_FOLLOWUP_MARKER="POTLUCK_BETA_FOLLOWUP_5B0E"
ALPHA_ONE_BODY="${WORK}/alpha-one.json"
BETA_ONE_BODY="${WORK}/beta-one.json"
curl -fsS -o "${ALPHA_ONE_BODY}" "${auth_header[@]}" -H "X-Conversation-Id: ${ALPHA_ID}" -d \
    "{\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly ${ALPHA_MARKER} and nothing else.\"}],\"max_tokens\":32,\"temperature\":0,\"seed\":123,\"reasoning_effort\":\"none\"}" \
    "http://${HOST}:${PORT}/v1/chat/completions" &
ALPHA_ONE_PID=$!
curl -fsS -o "${BETA_ONE_BODY}" "${auth_header[@]}" -H "X-Conversation-Id: ${BETA_ID}" -d \
    "{\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly ${BETA_MARKER} and nothing else.\"}],\"max_tokens\":32,\"temperature\":0,\"seed\":123,\"reasoning_effort\":\"none\"}" \
    "http://${HOST}:${PORT}/v1/chat/completions" &
BETA_ONE_PID=$!
wait "${ALPHA_ONE_PID}"
wait "${BETA_ONE_PID}"
ALPHA_ONE="$(<"${ALPHA_ONE_BODY}")"
BETA_ONE="$(<"${BETA_ONE_BODY}")"
ALPHA_TWO=$(curl -fsS "${auth_header[@]}" -H "X-Conversation-Id: ${ALPHA_ID}" -d \
    "{\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly ${ALPHA_MARKER} and nothing else.\"},{\"role\":\"assistant\",\"content\":\"${ALPHA_MARKER}\"},{\"role\":\"user\",\"content\":\"Reply with exactly ${ALPHA_FOLLOWUP_MARKER} and nothing else.\"}],\"max_tokens\":32,\"temperature\":0,\"seed\":123,\"reasoning_effort\":\"none\"}" \
    "http://${HOST}:${PORT}/v1/chat/completions")
BETA_TWO=$(curl -fsS "${auth_header[@]}" -H "X-Conversation-Id: ${BETA_ID}" -d \
    "{\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly ${BETA_MARKER} and nothing else.\"},{\"role\":\"assistant\",\"content\":\"${BETA_MARKER}\"},{\"role\":\"user\",\"content\":\"Reply with exactly ${BETA_FOLLOWUP_MARKER} and nothing else.\"}],\"max_tokens\":32,\"temperature\":0,\"seed\":123,\"reasoning_effort\":\"none\"}" \
    "http://${HOST}:${PORT}/v1/chat/completions")
python3 - "${ALPHA_ONE}" "${BETA_ONE}" "${ALPHA_TWO}" "${BETA_TWO}" \
    "${ALPHA_MARKER}" "${BETA_MARKER}" \
    "${ALPHA_FOLLOWUP_MARKER}" "${BETA_FOLLOWUP_MARKER}" <<'PY'
import json, sys
alpha_one, beta_one, alpha_two, beta_two = sys.argv[1:5]
alpha_marker, beta_marker, alpha_followup, beta_followup = sys.argv[5:]
for value, expected, other in (
    (alpha_one, alpha_marker, beta_marker),
    (beta_one, beta_marker, alpha_marker),
    (alpha_two, alpha_followup, beta_followup),
    (beta_two, beta_followup, alpha_followup),
):
    result = json.loads(value)
    content = result["choices"][0]["message"]["content"]
    assert content == expected, result
    assert other not in content, result
PY
CONVERSATION_HEALTH=$(curl -fsS "http://${HOST}:${PORT}/health")
python3 - "${CONVERSATION_HEALTH}" "${ALPHA_ID}" "${BETA_ID}" <<'PY'
import json, sys
health = json.loads(sys.argv[1])
ids = set(sys.argv[2:])
slots = [slot for slot in health["slots"] if slot["conversation"] in ids]
assert len(slots) == 2, health
assert {slot["conversation"] for slot in slots} == ids
assert slots[0]["index"] != slots[1]["index"], slots
PY
STATELESS=$(curl -fsS "${auth_header[@]}" -d \
    '{"prompt":"The capital of France is","n_predict":2}' \
    "http://${HOST}:${PORT}/completion")
python3 - "${STATELESS}" <<'PY'
import json, sys
result = json.loads(sys.argv[1])
assert result.get("content")
PY
STATELESS_HEALTH=$(curl -fsS "http://${HOST}:${PORT}/health")
python3 - "${STATELESS_HEALTH}" "${ALPHA_ID}" "${BETA_ID}" <<'PY'
import json, sys
health = json.loads(sys.argv[1])
expected = set(sys.argv[2:])
bound = {slot["conversation"] for slot in health["slots"] if slot["conversation"]}
assert bound == expected, health
PY
INVALID_CONVERSATION_STATUS=$(curl -sS -o "${WORK}/invalid-conversation.json" -w '%{http_code}' \
    "${auth_header[@]}" -H 'X-Conversation-Id: bad id' \
    -d '{"prompt":"hello","n_predict":2}' "http://${HOST}:${PORT}/completion")
[[ "${INVALID_CONVERSATION_STATUS}" == 400 ]]
python3 - "${WORK}/invalid-conversation.json" <<'PY'
import json, sys
error = json.load(open(sys.argv[1]))["error"]
assert error["type"] == "invalid_request_error"
assert error["message"].startswith("invalid X-Conversation-Id:")
PY
BLANK_CONVERSATION_STATUS=$(python3 - "${HOST}" "${PORT}" "${WORK}/blank-conversation.json" <<'PY'
import socket, sys

host, port, output = sys.argv[1], int(sys.argv[2]), sys.argv[3]
body = b'{"prompt":"hello","n_predict":2}'
request = (
    f"POST /completion HTTP/1.1\r\nHost: {host}:{port}\r\n"
    "Content-Type: application/json\r\nX-Conversation-Id:\r\n"
    f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
).encode() + body
with socket.create_connection((host, port), timeout=10) as connection:
    connection.sendall(request)
    response = b""
    while True:
        chunk = connection.recv(4096)
        if not chunk:
            break
        response += chunk
status = int(response.split(b" ", 2)[1])
payload = response.split(b"\r\n\r\n", 1)[1]
open(output, "wb").write(payload)
print(status)
PY
)
[[ "${BLANK_CONVERSATION_STATUS}" == 400 ]]
python3 - "${WORK}/blank-conversation.json" <<'PY'
import json, sys
error = json.load(open(sys.argv[1]))["error"]
assert error["type"] == "invalid_request_error"
assert error["message"].startswith("invalid X-Conversation-Id:")
PY
LONG_CONVERSATION_ID="$(python3 - <<'PY'
print("a" * 129)
PY
)"
LONG_CONVERSATION_STATUS=$(curl -sS -o "${WORK}/long-conversation.json" -w '%{http_code}' \
    "${auth_header[@]}" -H "X-Conversation-Id: ${LONG_CONVERSATION_ID}" \
    -d '{"prompt":"hello","n_predict":2}' "http://${HOST}:${PORT}/completion")
[[ "${LONG_CONVERSATION_STATUS}" == 400 ]]
CONVERSATION_N_STATUS=$(curl -sS -o "${WORK}/conversation-n.json" -w '%{http_code}' \
    "${auth_header[@]}" -H "X-Conversation-Id: ${ALPHA_ID}" \
    -d '{"prompt":"hello","n":2}' "http://${HOST}:${PORT}/completion")
[[ "${CONVERSATION_N_STATUS}" == 400 ]]
python3 - "${WORK}/conversation-n.json" <<'PY'
import json, sys
error = json.load(open(sys.argv[1]))["error"]
assert error["type"] == "invalid_request_error"
assert error["message"].startswith("unsupported combination:")
PY
python3 - "${WORK}/server.log" "${ALPHA_ID}" "${BETA_ID}" <<'PY'
import re, sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
pattern = re.compile(r"conversation (\S+) slot (\d+) seq (-?\d+) turn (\d+) prompt (\d+)")
for conversation in sys.argv[2:]:
    entries = [pattern.search(line) for line in text
               if f"conversation {conversation} slot" in line]
    entries = [entry for entry in entries if entry]
    assert len(entries) >= 2, conversation
    first = entries[:2]
    assert {entry.group(1) for entry in first} == {conversation}
    assert len({entry.group(2) for entry in first}) == 1, first
    assert len({entry.group(3) for entry in first}) == 1, first
    assert [int(entry.group(4)) for entry in first] == [1, 2], first
    assert int(first[1].group(5)) > int(first[0].group(5)), first
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

curl -fsS -N "${auth_header[@]}" \
    -d '{"messages":[{"role":"user","content":"A short fact about the Moon is"}],"max_tokens":4,"temperature":0,"seed":1,"stream":true,"reasoning_effort":"none"}' \
    "http://${HOST}:${PORT}/v1/chat/completions" >"${WORK}/concurrent-a.sse" &
CONCURRENT_A=$!
curl -fsS -N "${auth_header[@]}" \
    -d '{"messages":[{"role":"user","content":"A short fact about Mars is"}],"max_tokens":4,"temperature":1,"seed":2,"stream":true,"reasoning_effort":"none"}' \
    "http://${HOST}:${PORT}/v1/chat/completions" >"${WORK}/concurrent-b.sse" &
CONCURRENT_B=$!
curl -fsS -N "${auth_header[@]}" \
    -d '{"messages":[{"role":"user","content":"A short fact about Jupiter is"}],"max_tokens":4,"temperature":0,"seed":3,"stream":true,"reasoning_effort":"none"}' \
    "http://${HOST}:${PORT}/v1/chat/completions" >"${WORK}/concurrent-c.sse" &
CONCURRENT_C=$!
curl -fsS -N "${auth_header[@]}" \
    -d '{"messages":[{"role":"user","content":"A short fact about Venus is"}],"max_tokens":4,"temperature":1,"seed":4,"stream":true,"reasoning_effort":"none"}' \
    "http://${HOST}:${PORT}/v1/chat/completions" >"${WORK}/concurrent-d.sse" &
CONCURRENT_D=$!
wait "${CONCURRENT_A}"
wait "${CONCURRENT_B}"
wait "${CONCURRENT_C}"
wait "${CONCURRENT_D}"
python3 - "${WORK}/concurrent-a.sse" "${WORK}/concurrent-b.sse" \
    "${WORK}/concurrent-c.sse" "${WORK}/concurrent-d.sse" <<'PY'
import json, sys
for path in sys.argv[1:]:
    lines = [
        line[6:]
        for line in open(path, encoding="utf-8").read().splitlines()
        if line.startswith("data: ")
    ]
    assert lines and lines[-1] == "[DONE]", path
    chunks = [json.loads(line) for line in lines[:-1]]
    assert chunks, path
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant", path
    content = "".join(
        chunk["choices"][0]["delta"].get("content", "")
        for chunk in chunks if chunk["choices"]
    )
    assert content, path
    assert any(
        chunk["choices"][0]["finish_reason"] in {"stop", "length"}
        for chunk in chunks if chunk["choices"]
    ), path
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
python3 - "${HEALTH}" "${N_WORKERS}" "${FIXED_SPLIT_ACTIVE}" <<'PY'
import json, sys
health, expected, fixed_split = (
    json.loads(sys.argv[1]), int(sys.argv[2]), bool(int(sys.argv[3]))
)
assert health["status"] == "ok"
assert health["loading"] is False
assert health["ready"] is True
assert health["rebuilding"] is False
actual = health["workers"]
assert 1 <= actual <= expected
assert len(health["slots"]) == 4
windows = sorted(health["windows"], key=lambda w: w["index"])
assert windows and windows[0]["start"] == 0
assert [w["index"] for w in windows] == list(range(len(windows)))
for previous, current in zip(windows, windows[1:]):
    assert current["start"] == previous["end"]
assert all(w["start"] < w["end"] for w in windows)
assert all(0 <= w["owner"] < actual for w in windows)
assert len(windows) % actual == 0
round_count = len(windows) // actual
assert round_count >= (1 if fixed_split else 2)
owners_by_round = []
for round_index in range(round_count):
    group = windows[round_index * actual:(round_index + 1) * actual]
    for previous, current in zip(group, group[1:]):
        assert current["start"] == previous["end"]
    owners = [window["owner"] for window in group]
    assert len(set(owners)) == actual
    assert sorted(owners) == list(range(actual))
    owners_by_round.append(owners)
assert all(owners == owners_by_round[0] for owners in owners_by_round[1:])
assert len(windows) > actual or fixed_split
PY

MODELS=$(curl -fsS "http://${HOST}:${PORT}/v1/models")
python3 - "${MODELS}" "$(basename "${MODEL}")" <<'PY'
import json, sys
models, model = json.loads(sys.argv[1]), sys.argv[2]
assert models["object"] == "list"
assert models["data"] and models["data"][0]["id"] == model
PY
python3 - "${WORK}/server.log" "${HEALTH}" "${FIXED_SPLIT_ACTIVE}" <<'PY'
import json
import re
import sys

log_path, health_text, fixed_split_text = sys.argv[1:]
health = json.loads(health_text)
fixed_split = bool(int(fixed_split_text))
worker_count = int(health["workers"])
windows = sorted(health["windows"], key=lambda w: w["index"])
assert worker_count > 0
assert windows and windows[0]["start"] == 0
assert [w["index"] for w in windows] == list(range(len(windows)))
for previous, current in zip(windows, windows[1:]):
    assert current["start"] == previous["end"]
assert all(w["start"] < w["end"] for w in windows)
assert all(0 <= w["owner"] < worker_count for w in windows)
assert len(windows) % worker_count == 0
round_count = len(windows) // worker_count
assert round_count >= (1 if fixed_split else 2)

round_for_window = {}
owners_by_round = []
for round_index in range(round_count):
    group = windows[round_index * worker_count:(round_index + 1) * worker_count]
    owners = [window["owner"] for window in group]
    assert len(set(owners)) == worker_count
    assert sorted(owners) == list(range(worker_count))
    owners_by_round.append(owners)
    for window in group:
        round_for_window[window["index"]] = round_index
assert all(owners == owners_by_round[0] for owners in owners_by_round[1:])

owned_by_rank = {
    rank: [window["index"] for window in windows if window["owner"] == rank]
    for rank in range(worker_count)
}
assert all(owned_by_rank.values())
next_owned = {}
for rank, owned in owned_by_rank.items():
    for position, window in enumerate(owned):
        next_owned[window] = owned[(position + 1) % len(owned)]

trace_re = re.compile(
    r"PRP seq=(\d+) window=(\d+) round=(\d+) rank=(\d+) "
    r"event=([a-z_]+) bytes=(\d+)(?:\s|$)"
)
allowed_events = {
    "receive", "compute_begin", "compute_end", "send",
    "prefetch_begin", "prefetch_end",
    "startup_prefetch_begin", "startup_prefetch_end",
}
events = []
with open(log_path) as log:
    for line_number, line in enumerate(log, 1):
        if "PRP " not in line:
            continue
        match = trace_re.search(line)
        assert match, f"malformed PRP trace at line {line_number}: {line.rstrip()}"
        seq, window, round_index, rank, event, byte_count = match.groups()
        event_record = {
            "seq": int(seq),
            "window": int(window),
            "round": int(round_index),
            "rank": int(rank),
            "event": event,
            "bytes": int(byte_count),
            "line": line_number,
        }
        assert event in allowed_events
        assert 0 <= event_record["window"] < len(windows)
        assert event_record["rank"] == windows[event_record["window"]]["owner"]
        assert event_record["round"] == round_for_window[event_record["window"]]
        assert event_record["bytes"] >= 0
        events.append(event_record)

startup = [event for event in events if event["seq"] == 0]
assert startup
startup_streams = {}
for event in startup:
    assert event["event"] in {"startup_prefetch_begin", "startup_prefetch_end"}
    startup_streams.setdefault((event["rank"], event["window"]), []).append(event)
for stream in startup_streams.values():
    assert [event["event"] for event in stream] == [
        "startup_prefetch_begin", "startup_prefetch_end"
    ]
runtime = [event for event in events if event["seq"] > 0]
assert runtime
streams = {}
for event in runtime:
    streams.setdefault((event["seq"], event["rank"]), []).append(event)

for rank in range(worker_count):
    receive_sequences = [
        event["seq"] for event in runtime
        if event["rank"] == rank and event["event"] == "receive"
    ]
    assert receive_sequences == sorted(receive_sequences)

sequences = sorted({event["seq"] for event in runtime})
assert sequences
for seq in sequences:
    sequence_events = [event for event in runtime if event["seq"] == seq]
    received = [event for event in sequence_events if event["event"] == "receive"]
    assert {event["window"] for event in received} == set(range(len(windows)))
    observed_rounds = sorted({event["round"] for event in received})
    assert observed_rounds == list(range(round_count))
    for round_index in observed_rounds:
        round_received = [
            event for event in received if event["round"] == round_index
        ]
        assert len(round_received) == worker_count
        assert sorted(event["rank"] for event in round_received) == list(range(worker_count))
        expected = windows[
            round_index * worker_count:(round_index + 1) * worker_count
        ]
        assert {event["window"] for event in round_received} == {
            window["index"] for window in expected
        }

for (seq, rank), stream in streams.items():
    expected_windows = owned_by_rank[rank]
    receives = [event["window"] for event in stream if event["event"] == "receive"]
    assert receives == expected_windows
    compute_begin = {
        event["window"]: event for event in stream if event["event"] == "compute_begin"
    }
    assert set(compute_begin) == set(expected_windows)
    position = 0
    for current_position, current_window in enumerate(expected_windows):
        assert position + 4 <= len(stream)
        block = stream[position:position + 4]
        assert [event["event"] for event in block] == [
            "receive", "compute_begin", "compute_end", "send"
        ]
        assert all(event["window"] == current_window for event in block)
        assert block[0]["line"] < block[1]["line"] < block[2]["line"] < block[3]["line"]
        position += 4

        target_window = next_owned[current_window]
        if position < len(stream) and stream[position]["event"] == "prefetch_begin":
            assert position + 1 < len(stream)
            prefetch = stream[position:position + 2]
            assert [event["event"] for event in prefetch] == [
                "prefetch_begin", "prefetch_end"
            ]
            assert all(event["window"] == target_window for event in prefetch)
            assert prefetch[0]["line"] > block[3]["line"]
            assert prefetch[0]["line"] < prefetch[1]["line"]
            if target_window != expected_windows[0]:
                assert prefetch[1]["line"] < compute_begin[target_window]["line"]
            else:
                later = [candidate for candidate in sequences
                         if candidate > seq and (candidate, rank) in streams]
                if later:
                    later_begin = next(
                        event for event in streams[(later[0], rank)]
                        if event["event"] == "compute_begin" and
                        event["window"] == target_window
                    )
                    assert prefetch[1]["line"] < later_begin["line"]
            position += 2
        elif target_window != expected_windows[0]:
            raise AssertionError(
                f"rank {rank} sequence {seq} did not prefetch window {target_window}"
            )
    assert position == len(stream)
PY


PREEMPT_ID="chat.preempt"
PREEMPT_FIRST_BODY="${WORK}/preempt-first.json"
PREEMPT_FIRST_STATUS_FILE="${WORK}/preempt-first.status"
curl -sS -o "${PREEMPT_FIRST_BODY}" -w '%{http_code}' \
    "${auth_header[@]}" -H "X-Conversation-Id: ${PREEMPT_ID}" \
    -d '{"prompt":"Write a long continuation about the history of the Moon and its geology","n_predict":2048,"temperature":0}' \
    "http://${HOST}:${PORT}/completion" >"${PREEMPT_FIRST_STATUS_FILE}" 2>"${WORK}/preempt-first.curl" &
PREEMPT_PID=$!
PREEMPT_SLOT=""
for _ in $(seq 1 120); do
    CURRENT_HEALTH="$(curl -fsS "http://${HOST}:${PORT}/health" 2>/dev/null || true)"
    if PREEMPT_SLOT="$(python3 - "${CURRENT_HEALTH}" "${PREEMPT_ID}" 2>/dev/null <<'PY'
import json, sys
health = json.loads(sys.argv[1])
conversation = sys.argv[2]
for slot in health["slots"]:
    if slot["conversation"] == conversation and slot["state"] != "free":
        print(slot["index"])
        raise SystemExit(0)
raise SystemExit(1)
PY
)"; then
        break
    fi
    sleep 0.1
done
[[ -n "${PREEMPT_SLOT}" ]] || {
    printf 'preemption request never became active\n' >&2
    exit 1
}
PREEMPT_SECOND_STATUS=$(curl -sS -o "${WORK}/preempt-second.json" -w '%{http_code}' \
    "${auth_header[@]}" -H "X-Conversation-Id: ${PREEMPT_ID}" \
    -d '{"prompt":"Give one short fact about the Moon","n_predict":4,"temperature":0}' \
    "http://${HOST}:${PORT}/completion")
[[ "${PREEMPT_SECOND_STATUS}" == 200 ]]
if ! wait_pid_exit_bounded "${PREEMPT_PID}" 120; then
    printf 'preemption request did not exit after replacement\n' >&2
    terminate_pid_bounded "${PREEMPT_PID}" 5 || true
fi
PREEMPT_PID=""
[[ "$(<"${PREEMPT_FIRST_STATUS_FILE}")" == 503 ]]
python3 - "${PREEMPT_FIRST_BODY}" "${WORK}/preempt-second.json" <<'PY'
import json, sys
first = json.load(open(sys.argv[1]))
second = json.load(open(sys.argv[2]))
assert first["error"]["message"] == "request cancelled", first
assert first["error"]["type"] == "server_error", first
assert second.get("content"), second
PY
grep -q "conversation ${PREEMPT_ID} preempted slot ${PREEMPT_SLOT}" "${WORK}/server.log"
PREEMPT_AFTER_HEALTH=$(curl -fsS "http://${HOST}:${PORT}/health")
python3 - "${PREEMPT_AFTER_HEALTH}" "${PREEMPT_ID}" "${PREEMPT_SLOT}" <<'PY'
import json, sys
health = json.loads(sys.argv[1])
conversation, expected = sys.argv[2], int(sys.argv[3])
slots = [slot for slot in health["slots"] if slot["conversation"] == conversation]
assert len(slots) == 1, health
assert slots[0]["index"] == expected, slots

PY
if [[ "${POTLUCK_TEST_WORKER_LOSS:-0}" == 1 ]]; then
    RECOVERY_ALPHA_ID="chat.recovery.alpha"
    RECOVERY_BETA_ID="chat.recovery.beta"
    RECOVERY_ALPHA_BODY="${WORK}/recovery-alpha.json"
    RECOVERY_BETA_BODY="${WORK}/recovery-beta.json"
    curl -fsS -o "${RECOVERY_ALPHA_BODY}" "${auth_header[@]}" \
        -H "X-Conversation-Id: ${RECOVERY_ALPHA_ID}" \
        -d '{"prompt":"Remember the recovery alpha marker","n_predict":4,"temperature":0}' \
        "http://${HOST}:${PORT}/completion" &
    RECOVERY_ALPHA_PID=$!
    curl -fsS -o "${RECOVERY_BETA_BODY}" "${auth_header[@]}" \
        -H "X-Conversation-Id: ${RECOVERY_BETA_ID}" \
        -d '{"prompt":"Remember the recovery beta marker","n_predict":4,"temperature":0}' \
        "http://${HOST}:${PORT}/completion" &
    RECOVERY_BETA_PID=$!
    wait "${RECOVERY_ALPHA_PID}"
    wait "${RECOVERY_BETA_PID}"
    python3 "${RECOVERY_ALPHA_BODY}" "${RECOVERY_BETA_BODY}" <<'PY'
import json, sys
for path in sys.argv[1:]:
    assert json.load(open(path)).get("content"), path
PY
    REBUILD_CONVERSATION_HEALTH=$(curl -fsS "http://${HOST}:${PORT}/health")
    REBUILD_SUCCESSES_BEFORE=$(grep -c 'ring rebuild succeeded' "${WORK}/server.log" || true)
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
        REBUILD_SUCCESSES_NOW=$(grep -c 'ring rebuild succeeded' "${WORK}/server.log" || true)
        if (( REBUILD_SUCCESSES_NOW > REBUILD_SUCCESSES_BEFORE )); then
            break
        fi
        sleep 0.5
    done
    REBUILD_SUCCESSES_NOW=$(grep -c 'ring rebuild succeeded' "${WORK}/server.log" || true)
    (( REBUILD_SUCCESSES_NOW > REBUILD_SUCCESSES_BEFORE ))
    RECOVERY_HEALTH=$(curl -fsS "http://${HOST}:${PORT}/health")
    python3 - "${RECOVERY_HEALTH}" <<'PY'
import json, sys
health = json.loads(sys.argv[1])
assert health["status"] == "ok", health
assert health["workers"] == 2, health
assert len(health["windows"]) == 2, health
PY
    for _ in $(seq 1 20); do
        if ! kill -0 "${WORKER_PID}" 2>/dev/null; then
            break
        fi
        sleep 0.1
    done
    if kill -0 "${WORKER_PID}" 2>/dev/null; then
        printf 'worker-loss gate left the stopped worker alive: %s\n' "${WORKER_PID}" >&2
        exit 1
    fi
    RECOVERY_ALPHA=$(curl -fsS "${auth_header[@]}" \
        -H "X-Conversation-Id: ${RECOVERY_ALPHA_ID}" \
        -d '{"prompt":"Remember the recovery alpha marker. Continue the recovery alpha conversation","n_predict":4,"temperature":0}' \
        "http://${HOST}:${PORT}/completion")
    RECOVERY_BETA=$(curl -fsS "${auth_header[@]}" \
        -H "X-Conversation-Id: ${RECOVERY_BETA_ID}" \
        -d '{"prompt":"Remember the recovery beta marker. Continue the recovery beta conversation","n_predict":4,"temperature":0}' \
        "http://${HOST}:${PORT}/completion")
    python3 - "${RECOVERY_ALPHA}" "${RECOVERY_BETA}" <<'PY'
import json, sys
for value in sys.argv[1:]:
    assert json.loads(value).get("content"), value
PY
    RECOVERY_HEALTH_WITH_CONVERSATIONS=$(curl -fsS "http://${HOST}:${PORT}/health")
    python3 - "${REBUILD_CONVERSATION_HEALTH}" "${RECOVERY_HEALTH_WITH_CONVERSATIONS}" \
        "${RECOVERY_ALPHA_ID}" "${RECOVERY_BETA_ID}" <<'PY'
import json, sys
before = json.loads(sys.argv[1])
after = json.loads(sys.argv[2])
def slot_for(health, conversation):
    matches = [slot["index"] for slot in health["slots"]
               if slot["conversation"] == conversation]
    assert len(matches) == 1, (conversation, health)
    return matches[0]
for conversation in sys.argv[3:]:
    assert slot_for(before, conversation) == slot_for(after, conversation)
PY
    python3 - "${WORK}/server.log" "${RECOVERY_ALPHA_ID}" "${RECOVERY_BETA_ID}" <<'PY'
import re, sys
pattern = re.compile(r"conversation (\S+) slot (\d+) seq (-?\d+) turn (\d+) prompt (\d+)")
lines = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
for conversation in sys.argv[2:]:
    entries = [pattern.search(line) for line in lines
               if f"conversation {conversation} slot" in line]
    entries = [entry for entry in entries if entry]
    assert len(entries) == 2, entries
    assert [int(entry.group(4)) for entry in entries] == [1, 2], entries
    assert len({entry.group(2) for entry in entries}) == 1, entries
    assert len({entry.group(3) for entry in entries}) == 1, entries
    assert int(entries[1].group(5)) > int(entries[0].group(5)), entries
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
ANTHROPIC_SSE=$(curl -fsS -N "${auth_header[@]}" -d \
    "{\"model\":\"$(basename "${MODEL}")\",\"max_tokens\":3,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"Say hello\"}]}" \
    "http://${HOST}:${PORT}/v1/messages")
python3 - "${ANTHROPIC_SSE}" <<'PY'
import json, sys
events = []
current = None
for line in sys.argv[1].splitlines():
    if line.startswith("event: "):
        current = line[7:]
    elif line.startswith("data: ") and current:
        events.append((current, json.loads(line[6:])))
        current = None
names = [name for name, _ in events]
assert names[0] == "message_start"
assert "content_block_start" in names
assert "content_block_delta" in names
assert "content_block_stop" in names
assert names[-2:] == ["message_delta", "message_stop"], names
message_start = events[0][1]["message"]
assert message_start["type"] == "message"
assert message_start["role"] == "assistant"
assert events[-2][1]["delta"]["stop_reason"] in {"end_turn", "max_tokens", "stop_sequence", "tool_use"}
PY



STREAM_ABORT_ID="abort.stream"
STREAM_ABORT_OUTPUT="${WORK}/stream-abort.sse"
curl -sS -N "${auth_header[@]}" -H "X-Conversation-Id: ${STREAM_ABORT_ID}" -d \
    '{"prompt":"Write a long sentence about the Moon and its history","n_predict":512,"stream":true}' \
    "http://${HOST}:${PORT}/completion" >"${STREAM_ABORT_OUTPUT}" 2>"${WORK}/stream-abort.curl" &
STREAM_ABORT_PID=$!
STREAM_ABORT_SLOT=""
for _ in $(seq 1 120); do
    if grep -q '^data: ' "${STREAM_ABORT_OUTPUT}" 2>/dev/null; then
        CURRENT_HEALTH="$(curl -fsS "http://${HOST}:${PORT}/health" 2>/dev/null || true)"
        if STREAM_ABORT_SLOT="$(python3 - "${CURRENT_HEALTH}" "${STREAM_ABORT_ID}" 2>/dev/null <<'PY'
import json, sys
health = json.loads(sys.argv[1])
conversation = sys.argv[2]
for slot in health["slots"]:
    if slot["conversation"] == conversation and slot["state"] != "free":
        print(slot["index"])
        raise SystemExit(0)
raise SystemExit(1)
PY
)"; then
            break
        fi
    fi
    sleep 0.1
done
[[ -n "${STREAM_ABORT_SLOT}" ]] || {
    printf 'stream abort request never became active\n' >&2
    exit 1
}
kill "${STREAM_ABORT_PID}" 2>/dev/null || true
wait "${STREAM_ABORT_PID}" || true
STREAM_ABORT_PID=""
STREAM_ABORT_FREE=0
for _ in $(seq 1 120); do
    ABORT_HEALTH="$(curl -fsS "http://${HOST}:${PORT}/health" 2>/dev/null || true)"
    if python3 - "${ABORT_HEALTH}" <<'PY'
import json, sys
health = json.loads(sys.argv[1])
assert all(slot["state"] == "free" for slot in health["slots"])
PY
    then
        STREAM_ABORT_FREE=1
        break
    fi
    sleep 0.25
done
[[ "${STREAM_ABORT_FREE}" == 1 ]]
STREAM_ABORT_FOLLOWUP=$(curl -fsS "${auth_header[@]}" \
    -H "X-Conversation-Id: ${STREAM_ABORT_ID}" \
    -d '{"prompt":"Give one short fact about the Moon","n_predict":4,"temperature":0}' \
    "http://${HOST}:${PORT}/completion")
python3 - "${STREAM_ABORT_FOLLOWUP}" "${HOST}" <<'PY'
import json, sys
assert json.loads(sys.argv[1]).get("content")
PY
STREAM_ABORT_AFTER=$(curl -fsS "http://${HOST}:${PORT}/health")
python3 - "${STREAM_ABORT_AFTER}" "${STREAM_ABORT_ID}" "${STREAM_ABORT_SLOT}" <<'PY'
import json, sys
health = json.loads(sys.argv[1])
conversation, expected = sys.argv[2], int(sys.argv[3])
slots = [slot for slot in health["slots"] if slot["conversation"] == conversation]
assert len(slots) == 1, health
assert slots[0]["index"] == expected, slots
PY

NONSTREAM_ABORT_ID="abort.nonstream"
NONSTREAM_ABORT_BODY="${WORK}/nonstream-abort.json"
NONSTREAM_ABORT_STATUS_FILE="${WORK}/nonstream-abort.status"
curl -sS --max-time 2 -o "${NONSTREAM_ABORT_BODY}" -w '%{http_code}' \
    "${auth_header[@]}" -H "X-Conversation-Id: ${NONSTREAM_ABORT_ID}" \
    -d '{"prompt":"Write a very long detailed history of the Moon and its geology","n_predict":2048,"temperature":0}' \
    "http://${HOST}:${PORT}/completion" >"${NONSTREAM_ABORT_STATUS_FILE}" 2>"${WORK}/nonstream-abort.curl" &
NONSTREAM_ABORT_PID=$!
NONSTREAM_ABORT_SLOT=""
for _ in $(seq 1 40); do
    CURRENT_HEALTH="$(curl -fsS "http://${HOST}:${PORT}/health" 2>/dev/null || true)"
    if NONSTREAM_ABORT_SLOT="$(python3 - "${CURRENT_HEALTH}" "${NONSTREAM_ABORT_ID}" 2>/dev/null <<'PY'
import json, sys
health = json.loads(sys.argv[1])
conversation = sys.argv[2]
for slot in health["slots"]:
    if slot["conversation"] == conversation and slot["state"] != "free":
        print(slot["index"])
        raise SystemExit(0)
raise SystemExit(1)
PY
)"; then
        break
    fi
    sleep 0.1
done
[[ -n "${NONSTREAM_ABORT_SLOT}" ]] || {
    printf 'non-stream abort request never became active\n' >&2
    exit 1
}
NONSTREAM_ABORT_RC=0
wait "${NONSTREAM_ABORT_PID}" || NONSTREAM_ABORT_RC=$?
NONSTREAM_ABORT_PID=""
[[ "${NONSTREAM_ABORT_RC}" -ne 0 ]]
NONSTREAM_ABORT_FREE=0
for _ in $(seq 1 120); do
    ABORT_HEALTH="$(curl -fsS "http://${HOST}:${PORT}/health" 2>/dev/null || true)"
    if python3 - "${ABORT_HEALTH}" <<'PY'
import json, sys
health = json.loads(sys.argv[1])
assert all(slot["state"] == "free" for slot in health["slots"])
PY
    then
        NONSTREAM_ABORT_FREE=1
        break
    fi
    sleep 0.25
done
[[ "${NONSTREAM_ABORT_FREE}" == 1 ]]
NONSTREAM_ABORT_FOLLOWUP=$(curl -fsS "${auth_header[@]}" \
    -H "X-Conversation-Id: ${NONSTREAM_ABORT_ID}" \
    -d '{"prompt":"Give one short fact about the Moon","n_predict":4,"temperature":0}' \
    "http://${HOST}:${PORT}/completion")
python3 - "${NONSTREAM_ABORT_FOLLOWUP}" <<'PY'
import json, sys
assert json.loads(sys.argv[1]).get("content")
PY
NONSTREAM_ABORT_AFTER=$(curl -fsS "http://${HOST}:${PORT}/health")
python3 - "${NONSTREAM_ABORT_AFTER}" "${NONSTREAM_ABORT_ID}" "${NONSTREAM_ABORT_SLOT}" <<'PY'
import json, sys
health = json.loads(sys.argv[1])
conversation, expected = sys.argv[2], int(sys.argv[3])
slots = [slot for slot in health["slots"] if slot["conversation"] == conversation]
assert len(slots) == 1, health
assert slots[0]["index"] == expected, slots
PY
python3 - "${WORK}/server.log" <<'PY'
import sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
markers = [
    line for line in text.splitlines()
    if "client disconnected; cancelled request " in line
]
assert len(markers) == 2, markers
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
AUTH_PORT=$((PORT + 1))
AUTH_HOST="${POTLUCK_TEST_AUTH_HOST:-127.0.0.1}"
AUTH_KEY="${POTLUCK_TEST_API_KEY:-potluck-smoke-key}"
AUTH_ORIGIN="${POTLUCK_TEST_CORS_ORIGIN:-https://potluck.test}"
AUTH_BASE="http://${AUTH_HOST}:${AUTH_PORT}"
"${BIN}/potluck-server" -m "${MODEL}" --workers "${N_WORKERS}" \
    --slots 2 --ubatch 1 --host "${AUTH_HOST}" --port "${AUTH_PORT}" \
    --api-key "${AUTH_KEY}" --cors-origin "${AUTH_ORIGIN}" \
    >"${WORK}/auth-server.log" 2>&1 &
SRV=$!
for _ in $(seq 1 120); do
    grep -q 'listening on http' "${WORK}/auth-server.log" 2>/dev/null && break
    if ! kill -0 "${SRV}" 2>/dev/null; then
        sed -n '$p' "${WORK}/auth-server.log" >&2
        printf 'authenticated server exited during startup\n' >&2
        exit 1
    fi
    sleep 1
done
grep -q "HTTP bound while loading ring at http://${AUTH_HOST}:${AUTH_PORT}" \
    "${WORK}/auth-server.log" || {
    sed -n '$p' "${WORK}/auth-server.log" >&2
    printf 'authenticated server did not use its explicit bind address\n' >&2
    exit 1
}
grep -q 'listening on http' "${WORK}/auth-server.log" || {
    sed -n '$p' "${WORK}/auth-server.log" >&2
    printf 'authenticated server never became ready\n' >&2
    exit 1
}

AUTH_JSON_HEADERS=(-H 'Content-Type: application/json')
for auth_case in missing wrong; do
    if [[ "${auth_case}" == missing ]]; then
        AUTH_CHECK_HEADERS=("${AUTH_JSON_HEADERS[@]}")
    else
        AUTH_CHECK_HEADERS=("${AUTH_JSON_HEADERS[@]}" -H 'Authorization: Bearer wrong-key')
    fi
    for endpoint in health models unknown; do
        case "${endpoint}" in
            health) path="/health" ;;
            models) path="/v1/models" ;;
            unknown) path="/not-found" ;;
        esac
        response_file="${WORK}/auth-${auth_case}-${endpoint}.json"
        status=$(curl -sS -o "${response_file}" -w '%{http_code}' \
            "${AUTH_CHECK_HEADERS[@]}" "${AUTH_BASE}${path}")
        [[ "${status}" == 401 ]]
        python3 - "${response_file}" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
assert body["error"]["type"] == "authentication_error", body
assert body["error"]["param"] is None, body
assert body["error"]["code"] is None, body
PY
    done
    response_file="${WORK}/auth-${auth_case}-completion.json"
    status=$(curl -sS -o "${response_file}" -w '%{http_code}' \
        "${AUTH_CHECK_HEADERS[@]}" -X POST -d '{}' "${AUTH_BASE}/completion")
    [[ "${status}" == 401 ]]
    python3 - "${response_file}" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
assert body["error"]["type"] == "authentication_error", body
PY
    response_file="${WORK}/auth-${auth_case}-options.json"
    status=$(curl -sS -o "${response_file}" -w '%{http_code}' \
        "${AUTH_CHECK_HEADERS[@]}" -X OPTIONS "${AUTH_BASE}/completion")
    [[ "${status}" == 401 ]]
    python3 - "${response_file}" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
assert body["error"]["type"] == "authentication_error", body
PY
done
ALT_STATUS=$(curl -sS -o "${WORK}/auth-alternate-header.json" -w '%{http_code}' \
    "${AUTH_JSON_HEADERS[@]}" -H "X-Api-Key: ${AUTH_KEY}" "${AUTH_BASE}/health")
[[ "${ALT_STATUS}" == 401 ]]

AUTH_HEADERS=("${AUTH_JSON_HEADERS[@]}" -H "Authorization: Bearer ${AUTH_KEY}")
AUTH_COMPLETION=$(curl -fsS "${AUTH_HEADERS[@]}" -d '{"prompt":"The capital of France is","n_predict":2}' \
    "${AUTH_BASE}/completion")
python3 - "${AUTH_COMPLETION}" <<'PY'
import json, sys
body = json.loads(sys.argv[1])
assert isinstance(body["content"], str) and body["content"]
assert body["finish_reason"] in {"stop", "length"}
PY
AUTH_ANTHROPIC=$(curl -fsS "${AUTH_JSON_HEADERS[@]}" -H "x-api-key: ${AUTH_KEY}" -d \
    "{\"model\":\"$(basename "${MODEL}")\",\"max_tokens\":2,\"messages\":[{\"role\":\"user\",\"content\":\"Say hello\"}]}" \
    "${AUTH_BASE}/v1/messages")
python3 - "${AUTH_ANTHROPIC}" <<'PY'
import json, sys
body = json.loads(sys.argv[1])
assert body["type"] == "message"
assert body["role"] == "assistant"
assert body["content"]
PY
AUTH_NOT_FOUND_STATUS=$(curl -sS -o "${WORK}/auth-valid-not-found.json" -w '%{http_code}' \
    "${AUTH_HEADERS[@]}" "${AUTH_BASE}/not-found")
[[ "${AUTH_NOT_FOUND_STATUS}" == 404 ]]

curl -sS -D "${WORK}/cors-absent.headers" -o /dev/null \
    "${AUTH_HEADERS[@]}" "${AUTH_BASE}/health"
! grep -qi '^Access-Control-Allow-Origin:' "${WORK}/cors-absent.headers"
! grep -qi '^Access-Control-Allow-Credentials:' "${WORK}/cors-absent.headers"
curl -sS -D "${WORK}/cors-wrong.headers" -o /dev/null \
    "${AUTH_HEADERS[@]}" -H 'Origin: https://other-potluck.test' "${AUTH_BASE}/health"
! grep -qi '^Access-Control-Allow-Origin:' "${WORK}/cors-wrong.headers"
! grep -qi '^Access-Control-Allow-Credentials:' "${WORK}/cors-wrong.headers"
curl -sS -D "${WORK}/cors-exact.headers" -o /dev/null \
    "${AUTH_HEADERS[@]}" -H "Origin: ${AUTH_ORIGIN}" "${AUTH_BASE}/health"
grep -Fqi "Access-Control-Allow-Origin: ${AUTH_ORIGIN}" "${WORK}/cors-exact.headers"
! grep -qi '^Access-Control-Allow-Credentials:' "${WORK}/cors-exact.headers"
AUTH_OPTIONS_STATUS=$(curl -sS -D "${WORK}/cors-options.headers" -o /dev/null \
    -w '%{http_code}' -X OPTIONS -H "Origin: ${AUTH_ORIGIN}" \
    "${AUTH_HEADERS[@]}" "${AUTH_BASE}/completion")
[[ "${AUTH_OPTIONS_STATUS}" == 204 ]]
grep -Fqi "Access-Control-Allow-Origin: ${AUTH_ORIGIN}" "${WORK}/cors-options.headers"
grep -Fqi 'Access-Control-Allow-Headers: Content-Type, Authorization' \
    "${WORK}/cors-options.headers"
! grep -qi '^Access-Control-Allow-Credentials:' "${WORK}/cors-options.headers"
if ! stop_server; then
    exit 1
fi

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

printf 'POTLUCK-SERVER TEST PASSED: %s workers, prompt/error/health/models/auth/CORS/parity/SSE checks\n' "${N_WORKERS}"
