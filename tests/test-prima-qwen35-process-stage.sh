#!/bin/sh
# Drive the two-process stage test: one binary as "worker" (binds) and one as
# "head" (connects). CTest cannot launch a second process for a single test, so
# this script runs both roles and tears the worker down.
#
# Usage: test-prima-qwen35-process-stage.sh <binary> <model> [port]
set -e

BIN="$1"
MODEL="$2"
PORT="${3:-39271}"

"$BIN" worker "$MODEL" "$PORT" &
WK=$!
trap 'kill "$WK" 2>/dev/null || true' EXIT

# Give the worker a beat to bind its listener and load its split. The head
# retries the connection for up to 60 s, so this sleep is just a courtesy.
sleep 1

"$BIN" head "$MODEL" "$PORT"
RC=$?

kill "$WK" 2>/dev/null || true
wait "$WK" 2>/dev/null || true
exit "$RC"
