#!/usr/bin/env bash
# Copy the worker binary and one layer-window shard to a remote host.
# Usage: deploy.sh HOST SHARD_INDEX SHARD_COUNT
set -euo pipefail

if [[ $# -ne 3 ]]; then
    printf 'usage: %s HOST SHARD_INDEX SHARD_COUNT\n' "$0" >&2
    exit 2
fi

HOST=$1
INDEX=$2
COUNT=$3
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BIN="${BIN:-${REPO}/build/bin}"
MODEL="${MODEL:-${REPO}/models/Qwen3.5-0.8B-Q4_0.gguf}"
SHARD_DIR="${SHARD_DIR:-$(dirname "${MODEL}")/shards}"
REMOTE_DIR="${REMOTE_DIR:-~/potluck}"

if [[ ! "${INDEX}" =~ ^[0-9]+$ || ! "${COUNT}" =~ ^[1-9][0-9]*$ || ${INDEX} -ge ${COUNT} ]]; then
    printf 'invalid shard index/count: %s/%s\n' "${INDEX}" "${COUNT}" >&2
    exit 2
fi
[[ -x "${BIN}/potluck-worker" ]] || { printf 'missing worker binary: %s\n' "${BIN}/potluck-worker" >&2; exit 2; }

base=$(basename "${MODEL}")
stem=${base%.*}
shard="${SHARD_DIR}/${stem}.potluck-${INDEX}of${COUNT}.gguf"
[[ -s "${shard}" ]] || { printf 'missing shard: %s\n' "${shard}" >&2; exit 2; }

printf 'deploy: %s -> %s (shard %s/%s)\n' "${HOST}" "${REMOTE_DIR}" "${INDEX}" "${COUNT}"
ssh -o BatchMode=yes "${HOST}" "mkdir -p ${REMOTE_DIR}"
rsync -az --checksum "${BIN}/potluck-worker" "${shard}" "${HOST}:${REMOTE_DIR}/"
printf 'deploy complete: %s\n' "$(basename "${shard}")"
