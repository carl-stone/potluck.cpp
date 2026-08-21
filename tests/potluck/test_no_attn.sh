#!/usr/bin/env bash
# Regression coverage for a worker window containing only recurrent layers.
# Qwen3.5 layers [0,3) have no attention cache; this must still serve tokens.
set -euo pipefail

N_PREDICT="${1:-8}"
HOST="${2:-127.0.0.1}"

POTLUCK_BOUNDS=0,3,24 \
    bash "$(dirname "${BASH_SOURCE[0]}")/test_chain.sh" 2 "${N_PREDICT}" "${HOST}"

printf 'potluck no-attention boundary test passed\n'
