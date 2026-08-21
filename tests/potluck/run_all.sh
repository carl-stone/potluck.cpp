#!/usr/bin/env bash
# Runs the whole potluck feature-parity + correctness suite.
# Each test cleans up its own workers. Exit code is nonzero if any fails.
set -uo pipefail

cd "$(dirname "$0")"

TESTS=(
    "test_chain.sh 1 24 127.0.0.1"
    "test_chain.sh 2 48 127.0.0.1"
    "test_chain.sh 4 32 127.0.0.1"
    "test_chain.sh 6 32 127.0.0.1"
    "test_sampler.sh 2 32 127.0.0.1"
    "test_gpu.sh 3 24 127.0.0.1"
    "test_sched.sh 2 32 127.0.0.1"
    "test_lp.sh 32 127.0.0.1"
    "test_remove.sh 24 127.0.0.1"
    "test_prefetch.sh 2 32 127.0.0.1"
    "test_ring.sh 16 127.0.0.1"
    "test_spec.sh 3 24 127.0.0.1"
    "test_batch.sh 3 24 127.0.0.1"
    "test_chat.sh 3 24 127.0.0.1"
    "test_server.sh 2 8 127.0.0.1"
    "test_vs_llama_cli.sh 2 16 127.0.0.1"
    "test_shard.sh 127.0.0.1"
)

failed=0
for t in "${TESTS[@]}"; do
    name="${t%% *}"
    printf '== %s ==\n' "${name}"
    if bash ${t} >/tmp/potluck_suite_${name}.log 2>&1; then
        tail -1 /tmp/potluck_suite_${name}.log
    else
        printf 'FAILED: %s\n' "${name}"
        tail -5 /tmp/potluck_suite_${name}.log >&2
        failed=1
    fi
    pkill -f potluck-wor[k]er 2>/dev/null || true
    sleep 1
done

if (( failed )); then
    printf 'SUITE FAILED\n' >&2
    exit 1
fi
printf 'SUITE PASSED (%s tests)\n' "${#TESTS[@]}"
