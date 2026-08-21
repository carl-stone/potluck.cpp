#!/usr/bin/env bash
# Shared acceptance helpers.

# CUDA and x86 CPU reductions can choose a different near-tie token than the
# split reference. Accept that only when the caller opts in and the log proves
# that the run produced complete token streams rather than a transport error.
potluck_accept_backend_variance() {
    local log="$1"
    [[ "${POTLUCK_ALLOW_NUMERICAL_MISMATCH:-0}" == 1 ]] || return 1
    grep -qE 'head: (MISMATCH|RING MISMATCH):' "${log}" || return 1
    grep -qE 'ggml_cuda_(init|graph)' "${log}" || return 1

    local reference_count chain_count
    reference_count="$(sed -n 's/^head: reference_tokens(\([0-9][0-9]*\)).*/\1/p' "${log}" | tail -1)"
    chain_count="$(sed -n 's/^head: \(chain\|ring\)_tokens(\([0-9][0-9]*\)).*/\2/p' "${log}" | tail -1)"
    [[ -n "${reference_count}" && -n "${chain_count}" ]] || return 1
    [[ "${reference_count}" == "${chain_count}" ]] || return 1
    grep -qE '^head: generated text \([1-9][0-9]* tokens\):' "${log}" || return 1

    printf 'accepted CUDA numerical variance: complete streams, equal token counts\n' >&2
    return 0
}
