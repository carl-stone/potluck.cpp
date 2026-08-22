#!/usr/bin/env bash
# Shared resource guards for Potluck builds and model-backed checks.

potluck_safe_die() {
    printf 'potluck-safe: %s\n' "$*" >&2
    return 1
}

potluck_build_jobs() {
    if [[ -n "${POTLUCK_BUILD_JOBS:-}" ]]; then
        [[ "${POTLUCK_BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]] ||
            potluck_safe_die 'POTLUCK_BUILD_JOBS must be a positive integer' || return
        printf '%s\n' "${POTLUCK_BUILD_JOBS}"
        return
    fi

    local cores memory_bytes memory_gib jobs
    cores="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || printf '1\n')"
    case "$(uname -s)" in
        Darwin) memory_bytes="$(sysctl -n hw.memsize 2>/dev/null || printf '0\n')" ;;
        Linux)  memory_bytes="$(awk '/^MemTotal:/ { printf "%.0f\n", $2 * 1024; exit }' /proc/meminfo 2>/dev/null || printf '0\n')" ;;
        *)      memory_bytes=0 ;;
    esac
    memory_gib=$((memory_bytes / 1073741824))
    jobs=$((cores - 2))
    ((jobs > memory_gib / 3)) && jobs=$((memory_gib / 3))
    ((jobs > 6)) && jobs=6
    ((jobs < 1)) && jobs=1
    printf '%s\n' "${jobs}"
}

potluck_build_lock() {
    local lock_dir="${POTLUCK_BUILD_LOCK_DIR:-${TMPDIR:-/tmp}/potluck-build-${UID}.lock}"
    local stale_dir owner

    if mkdir "${lock_dir}" 2>/dev/null; then
        printf '%s\n' "${BASHPID:-$$}" > "${lock_dir}/pid"
        POTLUCK_ACTIVE_BUILD_LOCK="${lock_dir}"
        export POTLUCK_ACTIVE_BUILD_LOCK
        POTLUCK_ACTIVE_BUILD_OWNER="${BASHPID:-$$}"
        export POTLUCK_ACTIVE_BUILD_OWNER
        return
    fi

    owner="$(cat "${lock_dir}/pid" 2>/dev/null || true)"
    if [[ "${owner}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${owner}" 2>/dev/null; then
        potluck_safe_die "another build is active (PID ${owner})" || return
    fi

    stale_dir="${lock_dir}.stale.${BASHPID:-$$}"
    if ! mv "${lock_dir}" "${stale_dir}" 2>/dev/null; then
        potluck_safe_die 'another build acquired the lock' || return
    fi
    if ! mkdir "${lock_dir}" 2>/dev/null; then
        rm -rf "${stale_dir}"
        potluck_safe_die 'another build acquired the lock' || return
    fi
    printf '%s\n' "${BASHPID:-$$}" > "${lock_dir}/pid"
    rm -rf "${stale_dir}"
    POTLUCK_ACTIVE_BUILD_LOCK="${lock_dir}"
    export POTLUCK_ACTIVE_BUILD_LOCK
    POTLUCK_ACTIVE_BUILD_OWNER="${BASHPID:-$$}"
    export POTLUCK_ACTIVE_BUILD_OWNER
}

potluck_build_unlock() {
    local lock_dir="${POTLUCK_ACTIVE_BUILD_LOCK:-}"
    [[ -n "${lock_dir}" ]] || return 0
    rm -rf "${lock_dir}"
    unset POTLUCK_ACTIVE_BUILD_LOCK POTLUCK_ACTIVE_BUILD_OWNER
}

potluck_require_disk() {
    local path="$1"
    local required_gib="${2:-${POTLUCK_MIN_DISK_GIB:-10}}"
    local probe available_kib required_kib
    [[ "${required_gib}" =~ ^[0-9]+$ ]] ||
        potluck_safe_die 'required disk GiB must be a non-negative integer' || return

    probe="${path}"
    while [[ ! -e "${probe}" ]]; do
        [[ "${probe}" != "/" && "${probe}" != "." ]] || break
        probe="$(dirname "${probe}")"
    done
    available_kib="$(df -Pk "${probe}" | awk 'NR == 2 { print $4 }')"
    [[ "${available_kib}" =~ ^[0-9]+$ ]] ||
        potluck_safe_die "cannot determine free disk space for ${path}" || return
    required_kib=$((required_gib * 1024 * 1024))
    ((available_kib >= required_kib)) ||
        potluck_safe_die "need ${required_gib} GiB free for ${path}; only $((available_kib / 1024 / 1024)) GiB available" || return
}

potluck_available_memory_bytes() {
    case "$(uname -s)" in
        Darwin)
            vm_stat | awk '
                /page size of/ { gsub("[^0-9]", "", $8); page_size = $8 }
                /Pages free:/ { gsub("[^0-9]", "", $3); pages += $3 }
                /Pages inactive:/ { gsub("[^0-9]", "", $3); pages += $3 }
                /Pages speculative:/ { gsub("[^0-9]", "", $3); pages += $3 }
                END { printf "%.0f\n", pages * page_size }'
            ;;
        Linux)
            awk '/^MemAvailable:/ { printf "%.0f\n", $2 * 1024; exit }' /proc/meminfo
            ;;
        *)
            potluck_safe_die 'cannot determine available memory on this platform'
            ;;
    esac
}

potluck_require_memory_for_model() {
    local model="$1"
    local model_bytes available_bytes
    [[ "${POTLUCK_ALLOW_HEAVY:-0}" != "1" ]] || return 0
    [[ -f "${model}" ]] || potluck_safe_die "model not found: ${model}" || return

    case "$(uname -s)" in
        Darwin) model_bytes="$(stat -f %z "${model}")" ;;
        *)      model_bytes="$(stat -c %s "${model}")" ;;
    esac
    available_bytes="$(potluck_available_memory_bytes)" || return
    ((available_bytes * 10 >= model_bytes * 13)) ||
        potluck_safe_die "model needs 1.3x its $((model_bytes / 1024 / 1024)) MiB size; only $((available_bytes / 1024 / 1024)) MiB memory is available (set POTLUCK_ALLOW_HEAVY=1 for a supervised run)" || return
}

potluck_build() (
    local build_dir="$1"
    shift
    local jobs status=0
    jobs="$(potluck_build_jobs)" || exit
    potluck_require_disk "${build_dir}" "${POTLUCK_MIN_DISK_GIB:-10}" || exit
    potluck_build_lock || exit
    trap potluck_build_unlock EXIT
    trap 'exit 1' HUP INT TERM
    cmake --build "${build_dir}" --parallel "${jobs}" "$@" || status=$?
    potluck_build_unlock
    exit "${status}"
)
