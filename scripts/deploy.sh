#!/usr/bin/env bash
# Deploy a staged worker payload to a discovered SSH device.
#
# The device selects its platform from uname. The matching payload is copied
# without a compiler or source checkout and installed flat into ~/potluck.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

fail() {
    printf 'potluck-deploy: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
usage: deploy.sh --target SSH_TARGET [--dist-dir DIR] [--ssh-port PORT]

The target platform is discovered over SSH. The matching payload is selected
from DIR (default: dist/) and installed into ~/potluck on the target.
EOF
    exit 2
}

shell_quote() {
    local value="$1"
    value="${value//\'/\'\\\'\'}"
    printf "'%s'" "${value}"
}

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | cut -d' ' -f1
    else
        fail 'need sha256sum or shasum to verify files'
    fi
}

valid_target() {
    [[ "$1" != -* && "$1" =~ ^([A-Za-z0-9._-]+@)?[A-Za-z0-9._-]+$ ]]
}

payload_for_platform() {
    case "$1" in
        'Darwin arm64'*|'Darwin aarch64'*) printf 'mac-arm64\n' ;;
        'Linux x86_64'*|'Linux amd64'*) printf 'linux-x86_64\n' ;;
        'Linux aarch64'*) printf 'linux-aarch64\n' ;;
        'Linux arm64'*) printf 'linux-arm64\n' ;;
        *) return 1 ;;
    esac
}

target=""
ssh_port=""
dist_dir="${POTLUCK_DIST_DIR:-${REPO}/dist}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            [[ $# -ge 2 && -n "$2" ]] || usage
            target="$2"
            shift 2
            ;;
        --target=*)
            target="${1#*=}"
            [[ -n "${target}" ]] || usage
            shift
            ;;
        --ssh-port)
            [[ $# -ge 2 && -n "$2" ]] || usage
            ssh_port="$2"
            shift 2
            ;;
        --ssh-port=*)
            ssh_port="${1#*=}"
            [[ -n "${ssh_port}" ]] || usage
            shift
            ;;
        --dist-dir)
            [[ $# -ge 2 && -n "$2" ]] || usage
            dist_dir="$2"
            shift 2
            ;;
        --dist-dir=*)
            dist_dir="${1#*=}"
            [[ -n "${dist_dir}" ]] || usage
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            usage
            ;;
    esac
done

[[ -n "${target}" ]] || fail '--target is required'
valid_target "${target}" || fail "invalid SSH target: ${target}"
[[ -d "${dist_dir}" ]] || fail "distribution directory does not exist: ${dist_dir}"
command -v ssh >/dev/null 2>&1 || fail 'required command not found: ssh'
command -v scp >/dev/null 2>&1 || fail 'required command not found: scp'

if [[ -n "${ssh_port}" ]]; then
    [[ "${ssh_port}" =~ ^[1-9][0-9]*$ && "${ssh_port}" -le 65535 ]] ||
        fail "invalid SSH port: ${ssh_port}"
fi
ssh_args=(-o BatchMode=yes)
scp_args=(-q -o BatchMode=yes)
if [[ -n "${ssh_port}" ]]; then
    ssh_args+=(-p "${ssh_port}")
    scp_args+=(-P "${ssh_port}")
fi

remote_platform="$(ssh "${ssh_args[@]}" "${target}" 'uname -sm 2>/dev/null')" ||
    fail "cannot inspect platform on ${target}"
payload_name="$(payload_for_platform "${remote_platform}")" ||
    fail "unsupported remote platform: ${remote_platform}"
payload_dir="${dist_dir}/${payload_name}"
[[ -d "${payload_dir}" ]] ||
    fail "missing ${payload_name} payload: ${payload_dir}"
build_id="${payload_dir}/potluck-build-id"
[[ -f "${build_id}" && ! -L "${build_id}" ]] ||
    fail "payload has no potluck-build-id: ${payload_dir}"

# Validate the staged manifest before any bytes leave the head. The platform
# record prevents a valid payload from being installed under the wrong ABI.
declare -a payload_files
payload_files=()
payload_platform=""
commit_seen=0
while read -r field value extra; do
    [[ -n "${field:-}" ]] || continue
    case "${field}" in
        platform)
            [[ -n "${value:-}" && -z "${extra:-}" && -z "${payload_platform}" ]] ||
                fail 'invalid payload platform record'
            payload_platform="${value}"
            ;;
        commit)
            [[ -n "${value:-}" && -z "${extra:-}" && "${commit_seen}" -eq 0 ]] ||
                fail 'invalid payload commit record'
            commit_seen=1
            ;;
        *)
            checksum="${field}"
            name="${value:-}"
            [[ -z "${extra:-}" && "${checksum}" =~ ^[[:xdigit:]]{64}$ ]] ||
                fail 'invalid payload checksum record'
            [[ "${name}" =~ ^[A-Za-z0-9._+][A-Za-z0-9._+-]*$ &&
               "${name}" != potluck-build-id &&
               "${name}" != potluck-deploy.sha256 &&
               -f "${payload_dir}/${name}" &&
               ! -L "${payload_dir}/${name}" ]] ||
                fail "payload checksum names an invalid file: ${name}"
            actual="$(sha256_file "${payload_dir}/${name}")"
            [[ "${actual}" == "${checksum}" ]] ||
                fail "payload checksum mismatch: ${name}"
            for known in "${payload_files[@]}"; do
                [[ "${known}" != "${name}" ]] ||
                    fail "duplicate payload checksum: ${name}"
            done
            payload_files+=("${name}")
            ;;
    esac
done < "${build_id}"
[[ "${commit_seen}" -eq 1 ]] || fail 'payload manifest has no commit record'
[[ -n "${payload_platform}" && "${payload_platform}" == "${payload_name}" ]] ||
    fail "payload platform does not match ${payload_name}: ${payload_platform:-missing}"
for required in potluck-node potluck-worker; do
    found=0
    for known in "${payload_files[@]}"; do
        [[ "${known}" == "${required}" ]] && found=1
    done
    [[ "${found}" == 1 ]] ||
        fail "payload manifest does not checksum ${required}"
done
for name in "${payload_files[@]}"; do
    [[ -f "${payload_dir}/${name}" && ! -L "${payload_dir}/${name}" ]] ||
        fail "payload missing ${name}"
done
[[ -x "${payload_dir}/potluck-node" && -x "${payload_dir}/potluck-worker" ]] ||
    fail 'payload worker binaries are not executable'

local_stage="$(mktemp -d "${TMPDIR:-/tmp}/potluck-deploy.XXXXXX")"
cleanup() {
    if [[ -n "${local_stage}" && -d "${local_stage}" ]]; then
        rm -rf "${local_stage}"
    fi
}
trap cleanup EXIT

for name in "${payload_files[@]}"; do
    cp -f "${payload_dir}/${name}" "${local_stage}/${name}"
done
cp -f "${build_id}" "${local_stage}/potluck-build-id"

checksums="${local_stage}/potluck-deploy.sha256"
: > "${checksums}"
for path in "${local_stage}"/*; do
    name="$(basename "${path}")"
    [[ "${name}" == potluck-deploy.sha256 ]] && continue
    printf '%s  %s\n' "$(sha256_file "${path}")" "${name}" >> "${checksums}"
done

remote_home="$(ssh "${ssh_args[@]}" "${target}" 'printf %s "$HOME"')" ||
    fail "cannot inspect home directory on ${target}"
[[ "${remote_home}" == /* && "${remote_home}" != *$'\n'* ]] ||
    fail "invalid remote home directory: ${remote_home}"
remote_prefix="${remote_home}/potluck"
payload_digest="$(sha256_file "${build_id}")"
remote_tmp="${remote_prefix}/.incoming-${payload_digest}-$$"
remote_tmp_q="$(shell_quote "${remote_tmp}")"
remote_prefix_q="$(shell_quote "${remote_prefix}")"
ssh_cmd() {
    ssh "${ssh_args[@]}" "${target}" "$1"
}

ssh_cmd "set -eu; rm -rf ${remote_tmp_q}; mkdir -p ${remote_tmp_q}"
scp "${scp_args[@]}" "${local_stage}"/* "${target}:${remote_tmp}/"

verify_cmd="set -eu; cd ${remote_tmp_q}; if command -v sha256sum >/dev/null 2>&1; then sha256sum -c potluck-deploy.sha256; elif command -v shasum >/dev/null 2>&1; then while read -r sum name extra; do test -n \"\${sum}\"; test -z \"\${extra:-}\"; test \"\${sum}\" = \"\$(shasum -a 256 \"\${name}\" | cut -d' ' -f1)\"; done < potluck-deploy.sha256; else exit 127; fi"
if ! ssh_cmd "${verify_cmd}"; then
    ssh_cmd "rm -rf ${remote_tmp_q}" || true
    fail "checksum verification failed on ${target}"
fi

# Keep model and shard files untouched. Replace only the files owned by the
# portable payload, then leave the verified manifest as an install record.
install_files=("${payload_files[@]}" potluck-build-id potluck-deploy.sha256)
move_cmd="set -eu; mkdir -p ${remote_prefix_q};"
move_cmd+=" if [ -f ${remote_prefix_q}/potluck-build-id ]; then while read -r sum name extra; do case \"\${name}\" in libzmq.*|libsodium.*) case \"\${name}\" in */*) ;; *) rm -f ${remote_prefix_q}/\${name} ;; esac ;; esac; done < ${remote_prefix_q}/potluck-build-id; fi;"
for name in "${install_files[@]}"; do
    name_q="$(shell_quote "${name}")"
    move_cmd+=" mv ${remote_tmp_q}/${name_q} ${remote_prefix_q}/${name_q};"
done
move_cmd+=" chmod +x ${remote_prefix_q}/potluck-node ${remote_prefix_q}/potluck-worker; rm -rf ${remote_tmp_q}"
if ! ssh_cmd "${move_cmd}"; then
    ssh_cmd "rm -rf ${remote_tmp_q}" || true
    fail "cannot install payload on ${target}"
fi
ssh_cmd "set -eu; test -x ${remote_prefix_q}/potluck-node; test -x ${remote_prefix_q}/potluck-worker; test -f ${remote_prefix_q}/potluck-build-id" ||
    fail "remote payload layout is incomplete on ${target}"

printf 'potluck-deploy: target=%s payload=%s files=%s prefix=~/potluck\n' \
    "${target}" "${payload_name}" "${#payload_files[@]}"
