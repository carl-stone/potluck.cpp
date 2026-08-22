#!/usr/bin/env bash
# Check portable payload installation without a compiler, model, or source tree.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/potluck-packaging.XXXXXX")"
cleanup() {
    rm -rf "${tmp}"
}
trap cleanup EXIT

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

case "$(uname -s) $(uname -m)" in
    'Darwin arm64'*|'Darwin aarch64'*) local_payload_name='mac-arm64'; local_library='libzmq.5.dylib' ;;
    'Linux x86_64'*|'Linux amd64'*) local_payload_name='linux-x86_64'; local_library='libzmq.so.5' ;;
    'Linux aarch64'*) local_payload_name='linux-aarch64'; local_library='libzmq.so.5' ;;
    'Linux arm64'*) local_payload_name='linux-arm64'; local_library='libzmq.so.5' ;;
    *)
        printf 'unsupported test platform\n' >&2
        exit 2
        ;;
esac

dist="${tmp}/dist"
mkdir -p "${dist}"

make_payload() {
    local dir="$1"
    local marker="$2"
    local platform="$3"
    local library="$4"
    mkdir -p "${dir}"
    printf '%s node\n' "${marker}" > "${dir}/potluck-node"
    printf '%s worker\n' "${marker}" > "${dir}/potluck-worker"
    printf '%s library\n' "${marker}" > "${dir}/${library}"
    chmod +x "${dir}/potluck-node" "${dir}/potluck-worker"
    {
        printf 'platform %s\n' "${platform}"
        printf 'commit packaging-test\n'
        for name in potluck-node potluck-worker "${library}"; do
            printf '%s  %s\n' "$(sha256_file "${dir}/${name}")" "${name}"
        done
    } > "${dir}/potluck-build-id"
}

make_payload "${dist}/mac-arm64" mac-arm64 mac-arm64 libzmq.5.dylib
make_payload "${dist}/linux-x86_64" linux-x86_64 linux-x86_64 libzmq.so.5

# Local payload setup does not inspect a compiler or a source-build directory.
payload="${dist}/${local_payload_name}"
prefix="${tmp}/prefix"
HOME="${tmp}/home" bash "${REPO}/scripts/install.sh" \
    --payload "${payload}" --prefix "${prefix}"
[[ -x "${prefix}/potluck-node" ]]
[[ -x "${prefix}/potluck-worker" ]]
[[ -f "${prefix}/${local_library}" ]]
[[ -f "${prefix}/potluck-build-id" ]]
[[ ! -e "${prefix}/Qwen3.5-0.8B-Q4_0.gguf" ]]
printf 'PACKAGING PAYLOAD CHECK PASSED\n'

# A changed payload is rejected by the staged checksum manifest.
bad_payload="${tmp}/bad-${local_payload_name}"
cp -R "${payload}" "${bad_payload}"
printf 'tampered\n' >> "${bad_payload}/potluck-worker"
bad_output="${tmp}/bad-payload.out"
if HOME="${tmp}/bad-home" bash "${REPO}/scripts/install.sh" \
    --payload "${bad_payload}" --prefix "${tmp}/bad-prefix" \
    > "${bad_output}" 2>&1; then
    printf 'checksum mismatch was accepted\n' >&2
    exit 1
fi
[[ "$(<"${bad_output}")" == *'payload checksum mismatch'* ]]
printf 'PACKAGING CHECKSUM REJECTION PASSED\n'

fake_bin="${tmp}/fake-bin"
remote_home="${tmp}/remote-home"
mkdir -p "${fake_bin}" "${remote_home}/potluck"
printf 'keep this model\n' > "${remote_home}/potluck/existing.gguf"

cat > "${fake_bin}/ssh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
command="${!#}"
case "${command}" in
    'uname -sm 2>/dev/null')
        printf '%s\n' "${FAKE_SSH_PLATFORM}"
        ;;
    'printf %s "$HOME"')
        printf '%s' "${FAKE_REMOTE_HOME}"
        ;;
    *)
        HOME="${FAKE_REMOTE_HOME}" bash -c "${command}"
        ;;
esac
EOF

cat > "${fake_bin}/scp" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
args=("$@")
last=$(( ${#args[@]} - 1 ))
destination="${args[last]}"
destination_path="${destination#*:}"
mkdir -p "${destination_path}"
sources=()
for ((i = 0; i < last; ++i)); do
    case "${args[i]}" in
        -q) ;;
        -o|-P) i=$((i + 1)) ;;
        *) sources+=("${args[i]}") ;;
    esac
done
cp -f "${sources[@]}" "${destination_path}/"
if [[ "${FAKE_SCP_TAMPER:-0}" == 1 ]]; then
    printf 'tampered in transit\n' >> "${destination_path}/potluck-worker"
fi
EOF
chmod +x "${fake_bin}/ssh" "${fake_bin}/scp"

# A modified transfer is rejected by the remote checksum verification.
bad_remote_output="${tmp}/bad-remote.out"
if PATH="${fake_bin}:${PATH}" \
    POTLUCK_DIST_DIR="${dist}" \
    FAKE_REMOTE_HOME="${remote_home}" \
    FAKE_SSH_PLATFORM='Linux x86_64' \
    FAKE_SCP_TAMPER=1 \
    bash "${REPO}/scripts/deploy.sh" --target fake-target \
    > "${bad_remote_output}" 2>&1; then
    printf 'remote checksum mismatch was accepted\n' >&2
    exit 1
fi
[[ "$(<"${bad_remote_output}")" == *'checksum verification failed'* ]]
printf 'PACKAGING REMOTE CHECKSUM REJECTION PASSED\n'

# The discovered Linux platform selects the Linux payload and installs only
# managed files into the existing ~/potluck layout.
PATH="${fake_bin}:${PATH}" \
POTLUCK_DIST_DIR="${dist}" \
FAKE_REMOTE_HOME="${remote_home}" \
FAKE_SSH_PLATFORM='Linux x86_64' \
bash "${REPO}/scripts/deploy.sh" --target fake-target
remote_prefix="${remote_home}/potluck"
remote_files=("${remote_prefix}"/*)
[[ "${#remote_files[@]}" -eq 6 ]]
for name in \
    existing.gguf \
    libzmq.so.5 \
    potluck-build-id \
    potluck-deploy.sha256 \
    potluck-node \
    potluck-worker; do
    [[ -f "${remote_prefix}/${name}" ]]
done
[[ "$(<"${remote_prefix}/potluck-node")" == 'linux-x86_64 node' ]]
[[ "$(<"${remote_prefix}/potluck-worker")" == 'linux-x86_64 worker' ]]
[[ "$(<"${remote_prefix}/libzmq.so.5")" == 'linux-x86_64 library' ]]
[[ "$(<"${remote_prefix}/existing.gguf")" == 'keep this model' ]]
checksum_count=0
while read -r checksum name extra; do
    [[ "${checksum}" =~ ^[[:xdigit:]]{64}$ && -z "${extra:-}" ]]
    [[ -f "${remote_prefix}/${name}" ]]
    [[ "$(sha256_file "${remote_prefix}/${name}")" == "${checksum}" ]]
    checksum_count=$((checksum_count + 1))
done < "${remote_prefix}/potluck-deploy.sha256"
[[ "${checksum_count}" -eq 4 ]]
printf 'PACKAGING REMOTE INSTALL CHECK PASSED\n'
