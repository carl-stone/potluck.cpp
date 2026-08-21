#!/usr/bin/env bash
set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"

git -C "${REPO}" config --local core.hooksPath .githooks
chmod +x "${REPO}/.githooks/pre-push" "${REPO}/scripts/pre-push-check.sh"

printf 'Installed Potluck Git hooks for %s\n' "${REPO}"
printf 'pre-push runs scripts/pre-push-check.sh locally.\n'
