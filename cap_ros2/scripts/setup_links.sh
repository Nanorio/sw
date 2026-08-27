#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ln -sfn "${ROOT}/weights" "${WS_DIR}/weights"
ln -sfn "${ROOT}/staticSource" "${WS_DIR}/staticSource"
ln -sfn "${ROOT}/outcome" "${WS_DIR}/outcome"

echo "Links created in ${WS_DIR}"
