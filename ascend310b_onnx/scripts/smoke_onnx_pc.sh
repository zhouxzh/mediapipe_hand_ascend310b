#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${1:-${ROOT_DIR}/data/smoke_images/images/train_palm_ac44e9bd-97a1-4f28-8398-f825842fc59d.jpg}"

python "${ROOT_DIR}/scripts/run_image_onnx.py" "${IMAGE}" \
  --output "${ROOT_DIR}/runs/onnx_smoke_result.json"

