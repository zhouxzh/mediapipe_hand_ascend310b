#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="${1:-full}"
SOC_VERSION="${2:-${SOC_VERSION:-Ascend310B1}}"
ONNX_DIR="${ROOT_DIR}/models/onnx"
OUT_DIR="${ROOT_DIR}/models/ascend"

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

mkdir -p "${OUT_DIR}"

convert_one() {
  local onnx_name="$1"
  local output_stem="$2"
  local input_shape="$3"
  local onnx_path="${ONNX_DIR}/${onnx_name}"
  local output_prefix="${OUT_DIR}/${output_stem}"

  if [[ ! -f "${onnx_path}" ]]; then
    echo "[error] missing ONNX: ${onnx_path}" >&2
    return 1
  fi

  echo "[atc] ${onnx_name} -> ${output_prefix}.om (${SOC_VERSION})"
  atc \
    --framework=5 \
    --model="${onnx_path}" \
    --output="${output_prefix}" \
    --input_format=ND \
    --input_shape="input_1:${input_shape}" \
    --soc_version="${SOC_VERSION}" \
    --log=error
}

case "${VARIANT}" in
  full)
    convert_one "mediapipe_legacy_0_10_14_palm_detection_full.onnx" "mediapipe_legacy_0_10_14_palm_detection_full" "1,192,192,3"
    convert_one "mediapipe_legacy_0_10_14_hand_landmark_full.onnx" "mediapipe_legacy_0_10_14_hand_landmark_full" "1,224,224,3"
    ;;
  lite)
    convert_one "mediapipe_legacy_0_10_14_palm_detection_lite.onnx" "mediapipe_legacy_0_10_14_palm_detection_lite" "1,192,192,3"
    convert_one "mediapipe_legacy_0_10_14_hand_landmark_lite.onnx" "mediapipe_legacy_0_10_14_hand_landmark_lite" "1,224,224,3"
    ;;
  all)
    "${BASH_SOURCE[0]}" full "${SOC_VERSION}"
    "${BASH_SOURCE[0]}" lite "${SOC_VERSION}"
    ;;
  *)
    echo "usage: $0 [full|lite|all] [Ascend310B1|Ascend310B4|...]" >&2
    exit 2
    ;;
esac

echo "[done] OM models are in ${OUT_DIR}"

