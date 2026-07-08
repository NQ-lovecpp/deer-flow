#!/usr/bin/env bash
set -euo pipefail

COMFYUI_DIR="${COMFYUI_DIR:-/home/chen/comfyui/ComfyUI}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${MANIFEST:-${SCRIPT_DIR}/../workflows/comfy/model-manifest.json}"

if [[ ! -d "${COMFYUI_DIR}" ]]; then
  echo "ComfyUI directory not found: ${COMFYUI_DIR}" >&2
  exit 1
fi

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Model manifest not found: ${MANIFEST}" >&2
  exit 1
fi

python3 - "${MANIFEST}" "${COMFYUI_DIR}" <<'PY' | while IFS=$'\t' read -r url target; do
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
comfyui_dir = Path(sys.argv[2])
data = json.loads(manifest.read_text(encoding="utf-8"))
for model in data["models"]:
    print(model["modelscope_url"] + "\t" + str(comfyui_dir / model["target"]))
PY
  mkdir -p "$(dirname "${target}")"
  if [[ -s "${target}" ]]; then
    echo "SKIP ${target}"
    continue
  fi
  tmp="${target}.part"
  echo "DOWNLOAD $(basename "${target}")"
  curl \
    --fail \
    --location \
    --retry 20 \
    --retry-all-errors \
    --connect-timeout 30 \
    --speed-time 120 \
    --speed-limit 10240 \
    -C - \
    -o "${tmp}" \
    "${url}"
  mv "${tmp}" "${target}"
  echo "DONE ${target}"
done
