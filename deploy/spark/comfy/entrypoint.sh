#!/usr/bin/env sh
set -eu

cd /app/ComfyUI

# Intentional word splitting lets COMFYUI_EXTRA_ARGS contain normal main.py flags.
# Do not put secrets in COMFYUI_EXTRA_ARGS; it is visible in container metadata.
exec python3 main.py --listen 0.0.0.0 --port "${COMFYUI_PORT:-8188}" ${COMFYUI_EXTRA_ARGS:-}
