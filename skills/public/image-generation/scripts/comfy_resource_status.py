#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import spark_image_scheduler as scheduler  # noqa: E402


def main() -> int:
    summary = scheduler.status(
        comfy_url=os.getenv("COMFYUI_URL", scheduler.DEFAULT_COMFY_URL),
        vllm_metrics_url=os.getenv("VLLM_METRICS_URL"),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["admitted"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error reading ComfyUI resource status: {exc}", file=sys.stderr)
        raise SystemExit(1)
