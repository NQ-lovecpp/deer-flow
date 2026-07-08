#!/usr/bin/env python3
import json
import os
import sys
from urllib.request import urlopen


GIB = 1024 ** 3
DEFAULT_COMFY_URL = "http://host.docker.internal:8188"


def _get_json(url: str) -> dict:
    with urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _gb(value):
    if value is None:
        return None
    return round(float(value) / GIB, 2)


def main() -> int:
    host = os.getenv("COMFYUI_URL", DEFAULT_COMFY_URL).rstrip("/")
    stats = _get_json(f"{host}/system_stats")
    queue = _get_json(f"{host}/queue")

    system = stats.get("system") or {}
    devices = stats.get("devices") or []
    device = devices[0] if devices else {}
    summary = {
        "comfyui_url": host,
        "comfyui_version": system.get("comfyui_version"),
        "queue_running": len(queue.get("queue_running") or []),
        "queue_pending": len(queue.get("queue_pending") or []),
        "ram_total_gib": _gb(system.get("ram_total")),
        "ram_free_gib": _gb(system.get("ram_free")),
        "device": device.get("name"),
        "vram_total_gib": _gb(device.get("vram_total")),
        "vram_free_gib": _gb(device.get("vram_free")),
        "torch_vram_total_gib": _gb(device.get("torch_vram_total")),
        "torch_vram_free_gib": _gb(device.get("torch_vram_free")),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error reading ComfyUI resource status: {exc}", file=sys.stderr)
        raise SystemExit(1)
