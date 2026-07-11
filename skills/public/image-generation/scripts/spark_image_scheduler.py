#!/usr/bin/env python3
"""Deterministic resource policy for DeerFlow image jobs on DGX Spark."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import requests


GIB = 1024 ** 3
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_POLICY_PATH = SCRIPT_DIR.parent / "references" / "spark-runtime-policy.json"
DEFAULT_COMFY_URL = "http://host.docker.internal:8188"


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value in (None, "") else float(value)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value in (None, "") else int(value)


def load_policy(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    policy_path = Path(path or os.getenv("SPARK_IMAGE_POLICY_FILE") or DEFAULT_POLICY_PATH)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["budget"]["ai_memory_gib"] = _float_env(
        "SPARK_AI_MEMORY_BUDGET_GIB", policy["budget"]["ai_memory_gib"]
    )
    policy["budget"]["system_reserve_gib"] = _float_env(
        "SPARK_SYSTEM_RESERVE_GIB", policy["budget"]["system_reserve_gib"]
    )
    policy["budget"]["max_resident_image_dits"] = _int_env(
        "SPARK_COMFY_MAX_RESIDENT_DITS", policy["budget"]["max_resident_image_dits"]
    )
    policy["vllm"]["reserved_gib"] = _float_env(
        "SPARK_VLLM_RESERVED_GIB", policy["vllm"]["reserved_gib"]
    )
    policy["vllm"]["policy"] = os.getenv("SPARK_VLLM_POLICY", policy["vllm"]["policy"]).strip().lower()
    if policy["vllm"]["policy"] != "pinned":
        raise ValueError("SPARK_VLLM_POLICY must remain 'pinned' on this machine")
    return policy


def _gib(value: int | float | None) -> float | None:
    return None if value is None else round(float(value) / GIB, 2)


def _comfy_snapshot(comfy_url: str, session=requests) -> tuple[dict[str, Any], dict[str, Any]]:
    host = comfy_url.rstrip("/")
    stats_response = session.get(f"{host}/system_stats", timeout=10)
    stats_response.raise_for_status()
    queue_response = session.get(f"{host}/queue", timeout=10)
    queue_response.raise_for_status()
    history_response = session.get(f"{host}/history?max_items=5", timeout=10)
    history_response.raise_for_status()

    stats = stats_response.json()
    queue = queue_response.json()
    history = history_response.json()
    system = stats.get("system") or {}
    devices = stats.get("devices") or []
    device = devices[0] if devices else {}
    snapshot = {
        "url": host,
        "version": system.get("comfyui_version"),
        "queue_running": len(queue.get("queue_running") or []),
        "queue_pending": len(queue.get("queue_pending") or []),
        "ram_total_gib": _gib(system.get("ram_total")),
        "ram_available_gib": _gib(system.get("ram_free")),
        "device": device.get("name"),
        "vram_total_gib": _gib(device.get("vram_total")),
        "vram_free_gib": _gib(device.get("vram_free")),
        "torch_vram_total_gib": _gib(device.get("torch_vram_total")),
        "torch_vram_free_gib": _gib(device.get("torch_vram_free")),
    }
    return snapshot, history


def _latest_unet(history: dict[str, Any]) -> str | None:
    for entry in history.values():
        status = entry.get("status") or {}
        if status.get("status_str") not in (None, "success"):
            continue
        prompt_data = entry.get("prompt") or []
        prompt = prompt_data[2] if len(prompt_data) > 2 and isinstance(prompt_data[2], dict) else {}
        for node in prompt.values():
            if node.get("class_type") == "UNETLoader":
                return (node.get("inputs") or {}).get("unet_name")
    return None


def _parse_metric(text: str, metric_name: str) -> float | None:
    pattern = re.compile(rf"^{re.escape(metric_name)}(?:\{{[^}}]*\}})?\s+([-+0-9.eE]+)$", re.MULTILINE)
    values = [float(match.group(1)) for match in pattern.finditer(text)]
    return sum(values) if values else None


def _vllm_snapshot(policy: dict[str, Any], explicit_url: str | None = None, session=requests) -> dict[str, Any]:
    urls = [explicit_url] if explicit_url else list(policy["vllm"].get("metrics_urls") or [])
    last_error = None
    for url in urls:
        if not url:
            continue
        try:
            response = session.get(url, timeout=3)
            response.raise_for_status()
            text = response.text
            return {
                "service": policy["vllm"]["service"],
                "model": policy["vllm"]["model"],
                "policy": "pinned",
                "reserved_gib": policy["vllm"]["reserved_gib"],
                "metrics_url": url,
                "reachable": True,
                "requests_running": _parse_metric(text, "vllm:num_requests_running"),
                "requests_waiting": _parse_metric(text, "vllm:num_requests_waiting"),
                "error": None,
            }
        except Exception as exc:  # The fixed reservation still protects memory if metrics are unavailable.
            last_error = str(exc)
    return {
        "service": policy["vllm"]["service"],
        "model": policy["vllm"]["model"],
        "policy": "pinned",
        "reserved_gib": policy["vllm"]["reserved_gib"],
        "metrics_url": explicit_url,
        "reachable": False,
        "requests_running": None,
        "requests_waiting": None,
        "error": last_error or "no metrics URL configured",
    }


def status(
    policy: dict[str, Any] | None = None,
    comfy_url: str | None = None,
    vllm_metrics_url: str | None = None,
    session=requests,
) -> dict[str, Any]:
    policy = policy or load_policy()
    comfy_url = comfy_url or os.getenv("COMFYUI_URL", DEFAULT_COMFY_URL)
    comfy, history = _comfy_snapshot(comfy_url, session=session)
    resident_model = _latest_unet(history)
    released_detection = (policy.get("comfy") or {}).get("released_detection") or {}
    min_released_ram = released_detection.get("min_ram_available_gib")
    min_released_vram = released_detection.get("min_vram_free_gib")
    if (
        resident_model
        and min_released_ram is not None
        and min_released_vram is not None
        and comfy["ram_available_gib"] is not None
        and comfy["vram_free_gib"] is not None
        and comfy["ram_available_gib"] >= min_released_ram
        and comfy["vram_free_gib"] >= min_released_vram
    ):
        resident_model = None
    vllm = _vllm_snapshot(policy, explicit_url=vllm_metrics_url, session=session)
    queue_size = comfy["queue_running"] + comfy["queue_pending"]
    ram_available = comfy["ram_available_gib"]
    reserve = policy["budget"]["system_reserve_gib"]
    admitted = queue_size == 0 and (ram_available is None or ram_available >= reserve)
    reason = None
    if queue_size:
        reason = f"ComfyUI queue has {queue_size} item(s)"
    elif ram_available is not None and ram_available < reserve:
        reason = f"System available memory {ram_available:.1f}GiB is below the {reserve:.1f}GiB reserve"
    actions = []
    if not vllm["reachable"]:
        actions.append("verify_vllm_metrics")
    return {
        "machine": policy["machine"],
        "budget": {
            **policy["budget"],
            "configured_vllm_reservation_gib": policy["vllm"]["reserved_gib"],
        },
        "vllm": vllm,
        "comfy": comfy,
        "resident_model": resident_model,
        "selected_task": None,
        "selected_model": None,
        "estimated_reservation_gib": None,
        "admitted": admitted,
        "required_actions": actions,
        "rejection_reason": reason,
    }


def plan(
    task: str,
    policy: dict[str, Any] | None = None,
    comfy_url: str | None = None,
    vllm_metrics_url: str | None = None,
    session=requests,
) -> dict[str, Any]:
    policy = policy or load_policy()
    if task not in policy["tasks"]:
        raise ValueError(f"Unknown Spark image task {task!r}; use generate, edit, or layered")
    result = status(policy, comfy_url=comfy_url, vllm_metrics_url=vllm_metrics_url, session=session)
    target = policy["tasks"][task]
    expected_total = policy["vllm"]["reserved_gib"] + target["reservation_gib"]
    max_resident = policy["budget"]["max_resident_image_dits"]
    reasons = []
    if result["rejection_reason"]:
        reasons.append(result["rejection_reason"])
    if max_resident != 1:
        reasons.append("SPARK_COMFY_MAX_RESIDENT_DITS must be 1")
    if expected_total > policy["budget"]["ai_memory_gib"]:
        reasons.append(
            f"Pinned vLLM plus {task} reservation is {expected_total:.1f}GiB, "
            f"above the {policy['budget']['ai_memory_gib']:.1f}GiB AI budget"
        )

    actions = list(result["required_actions"])
    resident = result["resident_model"]
    target_model = target["diffusion_model"]
    if task == "layered":
        actions.extend(["release_comfy_before_load", f"load:{target_model}", "release_comfy_after_job"])
    elif resident == target_model:
        actions.append(f"reuse:{target_model}")
    else:
        actions.extend(["release_comfy_before_load", f"load:{target_model}"])

    result.update(
        {
            "selected_task": task,
            "selected_model": target,
            "estimated_reservation_gib": {
                "vllm_gib": policy["vllm"]["reserved_gib"],
                "image_task_gib": target["reservation_gib"],
                "combined_gib": expected_total,
            },
            "admitted": not reasons,
            "required_actions": actions,
            "rejection_reason": "; ".join(reasons) if reasons else None,
        }
    )
    return result


def release_comfy(comfy_url: str | None = None, session=requests) -> dict[str, Any]:
    host = (comfy_url or os.getenv("COMFYUI_URL", DEFAULT_COMFY_URL)).rstrip("/")
    queue_response = session.get(f"{host}/queue", timeout=10)
    queue_response.raise_for_status()
    queue = queue_response.json()
    queued = len(queue.get("queue_running") or []) + len(queue.get("queue_pending") or [])
    if queued:
        raise RuntimeError(f"Refusing to release Comfy models while {queued} queue item(s) exist")
    response = session.post(
        f"{host}/free",
        json={"unload_models": True, "free_memory": True},
        timeout=10,
    )
    response.raise_for_status()
    return {"released": True, "comfyui_url": host}


def _print_result(result: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(
        f"task={result.get('selected_task') or 'status'} admitted={result['admitted']} "
        f"resident={result.get('resident_model') or 'none'} "
        f"reason={result.get('rejection_reason') or 'none'}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DGX Spark image resource scheduler")
    parser.add_argument("--policy", default=None, help="Path to spark-runtime-policy.json")
    parser.add_argument("--comfy-url", default=None, help="ComfyUI base URL")
    parser.add_argument("--vllm-metrics-url", default=None, help="Optional vLLM metrics URL")
    subparsers = parser.add_subparsers(dest="command", required=True)
    status_parser = subparsers.add_parser("status", help="Show machine and live resource state")
    status_parser.add_argument("--json", action="store_true")
    plan_parser = subparsers.add_parser("plan", help="Plan one local image task")
    plan_parser.add_argument("--task", choices=["generate", "edit", "layered"], required=True)
    plan_parser.add_argument("--json", action="store_true")
    release_parser = subparsers.add_parser("release", help="Unload Comfy models only")
    release_parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    policy = load_policy(args.policy)
    if args.command == "status":
        result = status(policy, comfy_url=args.comfy_url, vllm_metrics_url=args.vllm_metrics_url)
        _print_result(result, args.json)
        return 0 if result["admitted"] else 2
    if args.command == "plan":
        result = plan(
            args.task,
            policy,
            comfy_url=args.comfy_url,
            vllm_metrics_url=args.vllm_metrics_url,
        )
        _print_result(result, args.json)
        return 0 if result["admitted"] else 2
    result = release_comfy(comfy_url=args.comfy_url)
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else "Comfy models released")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Spark image scheduler error: {exc}", file=sys.stderr)
        raise SystemExit(1)
