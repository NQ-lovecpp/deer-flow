import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "skills/public/image-generation/scripts/spark_image_scheduler.py"
SPEC = importlib.util.spec_from_file_location("spark_image_scheduler_test", SCRIPT_PATH)
scheduler = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scheduler
SPEC.loader.exec_module(scheduler)


class Response:
    def __init__(self, payload=None, text="", status_code=200):
        self.payload = payload if payload is not None else {}
        self.text = text
        self.status_code = status_code

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class Session:
    def __init__(self, *, ram_available_gib=64, queue_size=0, resident=None, running=0, waiting=0):
        self.ram_available_gib = ram_available_gib
        self.queue_size = queue_size
        self.resident = resident
        self.running = running
        self.waiting = waiting
        self.posts = []

    def get(self, url, timeout=10):
        if url.endswith("/system_stats"):
            return Response(
                {
                    "system": {
                        "comfyui_version": "0.18.1",
                        "ram_total": 128 * scheduler.GIB,
                        "ram_free": self.ram_available_gib * scheduler.GIB,
                    },
                    "devices": [
                        {
                            "name": "cuda:0 NVIDIA GB10",
                            "vram_total": 128 * scheduler.GIB,
                            "vram_free": 48 * scheduler.GIB,
                            "torch_vram_total": 20 * scheduler.GIB,
                            "torch_vram_free": 8 * scheduler.GIB,
                        }
                    ],
                }
            )
        if url.endswith("/queue"):
            return Response(
                {
                    "queue_running": [["job"]] if self.queue_size else [],
                    "queue_pending": [],
                }
            )
        if "/history?" in url:
            prompt = {}
            if self.resident:
                prompt = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": self.resident}}}
            return Response({"prompt-id": {"status": {"status_str": "success"}, "prompt": [0, 0, prompt]}})
        if url.endswith("/metrics"):
            return Response(
                text=(
                    f'vllm:num_requests_running{{engine="0"}} {self.running}\n'
                    f'vllm:num_requests_waiting{{engine="0"}} {self.waiting}\n'
                )
            )
        raise AssertionError(f"unexpected GET {url}")

    def post(self, url, json=None, timeout=10):
        self.posts.append((url, json))
        return Response({})


@pytest.fixture(autouse=True)
def clean_policy_env(monkeypatch):
    for name in (
        "SPARK_AI_MEMORY_BUDGET_GIB",
        "SPARK_SYSTEM_RESERVE_GIB",
        "SPARK_VLLM_RESERVED_GIB",
        "SPARK_VLLM_POLICY",
        "SPARK_COMFY_MAX_RESIDENT_DITS",
        "SPARK_IMAGE_POLICY_FILE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_policy_contains_machine_budget_and_models():
    policy = scheduler.load_policy()
    assert policy["machine"]["name"] == "NVIDIA DGX Spark"
    assert policy["budget"]["ai_memory_gib"] == 90
    assert policy["budget"]["system_reserve_gib"] == 32
    assert policy["vllm"]["policy"] == "pinned"
    assert policy["vllm"]["reserved_gib"] == 34
    assert policy["tasks"]["generate"]["reservation_gib"] == 42
    assert policy["tasks"]["edit"]["reservation_gib"] == 44
    assert policy["tasks"]["layered"]["reservation_gib"] == 52


def test_same_model_is_reused_without_release():
    policy = scheduler.load_policy()
    target = policy["tasks"]["generate"]["diffusion_model"]
    result = scheduler.plan("generate", policy, session=Session(resident=target))
    assert result["admitted"] is True
    assert f"reuse:{target}" in result["required_actions"]
    assert "release_comfy_before_load" not in result["required_actions"]


def test_switching_common_models_releases_only_comfy():
    policy = scheduler.load_policy()
    result = scheduler.plan(
        "edit",
        policy,
        session=Session(resident=policy["tasks"]["generate"]["diffusion_model"]),
    )
    assert result["admitted"] is True
    assert result["estimated_reservation_gib"]["combined_gib"] == 78
    assert result["required_actions"][-2:] == [
        "release_comfy_before_load",
        f"load:{policy['tasks']['edit']['diffusion_model']}",
    ]


def test_layered_is_exclusive_and_released_after_job():
    policy = scheduler.load_policy()
    result = scheduler.plan("layered", policy, session=Session())
    assert result["admitted"] is True
    assert result["estimated_reservation_gib"]["combined_gib"] == 86
    assert "release_comfy_before_load" in result["required_actions"]
    assert "release_comfy_after_job" in result["required_actions"]


def test_low_system_memory_and_busy_queue_are_rejected():
    policy = scheduler.load_policy()
    low = scheduler.plan("generate", policy, session=Session(ram_available_gib=31))
    busy = scheduler.plan("edit", policy, session=Session(queue_size=1))
    assert low["admitted"] is False
    assert "below the 32.0GiB reserve" in low["rejection_reason"]
    assert busy["admitted"] is False
    assert "queue has 1 item" in busy["rejection_reason"]


def test_hard_budget_is_enforced(monkeypatch):
    monkeypatch.setenv("SPARK_AI_MEMORY_BUDGET_GIB", "70")
    result = scheduler.plan("generate", scheduler.load_policy(), session=Session())
    assert result["admitted"] is False
    assert "above the 70.0GiB AI budget" in result["rejection_reason"]


def test_vllm_activity_is_reported_but_never_reclaimed():
    result = scheduler.plan("generate", scheduler.load_policy(), session=Session(running=2, waiting=1))
    assert result["admitted"] is True
    assert result["vllm"]["requests_running"] == 2
    assert result["vllm"]["requests_waiting"] == 1
    assert result["vllm"]["policy"] == "pinned"


def test_release_calls_only_comfy_free():
    session = Session()
    result = scheduler.release_comfy("http://comfy", session=session)
    assert result["released"] is True
    assert session.posts == [
        ("http://comfy/free", {"unload_models": True, "free_memory": True})
    ]


def test_scheduler_has_no_service_control_code():
    source = SCRIPT_PATH.read_text(encoding="utf-8").lower()
    for forbidden in ("docker stop", "docker restart", "docker kill", "docker rm", "subprocess"):
        assert forbidden not in source
