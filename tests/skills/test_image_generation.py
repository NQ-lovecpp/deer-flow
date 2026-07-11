import base64
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from skill_loader import FakeResp, load  # noqa: E402

img = load("image-generation")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ["GEMINI_API_KEY", "MINIMAX_API_KEY", "IMAGE_GENERATION_PROVIDER",
              "MINIMAX_API_HOST", "MINIMAX_IMAGE_MODEL", "COMFYUI_QWEN_STEPS",
              "COMFYUI_QWEN_CFG", "COMFYUI_QWEN_MEGAPIXELS",
              "COMFYUI_QWEN_UNET", "SPARK_LAYERED_ALLOW_LARGE"]:
        monkeypatch.delenv(k, raising=False)


def test_resolve_prefers_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    assert img._resolve_provider("IMAGE_GENERATION_PROVIDER", "gemini", True) == "gemini"


def test_resolve_falls_back_to_minimax(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    assert img._resolve_provider("IMAGE_GENERATION_PROVIDER", "gemini", False) == "minimax"


def test_resolve_override_wins(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("IMAGE_GENERATION_PROVIDER", "MiniMax")
    assert img._resolve_provider("IMAGE_GENERATION_PROVIDER", "gemini", True) == "minimax"


def test_resolve_falls_back_to_local_comfy_when_no_remote_provider(monkeypatch):
    assert img._resolve_provider("IMAGE_GENERATION_PROVIDER", "gemini", False) == "comfy"


def test_minimax_builds_payload_and_writes(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    raw = b"PNGBYTES"
    captured = {}

    def fake_post(url, headers=None, json=None, **kw):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return FakeResp({"data": {"image_base64": [base64.b64encode(raw).decode()]},
                         "base_resp": {"status_code": 0, "status_msg": "success"}})

    monkeypatch.setattr(img.requests, "post", fake_post)
    out = tmp_path / "o.jpg"
    prompt_file = tmp_path / "p.json"
    prompt_file.write_text("a red apple", encoding="utf-8")
    msg = img.generate_image(str(prompt_file), [], str(out), "16:9")

    assert out.read_bytes() == raw
    assert captured["url"].endswith("/v1/image_generation")
    assert captured["headers"]["Authorization"] == "Bearer m"
    assert captured["json"]["model"] == "image-01"
    assert captured["json"]["response_format"] == "base64"
    assert captured["json"]["aspect_ratio"] == "16:9"
    assert captured["json"]["n"] == 1
    assert captured["json"]["prompt_optimizer"] is True
    assert "Successfully generated image" in msg


def test_minimax_reference_image_as_data_url(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    captured = {}

    def fake_post(url, headers=None, json=None, **kw):
        captured["json"] = json
        return FakeResp({"data": {"image_base64": [base64.b64encode(b"x").decode()]},
                         "base_resp": {"status_code": 0}})

    monkeypatch.setattr(img.requests, "post", fake_post)
    ref = tmp_path / "ref.jpg"
    ref.write_bytes(b"\xff\xd8refbytes")
    prompt_file = tmp_path / "p.json"
    prompt_file.write_text("scene", encoding="utf-8")
    img.generate_image(str(prompt_file), [str(ref)], str(tmp_path / "o.jpg"), "1:1")

    subj = captured["json"]["subject_reference"]
    assert subj[0]["type"] == "character"
    assert subj[0]["image_file"].startswith("data:image/jpeg;base64,")
    import base64 as _b64
    encoded = subj[0]["image_file"].split(",", 1)[1]
    assert _b64.b64decode(encoded) == b"\xff\xd8refbytes"


def test_minimax_raises_on_base_resp_error(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")

    def fake_post(url, headers=None, json=None, **kw):
        return FakeResp({"base_resp": {"status_code": 1004, "status_msg": "auth failed"}})

    monkeypatch.setattr(img.requests, "post", fake_post)
    prompt_file = tmp_path / "p.json"
    prompt_file.write_text("x", encoding="utf-8")
    with pytest.raises(Exception) as e:
        img.generate_image(str(prompt_file), [], str(tmp_path / "o.jpg"), "1:1")
    assert "1004" in str(e.value)


def test_minimax_extracts_json_prompt_field(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    captured = {}

    def fake_post(url, headers=None, json=None, **kw):
        captured["json"] = json
        return FakeResp({"data": {"image_base64": [base64.b64encode(b"x").decode()]},
                         "base_resp": {"status_code": 0}})

    monkeypatch.setattr(img.requests, "post", fake_post)
    prompt_file = tmp_path / "p.json"
    prompt_file.write_text(
        '{"prompt": "a red barn at dawn", "style": "watercolor", '
        '"composition": "rule of thirds", "negative_prompt": "blurry"}',
        encoding="utf-8",
    )
    img.generate_image(str(prompt_file), [], str(tmp_path / "o.jpg"), "16:9")

    # Only the JSON `prompt` field reaches MiniMax — no other fields, no JSON syntax.
    assert captured["json"]["prompt"] == "a red barn at dawn"
    assert captured["json"]["prompt_optimizer"] is True


def test_minimax_plaintext_prompt_passes_through(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    captured = {}

    def fake_post(url, headers=None, json=None, **kw):
        captured["json"] = json
        return FakeResp({"data": {"image_base64": [base64.b64encode(b"x").decode()]},
                         "base_resp": {"status_code": 0}})

    monkeypatch.setattr(img.requests, "post", fake_post)
    prompt_file = tmp_path / "p.txt"
    prompt_file.write_text("a red apple on a table", encoding="utf-8")
    img.generate_image(str(prompt_file), [], str(tmp_path / "o.jpg"), "1:1")

    assert captured["json"]["prompt"] == "a red apple on a table"


def test_minimax_rejects_overlong_prompt_without_calling_api(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")

    def fake_post(url, headers=None, json=None, **kw):  # pragma: no cover
        raise AssertionError("must not call the API when the prompt is over the limit")

    monkeypatch.setattr(img.requests, "post", fake_post)
    prompt_file = tmp_path / "p.json"
    prompt_file.write_text('{"prompt": "' + "x" * 1600 + '"}', encoding="utf-8")
    out = tmp_path / "o.jpg"
    msg = img.generate_image(str(prompt_file), [], str(out), "16:9")

    assert "1500" in msg
    assert "character" in msg.lower()
    assert not out.exists()


def test_minimax_creates_nested_output_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")

    def fake_post(url, headers=None, json=None, **kw):
        return FakeResp({"data": {"image_base64": [base64.b64encode(b"img").decode()]},
                         "base_resp": {"status_code": 0}})

    monkeypatch.setattr(img.requests, "post", fake_post)
    prompt_file = tmp_path / "p.txt"
    prompt_file.write_text("a cat", encoding="utf-8")
    out = tmp_path / "nested" / "dir" / "o.jpg"
    img.generate_image(str(prompt_file), [], str(out), "1:1")

    assert out.read_bytes() == b"img"


def test_unknown_provider_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("IMAGE_GENERATION_PROVIDER", "openai")
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    pf = tmp_path / "p.json"
    pf.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        img.generate_image(str(pf), [], str(tmp_path / "o.jpg"), "1:1")


def test_guess_mime_by_extension():
    assert img._guess_mime("/a/b.png") == "image/png"
    assert img._guess_mime("/a/b.webp") == "image/webp"
    assert img._guess_mime("/a/b.jpg") == "image/jpeg"
    assert img._guess_mime("/a/b.unknown") == "image/jpeg"


def test_local_task_routing():
    assert img._resolve_local_task("auto", []) == "generate"
    assert img._resolve_local_task("auto", ["input.png"]) == "edit"
    assert img._resolve_local_task("layered", []) == "layered"
    assert img._resolve_local_task("layered", ["input.png"]) == "layered"
    with pytest.raises(ValueError):
        img._resolve_local_task("edit", [])


def test_layered_workflow_uses_fp8mixed_and_native_nodes(monkeypatch):
    monkeypatch.setattr(img, "_apply_spark_memory_budget", lambda kind, opts, options, aspect: opts)
    workflow, options = img._build_comfy_qwen_layered_workflow(
        "separate the subject from the background",
        "input.png",
        {"layers": 3, "resolution": 640, "preset": "fast", "seed": 7},
    )
    nodes = list(workflow.values())
    unet = next(node for node in nodes if node["class_type"] == "UNETLoader")
    latent = next(node for node in nodes if node["class_type"] == "EmptyQwenImageLayeredLatentImage")
    assert unet["inputs"]["unet_name"] == "qwen_image_layered_fp8mixed.safetensors"
    assert latent["inputs"]["layers"] == 3
    assert any(node["class_type"] == "ReferenceLatent" for node in nodes)
    assert any(node["class_type"] == "LatentCutToBatch" for node in nodes)
    assert options["mode"] == "image-to-layers"


def test_layered_automatic_limits_are_enforced(monkeypatch):
    monkeypatch.setattr(img, "_apply_spark_memory_budget", lambda kind, opts, options, aspect: opts)
    with pytest.raises(ValueError, match="limited to 4 layers"):
        img._resolve_qwen_layered_options({"layers": 5})
    with pytest.raises(ValueError, match="limited to 640px"):
        img._resolve_qwen_layered_options({"resolution": 1024})


def test_comfy_dry_run_does_not_queue(monkeypatch, tmp_path):
    monkeypatch.setenv("IMAGE_GENERATION_PROVIDER", "comfy")
    schedule = {
        "admitted": True,
        "resident_model": None,
        "budget": {"ai_memory_gib": 90},
        "estimated_reservation_gib": {"combined_gib": 76},
        "required_actions": ["release_comfy_before_load"],
    }
    monkeypatch.setattr(img, "_prepare_spark_task", lambda task, dry_run=False: {**schedule, "selected_task": task})
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("a red apple", encoding="utf-8")
    result = img.generate_image(
        str(prompt_file),
        [],
        str(tmp_path / "out.png"),
        task="generate",
        dry_run=True,
    )
    assert json.loads(result)["selected_task"] == "generate"
    assert not (tmp_path / "out.png").exists()
