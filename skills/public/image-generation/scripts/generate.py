import base64
import contextlib
import fcntl
import json
import os
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import spark_image_scheduler as spark_scheduler  # noqa: E402

MINIMAX_DEFAULT_HOST = "https://api.minimaxi.com"
COMFY_DEFAULT_HOST = "http://host.docker.internal:8188"
# MiniMax image-01 caps the prompt at 1500 characters and rejects longer requests
# with a generic "invalid params" error, so validate before calling the API.
MINIMAX_PROMPT_MAX_CHARS = 1500

QWEN_EDIT_PRESETS = {
    "fast": {"steps": 4, "cfg": 1.0, "denoise": 1.0, "megapixels": 0.5, "weight_dtype": "default", "use_lora": True},
    "balanced": {"steps": 20, "cfg": 4.0, "denoise": 1.0, "megapixels": 0.5, "weight_dtype": "default", "use_lora": False},
    "quality": {"steps": 36, "cfg": 4.0, "denoise": 1.0, "megapixels": 0.75, "weight_dtype": "default", "use_lora": False},
}

QWEN_IMAGE_PRESETS = {
    "fast": {"steps": 4, "cfg": 1.0, "weight_dtype": "default", "use_lora": False},
    "balanced": {"steps": 30, "cfg": 4.0, "weight_dtype": "default", "use_lora": False},
    "quality": {"steps": 50, "cfg": 4.0, "weight_dtype": "default", "use_lora": False},
}

QWEN_LAYERED_PRESETS = {
    "fast": {"steps": 20, "cfg": 2.5},
    "balanced": {"steps": 30, "cfg": 3.0},
    "quality": {"steps": 50, "cfg": 4.0},
}

QWEN_IMAGE_DIMENSIONS = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

GIB = 1024 ** 3



def validate_image(image_path: str) -> bool:
    """Validate if an image file can be opened and is not corrupted."""
    from PIL import Image  # lazy import: keeps module importable without Pillow

    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            image.load()
        return True
    except Exception as exc:
        print(f"Warning: Image '{image_path}' is invalid or corrupted: {exc}")
        return False


def _resolve_provider(override_env: str, existing_provider: str, has_existing_creds: bool) -> str:
    """Pick the generation provider.

    1. Explicit <SKILL>_PROVIDER override wins.
    2. Otherwise prefer the existing provider when its credentials are present.
    3. Otherwise fall back to MiniMax when MINIMAX_API_KEY is set.
    """
    override = os.getenv(override_env)
    if override:
        return override.strip().lower()
    if has_existing_creds:
        return existing_provider
    if os.getenv("MINIMAX_API_KEY"):
        return "minimax"
    # Prefer local ComfyUI when no remote credentials are configured. The
    # concrete local workflow is selected later from the request shape: reference
    # images use Qwen Image Edit, while pure text-to-image uses Qwen Image.
    return "comfy"


def _minimax_host() -> str:
    return os.getenv("MINIMAX_API_HOST", MINIMAX_DEFAULT_HOST).rstrip("/")


def _check_base_resp(payload: dict) -> None:
    base = payload.get("base_resp") or {}
    if base.get("status_code", 0) != 0:
        raise Exception(
            f"MiniMax error {base.get('status_code')}: {base.get('status_msg')}"
        )


def _guess_mime(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    return {
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(ext, "image/jpeg")


def _to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{_guess_mime(image_path)};base64,{b64}"


def _ensure_output_dir(output_file: str) -> None:
    """Create the output file's parent directory so nested paths don't fail."""
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


def _minimax_prompt(raw: str) -> str:
    """Extract the single text prompt MiniMax image-01 expects.

    The shared prompt file is structured JSON (a consolidated ``prompt`` plus
    Gemini-oriented fields like ``style`` / ``composition`` / ``negative_prompt``),
    but MiniMax consumes one string and expands it via ``prompt_optimizer``. The
    provider adapts the input itself — the caller never needs to know MiniMax is
    active. Use the JSON ``prompt`` field; fall back to the raw text for plain-text
    prompt files or JSON without a ``prompt`` field.
    """
    text = raw.strip()
    try:
        data = json.loads(text)
    except (ValueError, json.JSONDecodeError):
        return text
    if isinstance(data, dict):
        core = data.get("prompt")
        if isinstance(core, str) and core.strip():
            return core.strip()
    return text


def _generate_image_minimax(
    prompt: str, reference_images: list[str], output_file: str, aspect_ratio: str
) -> str:
    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        return "MINIMAX_API_KEY is not set"
    prompt = _minimax_prompt(prompt)
    if len(prompt) > MINIMAX_PROMPT_MAX_CHARS:
        return (
            f"Prompt is {len(prompt)} characters but MiniMax image-01 accepts at most "
            f"{MINIMAX_PROMPT_MAX_CHARS}. Shorten the prompt to stay within the limit; "
            f"reference images plus a tighter description usually recover the detail."
        )
    body = {
        "model": os.getenv("MINIMAX_IMAGE_MODEL", "image-01"),
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "response_format": "base64",
        "n": 1,
        "prompt_optimizer": True,
    }
    if reference_images:
        # Reference images are passed as character subjects as-is; unlike the Gemini
        # path we do not pre-validate them — invalid files surface as a MiniMax API error.
        body["subject_reference"] = [
            {"type": "character", "image_file": _to_data_url(p)} for p in reference_images
        ]
    response = requests.post(
        f"{_minimax_host()}/v1/image_generation",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    _check_base_resp(payload)
    images = (payload.get("data") or {}).get("image_base64") or []
    if not images:
        raise Exception("MiniMax returned no image data")
    _ensure_output_dir(output_file)
    with open(output_file, "wb") as f:
        f.write(base64.b64decode(images[0]))
    return f"Successfully generated image to {output_file}"


def _generate_image_gemini(
    prompt: str, reference_images: list[str], output_file: str, aspect_ratio: str
) -> str:
    parts = []
    valid_reference_images = []
    for ref_img in reference_images:
        if validate_image(ref_img):
            valid_reference_images.append(ref_img)
        else:
            print(f"Skipping invalid reference image: {ref_img}")
    if len(valid_reference_images) < len(reference_images):
        skipped = len(reference_images) - len(valid_reference_images)
        print(f"Note: {skipped} reference image(s) were skipped due to validation failure.")

    for reference_image in valid_reference_images:
        with open(reference_image, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
        parts.append({"inlineData": {"mimeType": "image/jpeg", "data": image_b64}})

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "GEMINI_API_KEY is not set"
    response = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-image-preview:generateContent",
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json={
            "generationConfig": {"imageConfig": {"aspectRatio": aspect_ratio}},
            "contents": [{"parts": [*parts, {"text": prompt}]}],
        },
    )
    response.raise_for_status()
    data = response.json()
    response_parts: list[dict] = data["candidates"][0]["content"]["parts"]
    image_parts = [part for part in response_parts if part.get("inlineData", False)]
    if len(image_parts) == 1:
        base64_image = image_parts[0]["inlineData"]["data"]
        _ensure_output_dir(output_file)
        with open(output_file, "wb") as f:
            f.write(base64.b64decode(base64_image))
        return f"Successfully generated image to {output_file}"
    raise Exception("Failed to generate image")



def _comfy_host() -> str:
    return os.getenv("COMFYUI_URL", COMFY_DEFAULT_HOST).rstrip("/")


def _comfy_prompt_text(raw: str) -> tuple[str, str]:
    text = raw.strip()
    negative = "blurry, low quality, distorted, deformed, watermark, text artifacts"
    try:
        data = json.loads(text)
    except (ValueError, json.JSONDecodeError):
        return text, negative
    if not isinstance(data, dict):
        return text, negative
    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        parts = []
        for key in ("subject", "style", "composition", "lighting", "color_palette", "background"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
        prompt = ", ".join(parts) if parts else text
    neg = data.get("negative_prompt")
    if isinstance(neg, str) and neg.strip():
        negative = neg.strip()
    return prompt.strip(), negative


def _comfy_dimensions(aspect_ratio: str) -> tuple[int, int]:
    mapping = {
        "1:1": (512, 512),
        "16:9": (768, 432),
        "9:16": (432, 768),
        "4:3": (704, 528),
        "3:4": (528, 704),
        "3:2": (768, 512),
        "2:3": (512, 768),
    }
    return mapping.get(aspect_ratio, mapping["1:1"])


def _qwen_image_dimensions(aspect_ratio: str) -> tuple[int, int]:
    return QWEN_IMAGE_DIMENSIONS.get(aspect_ratio, QWEN_IMAGE_DIMENSIONS["1:1"])


def _cap_dimensions(width: int, height: int, max_pixels: int | None) -> tuple[int, int]:
    if not max_pixels or width * height <= max_pixels:
        return width, height
    scale = (max_pixels / float(width * height)) ** 0.5
    capped_width = max(16, round(width * scale / 16) * 16)
    capped_height = max(16, round(height * scale / 16) * 16)
    return capped_width, capped_height


def _qwen_blank_dimensions(aspect_ratio: str) -> tuple[int, int]:
    base_w, base_h = _comfy_dimensions(aspect_ratio)
    target_pixels = 1024 * 1024
    scale = (target_pixels / float(base_w * base_h)) ** 0.5
    width = max(8, round(base_w * scale / 8) * 8)
    height = max(8, round(base_h * scale / 8) * 8)
    return width, height


def _create_qwen_blank_reference(aspect_ratio: str, output_file: str) -> str:
    from PIL import Image, ImageDraw

    width, height = _qwen_blank_dimensions(aspect_ratio)
    blank_dir = Path(output_file).parent if output_file else Path("/tmp")
    blank_dir.mkdir(parents=True, exist_ok=True)
    blank_path = blank_dir / f"qwen_blank_reference_{uuid.uuid4().hex[:12]}.png"
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    # A tiny low-contrast guide mark helps image-edit models understand this is
    # an editable canvas, while staying visually negligible for logo/poster work.
    mark = max(2, min(width, height) // 128)
    draw.rectangle((0, 0, mark, mark), fill=(250, 250, 250))
    image.save(blank_path)
    return str(blank_path)


def _build_comfy_sd15_workflow(prompt: str, negative: str, aspect_ratio: str) -> dict:
    width, height = _comfy_dimensions(aspect_ratio)
    ckpt = os.getenv("COMFYUI_CHECKPOINT", "v1-5-pruned-emaonly-fp16.safetensors")
    steps = int(os.getenv("COMFYUI_STEPS", "20"))
    cfg = float(os.getenv("COMFYUI_CFG", "7.0"))
    sampler = os.getenv("COMFYUI_SAMPLER", "euler")
    scheduler = os.getenv("COMFYUI_SCHEDULER", "normal")
    seed = int(os.getenv("COMFYUI_SEED", "0")) or int(time.time() * 1000) % 18446744073709551615
    prefix = f"deerflow_{uuid.uuid4().hex[:12]}"
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler, "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0], "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": prefix}},
    }



def _comfy_upload_image(image_path: str) -> str:
    host = _comfy_host()
    suffix = Path(image_path).suffix or ".jpg"
    remote_name = f"deerflow_{uuid.uuid4().hex[:12]}{suffix}"
    with open(image_path, "rb") as f:
        response = requests.post(
            f"{host}/upload/image",
            files={"image": (remote_name, f, _guess_mime(image_path))},
            data={"type": "input", "overwrite": "true"},
            timeout=120,
        )
    response.raise_for_status()
    payload = response.json()
    name = payload.get("name") or payload.get("filename")
    subfolder = payload.get("subfolder") or ""
    return f"{subfolder}/{name}" if subfolder else name



def _parse_bool(value: str | bool | None) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _qwen_option(options: dict | None, key: str, env_name: str, default, cast):
    if options and options.get(key) is not None:
        return cast(options[key])
    env_value = os.getenv(env_name)
    if env_value is not None and env_value != "":
        return cast(env_value)
    return default


def _qwen_option_multi(options: dict | None, key: str, env_names: tuple[str, ...], default, cast):
    if options and options.get(key) is not None:
        return cast(options[key])
    for env_name in env_names:
        env_value = os.getenv(env_name)
        if env_value is not None and env_value != "":
            return cast(env_value)
    return default


def _option_or_env_set(options: dict | None, key: str, env_names: tuple[str, ...]) -> bool:
    if options and options.get(key) is not None:
        return True
    return any(os.getenv(env_name) not in (None, "") for env_name in env_names)


def _qwen_seed(options: dict | None, specific_env: str = "COMFYUI_QWEN_SEED") -> int:
    seed_value = None
    if options and options.get("seed") is not None:
        seed_value = options["seed"]
    elif os.getenv(specific_env):
        seed_value = os.getenv(specific_env)
    elif os.getenv("COMFYUI_SEED"):
        seed_value = os.getenv("COMFYUI_SEED")
    return int(seed_value) if seed_value not in (None, "", "0", 0) else int(time.time() * 1000) % 18446744073709551615


def _qwen_use_lora(options: dict | None, env_name: str, default: bool) -> bool:
    requested_use_lora = (options or {}).get("use_lora") if options else None
    if requested_use_lora is not None:
        return bool(requested_use_lora)
    env_use_lora = _parse_bool(os.getenv(env_name))
    if env_use_lora is not None:
        return env_use_lora
    return default


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return float(value)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return int(value)


def _comfy_resource_snapshot() -> dict:
    host = _comfy_host()
    stats = requests.get(f"{host}/system_stats", timeout=10).json()
    queue = requests.get(f"{host}/queue", timeout=10).json()
    system = stats.get("system") or {}
    devices = stats.get("devices") or []
    device = devices[0] if devices else {}
    return {
        "ram_total": system.get("ram_total"),
        "ram_free": system.get("ram_free"),
        "device_name": device.get("name"),
        "vram_total": device.get("vram_total"),
        "vram_free": device.get("vram_free"),
        "torch_vram_total": device.get("torch_vram_total"),
        "torch_vram_free": device.get("torch_vram_free"),
        "queue_running": len(queue.get("queue_running") or []),
        "queue_pending": len(queue.get("queue_pending") or []),
    }


def _gb(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / GIB


def _format_gb(value: int | float | None) -> str:
    gb = _gb(value)
    return "unknown" if gb is None else f"{gb:.1f}GiB"


def _memory_budget_mode(options: dict | None) -> str:
    mode = (options or {}).get("memory_budget") or os.getenv("SPARK_IMAGE_MEMORY_BUDGET", "balanced")
    mode = str(mode).strip().lower()
    if mode not in {"off", "relaxed", "balanced", "conservative"}:
        raise ValueError("SPARK_IMAGE_MEMORY_BUDGET must be off, relaxed, balanced, or conservative")
    return mode


@contextlib.contextmanager
def _comfy_generation_lock():
    enabled = _parse_bool(os.getenv("SPARK_IMAGE_LOCK", "1"))
    if not enabled:
        yield
        return
    lock_path = os.getenv("SPARK_IMAGE_LOCK_FILE", "/tmp/deerflow_comfy_generation.lock")
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _apply_spark_memory_budget(kind: str, opts: dict, options: dict | None, aspect_ratio: str) -> dict:
    mode = _memory_budget_mode(options)
    if mode == "off":
        return opts

    snapshot = _comfy_resource_snapshot()
    print(
        "Spark memory snapshot: "
        f"ram_free={_format_gb(snapshot['ram_free'])}/{_format_gb(snapshot['ram_total'])}, "
        f"vram_free={_format_gb(snapshot['vram_free'])}/{_format_gb(snapshot['vram_total'])}, "
        f"torch_free={_format_gb(snapshot['torch_vram_free'])}/{_format_gb(snapshot['torch_vram_total'])}, "
        f"queue_running={snapshot['queue_running']}, queue_pending={snapshot['queue_pending']}"
    )

    max_queue = _int_env("SPARK_IMAGE_MAX_QUEUE_ITEMS", 0)
    queued = snapshot["queue_running"] + snapshot["queue_pending"]
    if queued > max_queue:
        raise RuntimeError(f"ComfyUI queue has {queued} item(s), above budget {max_queue}; wait or raise SPARK_IMAGE_MAX_QUEUE_ITEMS")

    ram_free_gb = _gb(snapshot["ram_free"])
    min_free_gb = _float_env(
        "SPARK_SYSTEM_RESERVE_GIB",
        _float_env("SPARK_IMAGE_MIN_SYSTEM_FREE_GB", 32.0),
    )
    low_free_gb = _float_env("SPARK_IMAGE_LOW_SYSTEM_FREE_GB", 56.0)
    if ram_free_gb is not None and ram_free_gb < min_free_gb:
        raise MemoryError(f"Spark free RAM is {ram_free_gb:.1f}GiB, below budget minimum {min_free_gb:.1f}GiB")

    should_conserve = mode == "conservative" or (ram_free_gb is not None and ram_free_gb < low_free_gb)
    if kind == "qwen_image":
        if mode == "relaxed":
            default_max_pixels = 0
        elif should_conserve:
            default_max_pixels = 1024 * 1024
        else:
            default_max_pixels = 0
        max_pixels = _int_env("SPARK_IMAGE_MAX_PIXELS", default_max_pixels)
        opts["max_pixels"] = max_pixels if max_pixels > 0 else None
        if should_conserve and not _option_or_env_set(options, "steps", ("COMFYUI_QWEN_IMAGE_STEPS", "COMFYUI_QWEN_STEPS")):
            opts["steps"] = min(opts["steps"], 20)
    elif kind == "qwen_edit":
        if should_conserve and not _option_or_env_set(options, "megapixels", ("COMFYUI_QWEN_MEGAPIXELS",)):
            opts["megapixels"] = min(float(opts["megapixels"] or 0.5), 0.35)
        if should_conserve and not _option_or_env_set(options, "steps", ("COMFYUI_QWEN_STEPS",)):
            opts["steps"] = min(opts["steps"], 20)
    elif kind == "qwen_layered":
        if should_conserve and not _option_or_env_set(options, "steps", ("COMFYUI_QWEN_LAYERED_STEPS",)):
            opts["steps"] = min(opts["steps"], 20)
    return opts


def _resolve_qwen_image_options(options: dict | None = None) -> dict:
    preset = (options or {}).get("preset") or os.getenv("COMFYUI_QWEN_IMAGE_PRESET") or os.getenv("COMFYUI_QWEN_PRESET", "quality")
    preset = str(preset).strip().lower()
    if preset not in QWEN_IMAGE_PRESETS:
        raise ValueError(f"Unknown Qwen preset: {preset!r} (use fast, balanced, or quality)")
    defaults = QWEN_IMAGE_PRESETS[preset]

    return {
        "preset": preset,
        "unet": os.getenv("COMFYUI_QWEN_IMAGE_UNET", "qwen_image_2512_fp8_e4m3fn.safetensors"),
        "clip": os.getenv("COMFYUI_QWEN_IMAGE_CLIP") or os.getenv("COMFYUI_QWEN_CLIP", "qwen_2.5_vl_7b_fp8_scaled.safetensors"),
        "vae": os.getenv("COMFYUI_QWEN_IMAGE_VAE") or os.getenv("COMFYUI_QWEN_VAE", "qwen_image_vae.safetensors"),
        "lora_name": _qwen_option_multi(options, "lora_name", ("COMFYUI_QWEN_IMAGE_LORA", "COMFYUI_QWEN_LORA"), "Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors", str),
        "lora_strength": _qwen_option_multi(options, "lora_strength", ("COMFYUI_QWEN_IMAGE_LORA_STRENGTH", "COMFYUI_QWEN_LORA_STRENGTH"), 1.0, float),
        "use_lora": _qwen_use_lora(options, "COMFYUI_QWEN_IMAGE_USE_LORA", defaults["use_lora"]),
        "weight_dtype": _qwen_option_multi(options, "weight_dtype", ("COMFYUI_QWEN_IMAGE_WEIGHT_DTYPE", "COMFYUI_QWEN_WEIGHT_DTYPE"), defaults["weight_dtype"], str),
        "steps": _qwen_option_multi(options, "steps", ("COMFYUI_QWEN_IMAGE_STEPS", "COMFYUI_QWEN_STEPS"), defaults["steps"], int),
        "cfg": _qwen_option_multi(options, "cfg", ("COMFYUI_QWEN_IMAGE_CFG", "COMFYUI_QWEN_CFG"), defaults["cfg"], float),
        "shift": _qwen_option_multi(options, "shift", ("COMFYUI_QWEN_IMAGE_SHIFT", "COMFYUI_QWEN_SHIFT"), 3.1, float),
        "sampler": _qwen_option_multi(options, "sampler", ("COMFYUI_QWEN_IMAGE_SAMPLER", "COMFYUI_QWEN_SAMPLER"), "euler", str),
        "scheduler": _qwen_option_multi(options, "scheduler", ("COMFYUI_QWEN_IMAGE_SCHEDULER", "COMFYUI_QWEN_SCHEDULER"), "simple", str),
        "seed": _qwen_seed(options, "COMFYUI_QWEN_IMAGE_SEED"),
        "max_pixels": None,
    }


def _build_comfy_qwen_image_workflow(prompt: str, negative: str, aspect_ratio: str, qwen_options: dict | None = None) -> dict:
    opts = _resolve_qwen_image_options(qwen_options)
    opts = _apply_spark_memory_budget("qwen_image", opts, qwen_options, aspect_ratio)
    width, height = _cap_dimensions(*_qwen_image_dimensions(aspect_ratio), opts.get("max_pixels"))
    prefix = f"deerflow_qwen_image_{uuid.uuid4().hex[:12]}"
    workflow: dict[str, dict] = {}

    def add(class_type: str, inputs: dict) -> str:
        node_id = str(len(workflow) + 1)
        workflow[node_id] = {"class_type": class_type, "inputs": inputs}
        return node_id

    unet_id = add("UNETLoader", {"unet_name": opts["unet"], "weight_dtype": opts["weight_dtype"]})
    model_ref = [unet_id, 0]
    if opts["use_lora"]:
        lora_id = add(
            "LoraLoaderModelOnly",
            {"model": model_ref, "lora_name": opts["lora_name"], "strength_model": opts["lora_strength"]},
        )
        model_ref = [lora_id, 0]

    sampling_id = add("ModelSamplingAuraFlow", {"model": model_ref, "shift": opts["shift"]})
    clip_id = add("CLIPLoader", {"clip_name": opts["clip"], "type": "qwen_image", "device": "default"})
    vae_id = add("VAELoader", {"vae_name": opts["vae"]})
    positive_id = add("CLIPTextEncode", {"clip": [clip_id, 0], "text": prompt})
    negative_id = add("CLIPTextEncode", {"clip": [clip_id, 0], "text": negative or ""})
    latent_id = add("EmptySD3LatentImage", {"width": width, "height": height, "batch_size": 1})
    sample_id = add(
        "KSampler",
        {
            "model": [sampling_id, 0],
            "seed": opts["seed"],
            "steps": opts["steps"],
            "cfg": opts["cfg"],
            "sampler_name": opts["sampler"],
            "scheduler": opts["scheduler"],
            "positive": [positive_id, 0],
            "negative": [negative_id, 0],
            "latent_image": [latent_id, 0],
            "denoise": 1.0,
        },
    )
    decoded_id = add("VAEDecode", {"samples": [sample_id, 0], "vae": [vae_id, 0]})
    add("SaveImage", {"images": [decoded_id, 0], "filename_prefix": prefix})
    return workflow


def _resolve_qwen_edit_options(options: dict | None = None) -> dict:
    preset = (options or {}).get("preset") or os.getenv("COMFYUI_QWEN_PRESET", "balanced")
    preset = str(preset).strip().lower()
    if preset not in QWEN_EDIT_PRESETS:
        raise ValueError(f"Unknown Qwen preset: {preset!r} (use fast, balanced, or quality)")
    defaults = QWEN_EDIT_PRESETS[preset]

    use_lora = _qwen_use_lora(options, "COMFYUI_QWEN_USE_LORA", defaults["use_lora"])

    if options and options.get("megapixels") is not None:
        megapixels = float(options["megapixels"])
    elif os.getenv("COMFYUI_QWEN_MEGAPIXELS"):
        megapixels = float(os.getenv("COMFYUI_QWEN_MEGAPIXELS", "0"))
    else:
        megapixels = float(defaults["megapixels"])
    if megapixels <= 0:
        megapixels = None

    return {
        "preset": preset,
        "unet": os.getenv("COMFYUI_QWEN_UNET", "qwen_image_edit_2509_fp8_e4m3fn.safetensors"),
        "clip": os.getenv("COMFYUI_QWEN_CLIP", "qwen_2.5_vl_7b_fp8_scaled.safetensors"),
        "vae": os.getenv("COMFYUI_QWEN_VAE", "qwen_image_vae.safetensors"),
        "lora_name": _qwen_option(options, "lora_name", "COMFYUI_QWEN_LORA", "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors", str),
        "lora_strength": _qwen_option(options, "lora_strength", "COMFYUI_QWEN_LORA_STRENGTH", 1.0, float),
        "use_lora": use_lora,
        "weight_dtype": _qwen_option(options, "weight_dtype", "COMFYUI_QWEN_WEIGHT_DTYPE", defaults["weight_dtype"], str),
        "steps": _qwen_option(options, "steps", "COMFYUI_QWEN_STEPS", defaults["steps"], int),
        "cfg": _qwen_option(options, "cfg", "COMFYUI_QWEN_CFG", defaults["cfg"], float),
        "denoise": _qwen_option(options, "denoise", "COMFYUI_QWEN_DENOISE", defaults["denoise"], float),
        "shift": _qwen_option(options, "shift", "COMFYUI_QWEN_SHIFT", 3.0, float),
        "megapixels": megapixels,
        "sampler": _qwen_option(options, "sampler", "COMFYUI_QWEN_SAMPLER", "euler", str),
        "scheduler": _qwen_option(options, "scheduler", "COMFYUI_QWEN_SCHEDULER", "simple", str),
        "seed": _qwen_seed(options),
    }


def _build_comfy_qwen_edit_workflow(prompt: str, uploaded_images: list[str], qwen_options: dict | None = None, negative: str = "") -> dict:
    if not uploaded_images:
        raise ValueError("Qwen Image Edit requires at least one uploaded reference image.")

    opts = _resolve_qwen_edit_options(qwen_options)
    opts = _apply_spark_memory_budget("qwen_edit", opts, qwen_options, "reference")
    prefix = f"deerflow_qwen_edit_{uuid.uuid4().hex[:12]}"
    workflow: dict[str, dict] = {}

    def add(class_type: str, inputs: dict) -> str:
        node_id = str(len(workflow) + 1)
        workflow[node_id] = {"class_type": class_type, "inputs": inputs}
        return node_id

    image_refs = []
    for uploaded_image in uploaded_images[:3]:
        image_id = add("LoadImage", {"image": uploaded_image})
        if opts["megapixels"] is not None:
            image_id = add(
                "ImageScaleToTotalPixels",
                {
                    "image": [image_id, 0],
                    "upscale_method": "lanczos",
                    "megapixels": opts["megapixels"],
                    "resolution_steps": 1,
                },
            )
        image_refs.append([image_id, 0])

    unet_id = add("UNETLoader", {"unet_name": opts["unet"], "weight_dtype": opts["weight_dtype"]})
    model_ref = [unet_id, 0]
    if opts["use_lora"]:
        lora_id = add(
            "LoraLoaderModelOnly",
            {"model": model_ref, "lora_name": opts["lora_name"], "strength_model": opts["lora_strength"]},
        )
        model_ref = [lora_id, 0]

    sampling_id = add("ModelSamplingAuraFlow", {"model": model_ref, "shift": opts["shift"]})
    cfgnorm_id = add("CFGNorm", {"model": [sampling_id, 0], "strength": 1.0})
    clip_id = add("CLIPLoader", {"clip_name": opts["clip"], "type": "qwen_image", "device": "default"})
    vae_id = add("VAELoader", {"vae_name": opts["vae"]})

    positive_inputs = {"clip": [clip_id, 0], "vae": [vae_id, 0], "prompt": prompt}
    negative_inputs = {"clip": [clip_id, 0], "vae": [vae_id, 0], "prompt": negative or ""}
    for index, image_ref in enumerate(image_refs, start=1):
        positive_inputs[f"image{index}"] = image_ref
        negative_inputs[f"image{index}"] = image_ref
    positive_id = add("TextEncodeQwenImageEditPlus", positive_inputs)
    negative_id = add("TextEncodeQwenImageEditPlus", negative_inputs)
    positive_id = add("FluxKontextMultiReferenceLatentMethod", {"conditioning": [positive_id, 0], "reference_latents_method": "index_timestep_zero"})
    negative_id = add("FluxKontextMultiReferenceLatentMethod", {"conditioning": [negative_id, 0], "reference_latents_method": "index_timestep_zero"})
    scaled_source_id = add("FluxKontextImageScale", {"image": image_refs[0]})
    latent_id = add("VAEEncode", {"pixels": [scaled_source_id, 0], "vae": [vae_id, 0]})
    sample_id = add(
        "KSampler",
        {
            "model": [cfgnorm_id, 0],
            "seed": opts["seed"],
            "steps": opts["steps"],
            "cfg": opts["cfg"],
            "sampler_name": opts["sampler"],
            "scheduler": opts["scheduler"],
            "positive": [positive_id, 0],
            "negative": [negative_id, 0],
            "latent_image": [latent_id, 0],
            "denoise": opts["denoise"],
        },
    )
    decoded_id = add("VAEDecode", {"samples": [sample_id, 0], "vae": [vae_id, 0]})
    add("SaveImage", {"images": [decoded_id, 0], "filename_prefix": prefix})
    return workflow


def _resolve_qwen_layered_options(options: dict | None = None) -> dict:
    policy = spark_scheduler.load_policy()
    defaults = policy["tasks"]["layered"]
    preset = (options or {}).get("preset") or os.getenv("COMFYUI_QWEN_LAYERED_PRESET", "balanced")
    preset = str(preset).strip().lower()
    if preset not in QWEN_LAYERED_PRESETS:
        raise ValueError(f"Unknown Qwen Layered preset: {preset!r} (use fast, balanced, or quality)")
    sampling = QWEN_LAYERED_PRESETS[preset]
    resolution = _qwen_option(
        options,
        "resolution",
        "COMFYUI_QWEN_LAYERED_RESOLUTION",
        defaults["default_resolution"],
        int,
    )
    layers = _qwen_option(
        options,
        "layers",
        "COMFYUI_QWEN_LAYERED_LAYERS",
        defaults["default_layers"],
        int,
    )
    allow_large = _parse_bool(os.getenv("SPARK_LAYERED_ALLOW_LARGE", "0"))
    if not allow_large and resolution > defaults["max_automatic_resolution"]:
        raise ValueError(
            f"Layered automatic resolution is limited to {defaults['max_automatic_resolution']}px; "
            "set SPARK_LAYERED_ALLOW_LARGE=1 only for a supervised experiment"
        )
    if not allow_large and layers > defaults["max_automatic_layers"]:
        raise ValueError(
            f"Layered automatic output is limited to {defaults['max_automatic_layers']} layers; "
            "set SPARK_LAYERED_ALLOW_LARGE=1 only for a supervised experiment"
        )
    if resolution < 16 or resolution % 16:
        raise ValueError("Qwen Image Layered resolution must be at least 16 and divisible by 16")
    if layers < 1:
        raise ValueError("Qwen Image Layered requires at least one layer")
    resolved = {
        "preset": preset,
        "unet": os.getenv("COMFYUI_QWEN_LAYERED_UNET", defaults["diffusion_model"]),
        "clip": os.getenv("COMFYUI_QWEN_LAYERED_CLIP", defaults["text_encoder"]),
        "vae": os.getenv("COMFYUI_QWEN_LAYERED_VAE", defaults["vae"]),
        "weight_dtype": _qwen_option(options, "weight_dtype", "COMFYUI_QWEN_LAYERED_WEIGHT_DTYPE", "default", str),
        "steps": _qwen_option(options, "steps", "COMFYUI_QWEN_LAYERED_STEPS", sampling["steps"], int),
        "cfg": _qwen_option(options, "cfg", "COMFYUI_QWEN_LAYERED_CFG", sampling["cfg"], float),
        "shift": _qwen_option(options, "shift", "COMFYUI_QWEN_LAYERED_SHIFT", 1.0, float),
        "seed": _qwen_seed(options, "COMFYUI_QWEN_LAYERED_SEED"),
        "sampler": _qwen_option(options, "sampler", "COMFYUI_QWEN_LAYERED_SAMPLER", "euler", str),
        "scheduler": _qwen_option(options, "scheduler", "COMFYUI_QWEN_LAYERED_SCHEDULER", "simple", str),
        "resolution": resolution,
        "layers": layers,
        "batch_size": 1,
    }
    return _apply_spark_memory_budget("qwen_layered", resolved, options, "layered")


def _build_comfy_qwen_layered_workflow(
    prompt: str,
    uploaded_image: str | None,
    qwen_options: dict | None = None,
) -> tuple[dict, dict]:
    opts = _resolve_qwen_layered_options(qwen_options)
    workflow = {}
    counter = 1

    def add(class_type: str, inputs: dict) -> str:
        nonlocal counter
        node_id = str(counter)
        counter += 1
        workflow[node_id] = {"class_type": class_type, "inputs": inputs}
        return node_id

    unet_id = add("UNETLoader", {"unet_name": opts["unet"], "weight_dtype": opts["weight_dtype"]})
    model_id = add("ModelSamplingAuraFlow", {"model": [unet_id, 0], "shift": opts["shift"]})
    clip_id = add("CLIPLoader", {"clip_name": opts["clip"], "type": "qwen_image", "device": "default"})
    vae_id = add("VAELoader", {"vae_name": opts["vae"]})
    positive_id = add("CLIPTextEncode", {"text": prompt, "clip": [clip_id, 0]})
    negative_id = add("CLIPTextEncode", {"text": "", "clip": [clip_id, 0]})

    if uploaded_image:
        load_id = add("LoadImage", {"image": uploaded_image})
        scale_id = add(
            "ImageScaleToMaxDimension",
            {"image": [load_id, 0], "upscale_method": "lanczos", "largest_size": opts["resolution"]},
        )
        source_latent_id = add("VAEEncode", {"pixels": [scale_id, 0], "vae": [vae_id, 0]})
        positive_id = add(
            "ReferenceLatent",
            {"conditioning": [positive_id, 0], "latent": [source_latent_id, 0]},
        )
        negative_id = add(
            "ReferenceLatent",
            {"conditioning": [negative_id, 0], "latent": [source_latent_id, 0]},
        )
        size_id = add("GetImageSize", {"image": [scale_id, 0]})
        latent_inputs = {
            "width": [size_id, 0],
            "height": [size_id, 1],
            "layers": opts["layers"],
            "batch_size": 1,
        }
        mode = "image-to-layers"
    else:
        latent_inputs = {
            "width": opts["resolution"],
            "height": opts["resolution"],
            "layers": opts["layers"],
            "batch_size": 1,
        }
        mode = "text-to-layers"

    latent_id = add("EmptyQwenImageLayeredLatentImage", latent_inputs)
    sample_id = add(
        "KSampler",
        {
            "model": [model_id, 0],
            "seed": opts["seed"],
            "steps": opts["steps"],
            "cfg": opts["cfg"],
            "sampler_name": opts["sampler"],
            "scheduler": opts["scheduler"],
            "positive": [positive_id, 0],
            "negative": [negative_id, 0],
            "latent_image": [latent_id, 0],
            "denoise": 1.0,
        },
    )
    batch_id = add("LatentCutToBatch", {"samples": [sample_id, 0], "dim": "t", "slice_size": 1})
    decoded_id = add("VAEDecode", {"samples": [batch_id, 0], "vae": [vae_id, 0]})
    prefix = f"deerflow_qwen_layered_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    add("SaveImage", {"images": [decoded_id, 0], "filename_prefix": prefix})
    return workflow, {**opts, "mode": mode}


def _download_comfy_images(images: list[dict], output_file: str) -> list[dict]:
    host = _comfy_host()
    downloaded = []
    for index, image in enumerate(images):
        query = urlencode(
            {
                "filename": image["filename"],
                "subfolder": image.get("subfolder", ""),
                "type": image.get("type", "output"),
            }
        )
        response = requests.get(f"{host}/view?{query}", timeout=60)
        response.raise_for_status()
        if index == 0:
            destination = Path(output_file)
        else:
            destination = Path(output_file).with_name(
                f"{Path(output_file).stem}.layer-{index:02d}{Path(output_file).suffix or '.png'}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.content)
        downloaded.append(
            {
                "index": index,
                "role": "composite" if index == 0 else "rgba_layer",
                "path": str(destination),
                "comfy_filename": image["filename"],
            }
        )
    return downloaded


def _run_comfy_workflow_all(workflow: dict) -> tuple[str, list[dict]]:
    host = _comfy_host()
    client_id = f"deerflow-{uuid.uuid4().hex}"
    queued = requests.post(f"{host}/prompt", json={"prompt": workflow, "client_id": client_id}, timeout=30)
    queued.raise_for_status()
    prompt_id = queued.json()["prompt_id"]
    deadline = time.time() + int(os.getenv("COMFYUI_TIMEOUT", "900"))
    history = None
    while time.time() < deadline:
        response = requests.get(f"{host}/history/{prompt_id}", timeout=30)
        response.raise_for_status()
        data = response.json()
        if prompt_id in data:
            history = data[prompt_id]
            break
        time.sleep(1)
    if history is None:
        raise TimeoutError(f"Timed out waiting for ComfyUI prompt {prompt_id}")
    status = history.get("status") or {}
    if status.get("status_str") == "error":
        messages = status.get("messages") or []
        raise Exception(f"ComfyUI workflow failed: {messages[-1] if messages else status}")
    images = []
    for node in (history.get("outputs") or {}).values():
        images.extend(node.get("images") or [])
    if not images:
        raise Exception("ComfyUI generated no images")
    return prompt_id, images

def _run_comfy_workflow(workflow: dict, output_file: str) -> str:
    _, images = _run_comfy_workflow_all(workflow)
    downloaded = _download_comfy_images(images[:1], output_file)
    return downloaded[0]["comfy_filename"]



def _generate_image_comfy_qwen_edit(prompt: str, reference_images: list[str], output_file: str, aspect_ratio: str, qwen_options: dict | None = None) -> str:
    prompt_text, negative = _comfy_prompt_text(prompt)
    effective_reference_images = list(reference_images)
    if not effective_reference_images:
        raise ValueError(
            "Qwen Image Edit requires at least one reference image. "
            "For pure text-to-image generation, use IMAGE_GENERATION_PROVIDER=comfy "
            "without --reference-images so the script can route to the text-to-image fallback."
        )
    if len(effective_reference_images) > 3:
        print("Warning: Qwen Image Edit works best with 1-3 input images; using the first 3 reference images.")
    uploaded_images = [_comfy_upload_image(path) for path in effective_reference_images[:3]]
    workflow = _build_comfy_qwen_edit_workflow(prompt_text, uploaded_images, qwen_options=qwen_options, negative=negative)
    filename = _run_comfy_workflow(workflow, output_file)
    preset = (qwen_options or {}).get("preset") or os.getenv("COMFYUI_QWEN_PRESET", "balanced")
    return f"Successfully generated image edit to {output_file} via local Qwen Image Edit preset={preset} ({filename})"


def _generate_image_comfy_qwen_image(prompt: str, reference_images: list[str], output_file: str, aspect_ratio: str, qwen_options: dict | None = None) -> str:
    if reference_images:
        print("Warning: local Qwen Image text-to-image provider ignores reference images.")
    prompt_text, negative = _comfy_prompt_text(prompt)
    workflow = _build_comfy_qwen_image_workflow(prompt_text, negative, aspect_ratio, qwen_options=qwen_options)
    filename = _run_comfy_workflow(workflow, output_file)
    preset = (qwen_options or {}).get("preset") or os.getenv("COMFYUI_QWEN_IMAGE_PRESET") or os.getenv("COMFYUI_QWEN_PRESET", "quality")
    return f"Successfully generated image to {output_file} via local Qwen Image 2512 preset={preset} ({filename})"


def _generate_image_comfy_qwen_layered(
    prompt: str,
    reference_images: list[str],
    output_file: str,
    qwen_options: dict | None = None,
) -> str:
    prompt_text, _ = _comfy_prompt_text(prompt)
    if len(reference_images) > 1:
        print("Warning: Qwen Image Layered decomposition uses only the first reference image.")
    uploaded_image = _comfy_upload_image(reference_images[0]) if reference_images else None
    workflow, opts = _build_comfy_qwen_layered_workflow(prompt_text, uploaded_image, qwen_options=qwen_options)
    prompt_id, images = _run_comfy_workflow_all(workflow)
    downloaded = _download_comfy_images(images, output_file)
    manifest_path = Path(output_file).with_name(f"{Path(output_file).stem}.layers.json")
    manifest = {
        "prompt_id": prompt_id,
        "task": "layered",
        "mode": opts["mode"],
        "model": opts["unet"],
        "layers_requested": opts["layers"],
        "resolution": opts["resolution"],
        "files": downloaded,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return (
        f"Successfully generated {len(downloaded) - 1} RGBA layer(s) plus composite via local "
        f"Qwen Image Layered; manifest={manifest_path}"
    )


def _resolve_local_task(task: str, reference_images: list[str]) -> str:
    normalized = str(task or "auto").strip().lower()
    if normalized == "auto":
        return "edit" if reference_images else "generate"
    if normalized not in {"generate", "edit", "layered"}:
        raise ValueError("task must be auto, generate, edit, or layered")
    if normalized == "edit" and not reference_images:
        raise ValueError("The edit task requires at least one reference image")
    return normalized


def _prepare_spark_task(task: str, dry_run: bool = False) -> dict:
    schedule = spark_scheduler.plan(task, comfy_url=_comfy_host())
    print(
        "Spark schedule: "
        f"task={task}, admitted={schedule['admitted']}, resident={schedule['resident_model'] or 'none'}, "
        f"combined={schedule['estimated_reservation_gib']['combined_gib']:.1f}GiB/"
        f"{schedule['budget']['ai_memory_gib']:.1f}GiB, actions={','.join(schedule['required_actions']) or 'none'}"
    )
    if not schedule["admitted"]:
        raise MemoryError(schedule["rejection_reason"] or "Spark image task was rejected by resource policy")
    if not dry_run and "release_comfy_before_load" in schedule["required_actions"]:
        spark_scheduler.release_comfy(comfy_url=_comfy_host())
        time.sleep(1)
    return schedule


def _release_layered_models() -> None:
    try:
        spark_scheduler.release_comfy(comfy_url=_comfy_host())
    except Exception as exc:
        print(f"Warning: unable to release Layered models after the job: {exc}", file=sys.stderr)


def _generate_image_comfy_sd15(prompt: str, reference_images: list[str], output_file: str, aspect_ratio: str) -> str:
    if reference_images:
        print("Warning: local ComfyUI SD1.5 provider currently ignores reference images.")
    prompt_text, negative = _comfy_prompt_text(prompt)
    workflow = _build_comfy_sd15_workflow(prompt_text, negative, aspect_ratio)
    filename = _run_comfy_workflow(workflow, output_file)
    return f"Successfully generated image to {output_file} via local ComfyUI SD1.5 ({filename})"

def generate_image(
    prompt_file: str,
    reference_images: list[str],
    output_file: str,
    aspect_ratio: str = "16:9",
    qwen_options: dict | None = None,
    task: str = "auto",
    dry_run: bool = False,
) -> str:
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read()
    provider = _resolve_provider(
        "IMAGE_GENERATION_PROVIDER", "gemini", bool(os.getenv("GEMINI_API_KEY"))
    )
    provider_tasks = {
        "comfy_qwen_image": "generate",
        "qwen_image": "generate",
        "qwen-image": "generate",
        "comfy_qwen_edit": "edit",
        "qwen_image_edit": "edit",
        "qwen-edit": "edit",
        "comfy_qwen_layered": "layered",
        "qwen_image_layered": "layered",
        "qwen-layered": "layered",
    }
    explicit_provider_task = provider_tasks.get(provider)
    selected_task = _resolve_local_task(task, reference_images)
    if explicit_provider_task and task not in (None, "", "auto") and selected_task != explicit_provider_task:
        raise ValueError(f"Provider {provider!r} conflicts with --task {selected_task!r}")
    if explicit_provider_task:
        selected_task = explicit_provider_task

    if dry_run and (provider == "comfy" or explicit_provider_task):
        with _comfy_generation_lock():
            return json.dumps(_prepare_spark_task(selected_task, dry_run=True), ensure_ascii=False, indent=2)
    if provider in ("comfy_qwen_edit", "qwen_image_edit", "qwen-edit"):
        with _comfy_generation_lock():
            _prepare_spark_task("edit")
            return _generate_image_comfy_qwen_edit(prompt, reference_images, output_file, aspect_ratio, qwen_options=qwen_options)
    if provider in ("comfy_qwen_image", "qwen_image", "qwen-image"):
        with _comfy_generation_lock():
            _prepare_spark_task("generate")
            return _generate_image_comfy_qwen_image(prompt, reference_images, output_file, aspect_ratio, qwen_options=qwen_options)
    if provider in ("comfy_qwen_layered", "qwen_image_layered", "qwen-layered"):
        with _comfy_generation_lock():
            schedule = _prepare_spark_task("layered")
            try:
                return _generate_image_comfy_qwen_layered(prompt, reference_images, output_file, qwen_options=qwen_options)
            finally:
                if "release_comfy_after_job" in schedule["required_actions"]:
                    _release_layered_models()
    if provider == "comfy":
        with _comfy_generation_lock():
            schedule = _prepare_spark_task(selected_task)
            if selected_task == "edit":
                return _generate_image_comfy_qwen_edit(prompt, reference_images, output_file, aspect_ratio, qwen_options=qwen_options)
            if selected_task == "generate":
                return _generate_image_comfy_qwen_image(prompt, reference_images, output_file, aspect_ratio, qwen_options=qwen_options)
            try:
                return _generate_image_comfy_qwen_layered(prompt, reference_images, output_file, qwen_options=qwen_options)
            finally:
                if "release_comfy_after_job" in schedule["required_actions"]:
                    _release_layered_models()
    if provider in ("comfy_sd15", "sd15", "sd-1.5"):
        with _comfy_generation_lock():
            return _generate_image_comfy_sd15(prompt, reference_images, output_file, aspect_ratio)
    if provider == "minimax":
        return _generate_image_minimax(prompt, reference_images, output_file, aspect_ratio)
    if provider in ("gemini", "google"):
        return _generate_image_gemini(prompt, reference_images, output_file, aspect_ratio)
    raise ValueError(f"Unknown image provider: {provider!r} (use 'gemini', 'minimax', 'comfy', 'comfy_qwen_image', 'comfy_qwen_edit', 'comfy_qwen_layered', or 'comfy_sd15')")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate images using Gemini, MiniMax, or local Qwen/ComfyUI")
    parser.add_argument("--prompt-file", required=True, help="Absolute path to prompt file")
    parser.add_argument("--reference-images", nargs="*", default=[],
                        help="Absolute paths to reference images (space-separated)")
    parser.add_argument("--output-file", required=True, help="Output path for generated image")
    parser.add_argument("--aspect-ratio", required=False, default="16:9",
                        help="Aspect ratio of the generated image")
    parser.add_argument("--task", choices=["auto", "generate", "edit", "layered"], default="auto",
                        help="Local Qwen task. Auto selects edit when references exist, otherwise generate.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the Spark scheduling decision without queueing a ComfyUI workflow")
    parser.add_argument("--qwen-preset", choices=["fast", "balanced", "quality"], default=None,
                        help="Qwen preset. Qwen Image defaults to quality; Qwen Image Edit 2509 defaults to balanced.")
    parser.add_argument("--qwen-steps", type=int, default=None, help="Qwen KSampler steps override")
    parser.add_argument("--qwen-cfg", type=float, default=None, help="Qwen KSampler CFG override")
    parser.add_argument("--qwen-denoise", type=float, default=None, help="Qwen KSampler denoise override")
    parser.add_argument("--qwen-seed", type=int, default=None, help="Qwen seed override; omit or 0 for random")
    parser.add_argument("--qwen-sampler", default=None, help="Qwen sampler override, e.g. euler")
    parser.add_argument("--qwen-scheduler", default=None, help="Qwen scheduler override, e.g. simple")
    parser.add_argument("--qwen-shift", type=float, default=None, help="Qwen ModelSamplingAuraFlow shift override")
    parser.add_argument("--qwen-megapixels", type=float, default=None,
                        help="Optional pre-scale megapixels before Qwen conditioning/VAE. Defaults come from the selected preset.")
    parser.add_argument("--qwen-layers", type=int, default=None,
                        help="Qwen Image Layered RGBA layer count; automatic jobs are capped at 4")
    parser.add_argument("--qwen-resolution", type=int, default=None,
                        help="Qwen Image Layered maximum dimension; automatic jobs default to and are capped at 640")
    parser.add_argument("--qwen-weight-dtype", default=None,
                        help="Qwen UNETLoader weight_dtype override. Defaults to 'default'; use fp8_e4m3fn_fast only as an explicit preview/experimental choice.")
    parser.add_argument("--qwen-lora-name", default=None, help="Qwen Lightning LoRA filename override")
    parser.add_argument("--qwen-lora-strength", type=float, default=None, help="Qwen LoRA strength override")
    parser.add_argument("--memory-budget", choices=["off", "relaxed", "balanced", "conservative"], default=None,
                        help="Spark/ComfyUI memory guard. Defaults to SPARK_IMAGE_MEMORY_BUDGET or balanced.")
    lora_group = parser.add_mutually_exclusive_group()
    lora_group.add_argument("--qwen-use-lora", dest="qwen_use_lora", action="store_true", default=None,
                            help="Force-enable Qwen Lightning LoRA")
    lora_group.add_argument("--qwen-no-lora", dest="qwen_use_lora", action="store_false",
                            help="Force-disable Qwen Lightning LoRA")
    args = parser.parse_args()

    qwen_options = {
        "preset": args.qwen_preset,
        "steps": args.qwen_steps,
        "cfg": args.qwen_cfg,
        "denoise": args.qwen_denoise,
        "seed": args.qwen_seed,
        "sampler": args.qwen_sampler,
        "scheduler": args.qwen_scheduler,
        "shift": args.qwen_shift,
        "megapixels": args.qwen_megapixels,
        "layers": args.qwen_layers,
        "resolution": args.qwen_resolution,
        "weight_dtype": args.qwen_weight_dtype,
        "lora_name": args.qwen_lora_name,
        "lora_strength": args.qwen_lora_strength,
        "use_lora": args.qwen_use_lora,
        "memory_budget": args.memory_budget,
    }

    try:
        print(generate_image(args.prompt_file, args.reference_images,
                             args.output_file, args.aspect_ratio,
                             qwen_options=qwen_options,
                             task=args.task,
                             dry_run=args.dry_run))
    except Exception as e:
        print(f"Error while generating image: {e}")
        sys.exit(1)
