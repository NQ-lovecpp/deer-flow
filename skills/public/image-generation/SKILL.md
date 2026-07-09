---
name: image-generation
description: Use this skill when the user requests image generation, text-to-image, image editing, product/portrait/scene visuals, reference-image composition, or local ComfyUI/Qwen workflows. Defaults to local Qwen Image for generation and Qwen Image Edit 2509 for image modification, with Qwen Image Layered available as an explicit experimental workflow.
---

# Image Generation Skill

## Default Route

Use local ComfyUI first unless the user explicitly asks for a remote provider.

- Text-to-image with no reference images: use Qwen Image 2512 through `IMAGE_GENERATION_PROVIDER=comfy` or `comfy_qwen_image`.
- Image modification with one to three real reference images: use Qwen Image Edit 2509 through `IMAGE_GENERATION_PROVIDER=comfy` or `comfy_qwen_edit`.
- Layer decomposition or layered generation: treat Qwen Image Layered as experimental. Use the bundled UI workflow in ComfyUI first; do not claim the CLI path is production-ready until `generate.py` has a layered API builder.
- Do not use Qwen Image Edit with a blank dummy canvas for pure text-to-image. Omit `--reference-images` so the script routes to Qwen Image.

Prefer plain `.txt` prompt files. JSON prompt files are accepted for compatibility, but local Qwen sampling controls must be CLI flags, not JSON fields.

## Required Preflight

Before local generation, inspect Spark/Comfy state:

```bash
COMFYUI_URL=http://127.0.0.1:8188 \
  python /mnt/skills/public/image-generation/scripts/comfy_resource_status.py
```

Scheduling rules:

- Do not start a job when ComfyUI has a running or pending queue item unless the user explicitly asks for parallel generation.
- Keep `--memory-budget balanced` by default.
- Use `--memory-budget conservative` for batches, background work, or when free system RAM is below 56GiB.
- Use `--memory-budget relaxed` only for one foreground high-quality job after checking the queue.
- Use `--memory-budget off` only for debugging.

Budget environment defaults enforced by `generate.py`:

- `SPARK_IMAGE_MEMORY_BUDGET=balanced`
- `SPARK_IMAGE_MIN_SYSTEM_FREE_GB=32`
- `SPARK_IMAGE_LOW_SYSTEM_FREE_GB=56`
- `SPARK_IMAGE_MAX_QUEUE_ITEMS=0`
- `SPARK_IMAGE_LOCK=1`

## Model Defaults

Use these Spark defaults:

- Qwen Image 2512 diffusion: `qwen_image_2512_fp8_e4m3fn.safetensors`
- Qwen Image Edit 2509 diffusion: `qwen_image_edit_2509_fp8_e4m3fn.safetensors`
- Qwen Image Edit 2509 Lightning LoRA for explicit fast previews: `Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors`
- Qwen Image Layered diffusion: `qwen_image_layered_fp8mixed.safetensors`
- Shared text encoder: `qwen_2.5_vl_7b_fp8_scaled.safetensors`
- Standard VAE: `qwen_image_vae.safetensors`
- Layered VAE: `qwen_image_layered_vae.safetensors`

Prefer FP8-family weights on Spark. Use bf16 only for deliberate comparison or debugging. Treat 2511 as an explicit fallback via `COMFYUI_QWEN_UNET=qwen_image_edit_2511_fp8mixed.safetensors`, not the default edit route.

## Generate A New Image

Create a prompt file, then call `generate.py` without reference images:

```bash
cat > /mnt/user-data/workspace/image.txt <<'EOF'
A cinematic product photo of a matte black espresso machine on a stainless counter, morning side light, realistic reflections, premium editorial photography, no text.
EOF

IMAGE_GENERATION_PROVIDER=comfy \
  python /mnt/skills/public/image-generation/scripts/generate.py \
    --prompt-file /mnt/user-data/workspace/image.txt \
    --output-file /mnt/user-data/outputs/image.png \
    --aspect-ratio 4:3 \
    --qwen-preset quality \
    --memory-budget balanced
```

Qwen Image defaults: `quality` preset, about 50 steps, CFG 4.0, official Qwen Image dimensions for the requested aspect ratio.

## Edit Or Compose Images

Pass the main image as the first reference. Pass up to two more images as secondary references.

```bash
cat > /mnt/user-data/workspace/edit.txt <<'EOF'
Replace the background with a clean bright kitchen. Keep the product shape, logo placement, camera angle, and realistic reflections unchanged.
EOF

IMAGE_GENERATION_PROVIDER=comfy_qwen_edit \
  python /mnt/skills/public/image-generation/scripts/generate.py \
    --prompt-file /mnt/user-data/workspace/edit.txt \
    --reference-images /mnt/user-data/uploads/input.png \
    --output-file /mnt/user-data/outputs/edited.png \
    --aspect-ratio 4:3 \
    --qwen-preset balanced \
    --memory-budget balanced
```

For multi-image composition:

```bash
IMAGE_GENERATION_PROVIDER=comfy_qwen_edit \
  python /mnt/skills/public/image-generation/scripts/generate.py \
    --prompt-file /mnt/user-data/workspace/composite.txt \
    --reference-images /mnt/user-data/uploads/person.jpg /mnt/user-data/uploads/product.png /mnt/user-data/uploads/scene.jpg \
    --output-file /mnt/user-data/outputs/composite.png \
    --qwen-preset balanced
```

Qwen Image Edit 2509 defaults: `balanced` preset, 20 steps, CFG 4.0, denoise 1.0, 0.5MP input scaling, LoRA off. Use `--qwen-use-lora --qwen-preset fast` only for explicit quick preview experiments.

## Layered Experiments

Use Qwen Image Layered only when the user asks for layers, decomposition, foreground/background separation, or editable layered outputs.

Current status:

- Model files are listed in `workflows/comfy/model-manifest.json`.
- The UI workflow is `workflows/comfy/ui/qwen-image-layered.json`.
- The ComfyUI node `EmptyQwenImageLayeredLatentImage` is available on Spark.
- The CLI script does not yet provide a stable layered API builder. If `IMAGE_GENERATION_PROVIDER=comfy_qwen_layered` is requested, report that the model should be tried through the bundled ComfyUI workflow first.

## Controls

Useful CLI controls:

- `--qwen-preset fast|balanced|quality`
- `--qwen-steps`
- `--qwen-cfg`
- `--qwen-denoise`
- `--qwen-seed`
- `--qwen-sampler`
- `--qwen-scheduler`
- `--qwen-shift`
- `--qwen-megapixels`
- `--qwen-weight-dtype`
- `--qwen-use-lora` / `--qwen-no-lora`
- `--qwen-lora-name`
- `--qwen-lora-strength`
- `--memory-budget off|relaxed|balanced|conservative`

Do not default to `fp8_e4m3fn_fast`; it has produced unstable colorful mosaic outputs on Spark. Use `weight_dtype=default` unless explicitly experimenting.

## Safety And Failure Modes

- Uploaded/reference files are usually under `/mnt/user-data/uploads/`.
- Validate that the first reference image is the actual image to edit.
- If more than three references are supplied, use the most relevant first three.
- For strict removal of labels, stickers, watermarks, or large occluders, ask for or create a mask/inpaint workflow instead of relying on generic edit prompting.
- Do not remove watermarks or copyright marks from third-party images.
- If generation appears stuck, check `/queue`, `/system_stats`, and ComfyUI logs before retrying.

## Bundled Resources

- `scripts/generate.py`: provider routing, Qwen API workflow construction, Spark memory budget enforcement, and output download.
- `scripts/comfy_resource_status.py`: queue, RAM, and VRAM snapshot.
- `scripts/download_qwen_comfy_models.sh`: ModelScope model downloader.
- `workflows/comfy/model-manifest.json`: model file manifest and target paths.
- `workflows/comfy/ui/qwen-image-2512.json`: Qwen Image UI workflow.
- `workflows/comfy/ui/qwen-image-edit-2509.json`: Qwen Image Edit 2509 UI workflow.
- `workflows/comfy/ui/qwen-image-edit-2511.json`: optional 2511 fallback workflow reference.
- `workflows/comfy/ui/qwen-image-layered.json`: Qwen Image Layered UI workflow.
- `templates/doraemon.md`: read only for Doraemon comic-style requests.
