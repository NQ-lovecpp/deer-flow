---
name: image-generation
description: Generate new images, edit or compose reference images, and create RGBA layer decompositions with local Qwen Image models on DGX Spark. Use for text-to-image, image modification, product or portrait visuals, multi-reference composition, foreground/background separation, editable layers, and ComfyUI image workflows. Always inspect the Spark resource policy before local work.
---

# Image Generation

## Route The Task

Use local ComfyUI unless the user explicitly requests a remote provider.

| Request | Task | Model |
|---|---|---|
| Generate without reference images | `generate` | Qwen Image 2512 FP8 |
| Modify or compose 1-3 reference images | `edit` | Qwen Image Edit 2509 FP8 |
| Split foreground/background, produce RGBA layers, or request editable layers | `layered` | Qwen Image Layered FP8 mixed |

Use `--task auto` for normal work. It selects `edit` when reference images exist and `generate` otherwise. Use `--task layered` only for an explicit layer request. Never use Edit with a blank dummy canvas for text-to-image.

Treat Qwen Image Edit 2511 as an explicit compatibility fallback only. Do not select it automatically.

## Respect The Spark Contract

Read [references/spark-runtime-policy.json](references/spark-runtime-policy.json) when exact machine or model data is needed.

- Machine: NVIDIA DGX Spark, GB10 Grace Blackwell, 128GiB coherent unified memory.
- Total AI memory budget: 90GiB.
- Preserve at least 32GiB system-available memory.
- Reserve 34GiB for `vllm_qwen36` and keep its policy `pinned`.
- Never stop, restart, remove, or reconfigure vLLM for image work.
- Keep at most one Comfy diffusion model resident. Reuse it for consecutive same-model jobs; release Comfy models before switching.
- Treat Layered as low-priority and exclusive within Comfy. Release it after every job. Reject it when the budget is unavailable.
- Keep Layered automatic jobs at 640px, batch 1, and at most 4 requested layers.

The scripts enforce these rules. Do not bypass a rejected plan by changing environment variables or using `--memory-budget off`.

## Inspect Before Running

Run the scheduler before every local image task:

```bash
python /mnt/skills/public/image-generation/scripts/spark_image_scheduler.py \
  status --json
```

Plan a specific task when the route or budget is uncertain:

```bash
python /mnt/skills/public/image-generation/scripts/spark_image_scheduler.py \
  plan --task generate --json
```

Proceed only when `admitted` is `true`. Report `rejection_reason` instead of retrying blindly. A missing vLLM metrics connection is diagnostic only; the scheduler still reserves its full 34GiB and never manipulates it.

## Generate

```bash
IMAGE_GENERATION_PROVIDER=comfy \
python /mnt/skills/public/image-generation/scripts/generate.py \
  --task generate \
  --prompt-file /mnt/user-data/workspace/prompt.txt \
  --output-file /mnt/user-data/outputs/image.png \
  --aspect-ratio 4:3 \
  --qwen-preset quality
```

Default model: `qwen_image_2512_fp8_e4m3fn.safetensors`. Use `balanced` for routine iteration and `quality` for a final foreground job.

## Edit Or Compose

Pass the primary image first and at most two secondary references.

```bash
IMAGE_GENERATION_PROVIDER=comfy \
python /mnt/skills/public/image-generation/scripts/generate.py \
  --task edit \
  --prompt-file /mnt/user-data/workspace/edit.txt \
  --reference-images /mnt/user-data/uploads/input.png \
  --output-file /mnt/user-data/outputs/edited.png \
  --qwen-preset balanced
```

Default model: `qwen_image_edit_2509_fp8_e4m3fn.safetensors`. Use the 2509 Lightning LoRA only for an explicitly requested fast preview.

## Create Layers

For decomposition, pass one source image. Without a reference image, Layered generates a layered image from text.

```bash
IMAGE_GENERATION_PROVIDER=comfy \
python /mnt/skills/public/image-generation/scripts/generate.py \
  --task layered \
  --prompt-file /mnt/user-data/workspace/layers.txt \
  --reference-images /mnt/user-data/uploads/input.png \
  --output-file /mnt/user-data/outputs/layered.png \
  --qwen-layers 3 \
  --qwen-resolution 640 \
  --qwen-preset balanced
```

The requested output is the reconstructed composite. Additional files use `.layer-01.png`, `.layer-02.png`, and so on. The `.layers.json` sidecar records the model, prompt id, mode, requested layer count, and every output path.

Layer prompts describe the complete scene, including partially hidden content. They do not assign exact semantics to layer numbers.

## Useful Controls

- `--task auto|generate|edit|layered`
- `--dry-run`
- `--qwen-preset fast|balanced|quality`
- `--qwen-steps`, `--qwen-cfg`, `--qwen-seed`
- `--qwen-layers`, `--qwen-resolution`
- `--qwen-use-lora`, `--qwen-no-lora`
- `--memory-budget balanced|conservative|relaxed|off` controls quality conservation only; it never disables the 90GiB scheduler.

Do not default to `fp8_e4m3fn_fast`; it has produced unstable mosaic output on Spark. Use `weight_dtype=default`.

## Failure Rules

- Do not queue parallel Comfy jobs.
- If the queue is non-empty, wait and inspect it.
- If memory is below policy, reduce Image/Edit resolution or reject Layered. Never reclaim vLLM.
- For masks, strict object removal, or alpha repair, use a dedicated Comfy inpaint/composite workflow.
- Do not remove third-party copyright marks or watermarks.

## Bundled Tools

- `scripts/spark_image_scheduler.py`: machine status, task plan, and Comfy-only release.
- `scripts/generate.py`: provider routing, enforced scheduling, Qwen workflows, and output download.
- `scripts/comfy_resource_status.py`: compatibility wrapper for full scheduler status.
- `workflows/comfy/model-manifest.json`: model files and storage paths.
- `workflows/comfy/ui/`: editable ComfyUI workflows for Image, Edit, and Layered.
