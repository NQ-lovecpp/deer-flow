# Spark ComfyUI Compose

This directory is the Spark deployment recipe for running ComfyUI as a Docker
Compose service while keeping DeerFlow skills as the scheduler/control plane.

The container owns the runtime. The host owns the large mutable state:

- model weights
- input images
- generated outputs
- ComfyUI user/workflow state
- custom nodes

Do not commit model weights, generated images, or a real `.env` file.

## Why Compose Here

The current host ComfyUI daemon is the known-good baseline. This Compose setup is
intended to become the reproducible service wrapper after parity smoke tests pass.

DeerFlow should continue to call a stable URL:

```bash
COMFYUI_URL=http://127.0.0.1:8188
```

Scheduling remains in the image-generation skill and `generate.py`: queue checks,
memory budget, file locking, and Qwen workflow selection happen before a prompt is
posted to ComfyUI. Compose only manages process lifecycle and runtime config.

## Layout

```text
deploy/spark/comfy/
  compose.yaml
  Dockerfile
  entrypoint.sh
  .env.example
  README.md
```

The default host paths match the existing Spark installation:

```text
/home/chen/comfyui/ComfyUI/models
/home/chen/comfyui/ComfyUI/input
/home/chen/comfyui/ComfyUI/output
/home/chen/comfyui/ComfyUI/user
/home/chen/comfyui/ComfyUI/custom_nodes
```

## First Run

From the repository root:

```bash
cp deploy/spark/comfy/.env.example deploy/spark/comfy/.env
docker compose --env-file deploy/spark/comfy/.env \
  -f deploy/spark/comfy/compose.yaml config
```

The default build uses China-friendly sources: GitCode for the ComfyUI Git mirror
and TUNA for apt/pip. It also uses host networking so clone/install steps follow
the same network path as the Spark host. Switch `COMFYUI_REPO_URL` back to GitHub
only if the mirror lags or misses a target commit.

The Dockerfile keeps the CUDA/PyTorch stack from the NGC base image and filters
`torch`, `torchvision`, and `torchaudio` out of ComfyUI `requirements.txt`. Do not
let pip replace those packages unless you are deliberately changing CUDA stacks.

Build the image:

```bash
docker compose --env-file deploy/spark/comfy/.env \
  -f deploy/spark/comfy/compose.yaml build comfyui
```

Start the service:

```bash
docker compose --env-file deploy/spark/comfy/.env \
  -f deploy/spark/comfy/compose.yaml up -d comfyui
```

Check health:

```bash
curl -fsS http://127.0.0.1:8188/system_stats | python3 -m json.tool
```

Then verify through the skill-side resource probe:

```bash
COMFYUI_URL=http://127.0.0.1:8188 \
  python3 skills/public/image-generation/scripts/comfy_resource_status.py
```

## Side-by-side Trial

The existing host daemon already listens on `8188`. For a non-disruptive trial,
set this in `deploy/spark/comfy/.env`:

```env
COMFYUI_PORT=8189
COMFYUI_URL=http://127.0.0.1:8189
```

Then start Compose and run a smoke test against port `8189`:

```bash
COMFYUI_URL=http://127.0.0.1:8189 \
IMAGE_GENERATION_PROVIDER=comfy_qwen_image \
SPARK_IMAGE_MEMORY_BUDGET=conservative \
python3 skills/public/image-generation/scripts/generate.py \
  --prompt-file /tmp/qwen2512-smoke.txt \
  --output-file /tmp/qwen2512-smoke.png \
  --aspect-ratio 1:1 \
  --qwen-preset fast \
  --memory-budget conservative
```

When parity is confirmed, stop the host daemon and switch `COMFYUI_PORT` back to
`8188`.

## Model Setup

Models are bind-mounted from the host, not baked into the image. To refresh the
Qwen model set from ModelScope, run from the repository root:

```bash
COMFYUI_DIR=/home/chen/comfyui/ComfyUI \
  skills/public/image-generation/scripts/download_qwen_comfy_models.sh
```

The model manifest lives at:

```text
skills/public/image-generation/workflows/comfy/model-manifest.json
```

## Operations

Common commands:

```bash
docker compose --env-file deploy/spark/comfy/.env -f deploy/spark/comfy/compose.yaml ps
docker compose --env-file deploy/spark/comfy/.env -f deploy/spark/comfy/compose.yaml logs -f comfyui
docker compose --env-file deploy/spark/comfy/.env -f deploy/spark/comfy/compose.yaml restart comfyui
docker compose --env-file deploy/spark/comfy/.env -f deploy/spark/comfy/compose.yaml down
```

The image is pinned by the full `COMFYUI_GIT_REF` SHA for parity with the tested
Spark host daemon. Set it to a newer commit or branch only when you are ready to
rebuild and rerun Qwen smoke tests.
