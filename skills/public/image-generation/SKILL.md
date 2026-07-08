---
name: image-generation
description: Use this skill when the user requests to generate, create, imagine, or visualize images including characters, scenes, products, or any visual content. Supports plain-text prompts, reference images, and explicit CLI controls for local ComfyUI/Qwen generation.
---

# Image Generation Skill

## Overview

This skill generates images using a prompt file and a Python script. Prefer plain-text prompt files (`.txt`) for local ComfyUI/Qwen workflows. JSON is only a compatibility format for older/provider-specific prompts; it does not control ComfyUI sampling quality.

## Core Capabilities

- Create concise plain-text prompts for AIGC image generation
- Support multiple reference images for style/composition guidance
- Edit an uploaded/reference image when `IMAGE_GENERATION_PROVIDER=comfy_qwen_edit` is configured
- Generate images through automated Python script execution
- Route local text-to-image and image-editing differently so Qwen Edit is not misused as a blank-canvas generator
- Handle various image generation scenarios (character design, scenes, products, image editing, product retouching, etc.)

## Workflow

### Step 1: Understand Requirements

When a user requests image generation, identify:

- Subject/content: What should be in the image
- Style preferences: Art style, mood, color palette
- Technical specs: Aspect ratio, composition, lighting
- Reference images: Any images to guide generation or edit. Uploaded images are usually under `/mnt/user-data/uploads/`.
- For image editing requests, pass the target image as the first `--reference-images` argument. Do not claim image editing is unavailable when the local provider is `comfy_qwen_edit`.
- You don't need to check the folder under `/mnt/user-data` unless you need to find an uploaded image path.

### Step 2: Create Prompt File

Generate a plain-text prompt file in `/mnt/user-data/workspace/` with naming pattern: `{descriptive-name}.txt`. Put the visual instruction directly in prose.

Use JSON only when a specific non-local provider requires it. Do not put sampling controls such as quality, steps, CFG, denoise, LoRA, seed, sampler, or scheduler inside JSON; local ComfyUI/Qwen ignores those as real controls. Use CLI arguments instead.

### Step 3: Execute Generation

Call the Python script:
```bash
python /mnt/skills/public/image-generation/scripts/generate.py \
  --prompt-file /mnt/user-data/workspace/prompt-file.txt \
  --reference-images /path/to/ref1.jpg /path/to/ref2.png \
  --output-file /mnt/user-data/outputs/generated-image.jpg
  --aspect-ratio 16:9
```

Parameters:

- `--prompt-file`: Absolute path to a prompt file (plain `.txt` preferred; JSON only for compatibility)
- `--reference-images`: Absolute paths to reference images (optional, space-separated)
- `--output-file`: Absolute path to output image file (required)
- `--aspect-ratio`: Aspect ratio of the generated image (optional, default: 16:9)

[!NOTE]
Do NOT read the python file, just call it with the parameters. For local ComfyUI/Qwen, CLI flags are the source of truth for quality and sampling behavior; JSON fields like `technical.quality` are not real controls.


## Local ComfyUI / Qwen Image Edit Provider

Use `IMAGE_GENERATION_PROVIDER=comfy` for the normal local route. The script automatically selects the right local workflow: with `--reference-images` it uses Qwen Image Edit; without reference images it uses the text-to-image fallback. Do not use Qwen Image Edit for pure text-to-image by inventing a blank canvas; that produces noisy, unusable results. Use `IMAGE_GENERATION_PROVIDER=comfy_qwen_edit` only when you have 1-3 real reference/input images to edit or guide.

Use plain-text prompt files and explicit `--qwen-*` CLI parameters for Qwen image editing; do not encode Qwen quality controls in JSON.

Use it for requests such as:

- Change text, dates, colors, background, clothing, or style in an uploaded image
- Preserve the original image style while changing specified details
- Combine or reference multiple uploaded images, such as person + product, person + scene, or up to three visual references
- Product retouching, poster edits, meme/text edits, portrait style transfer, and other user-owned image edits

Do not claim image input is unavailable. Uploaded files under `/mnt/user-data/uploads/` should be passed as `--reference-images`. If there is no uploaded/reference image, do not force Qwen Image Edit; call the script with `IMAGE_GENERATION_PROVIDER=comfy` and no `--reference-images` so it routes to the text-to-image fallback.

Example pure text-to-image command with the local fallback:

```bash
cat > /mnt/user-data/workspace/logo-prompt.txt <<EOF
A clean vector-style logo of 1I/ʻOumuamua, elongated interstellar object silhouette, elegant orbital arc, high contrast, minimal shapes, professional astronomy brand mark, no text.
EOF

IMAGE_GENERATION_PROVIDER=comfy python /mnt/skills/public/image-generation/scripts/generate.py   --prompt-file /mnt/user-data/workspace/logo-prompt.txt   --output-file /mnt/user-data/outputs/oumuamua-logo.jpg   --aspect-ratio 1:1
```

Example balanced-quality image editing command:

```bash
cat > /mnt/user-data/workspace/edit-prompt.txt <<'EOF'
Replace the dates "19", "20", "21" with "3", "4", "5" respectively. Keep the same style, font, layout, and background.
EOF

IMAGE_GENERATION_PROVIDER=comfy_qwen_edit python /mnt/skills/public/image-generation/scripts/generate.py \
  --prompt-file /mnt/user-data/workspace/edit-prompt.txt \
  --reference-images /mnt/user-data/uploads/input.jpeg \
  --output-file /mnt/user-data/outputs/edited-image.jpg \
  --aspect-ratio 16:9 \
  --qwen-preset balanced
```

Multi-image reference example:

```bash
IMAGE_GENERATION_PROVIDER=comfy_qwen_edit python /mnt/skills/public/image-generation/scripts/generate.py \
  --prompt-file /mnt/user-data/workspace/product-poster.txt \
  --reference-images /mnt/user-data/uploads/person.jpg /mnt/user-data/uploads/product.png /mnt/user-data/uploads/scene.jpg \
  --output-file /mnt/user-data/outputs/product-poster.jpg \
  --qwen-preset balanced
```

Qwen preset guidance:

- `--qwen-preset balanced` is the default and recommended first try. It disables Lightning LoRA and uses about 20 steps / CFG 2.5 / denoise 1.0 / 0.5MP / `weight_dtype=default`.
- `--qwen-preset quality` is slower and uses about 40 steps / CFG 4.0 / denoise 1.0 / 1.0MP / `weight_dtype=default`.
- `--qwen-preset fast --qwen-use-lora` is for quick previews only. It uses the local 4-step Lightning LoRA and may look rough.

Useful Qwen controls:

- `--qwen-steps`: sampling steps
- `--qwen-cfg`: CFG / true CFG strength
- `--qwen-denoise`: how much to change the source image; lower preserves more, higher changes more
- `--qwen-seed`: fixed seed for reproducibility
- `--qwen-sampler` and `--qwen-scheduler`: ComfyUI KSampler controls
- `--qwen-shift`: ModelSamplingAuraFlow shift
- `--qwen-megapixels`: pre-scale before Qwen conditioning/VAE; defaults come from the selected preset
- `--qwen-weight-dtype`: UNETLoader weight dtype; default is `default`. Do not use `fp8_e4m3fn_fast` unless the user explicitly asks for an experimental preview.
- `--qwen-use-lora` / `--qwen-no-lora`: force Lightning LoRA on or off
- `--qwen-lora-name` and `--qwen-lora-strength`: override LoRA file and strength

Important notes for this local provider:

- `IMAGE_GENERATION_PROVIDER=comfy` is the default local route. With reference images it uses Qwen Image Edit; without reference images it uses the text-to-image fallback.
- Qwen Image Edit requires real reference/input images. Do not use blank, generated, or dummy images just to satisfy the API.
- Qwen Image Edit 2509 works best with 1-3 input images. If more are supplied, use the most relevant first three.
- The first reference image is the main image to edit and is encoded into the edit latent. The second and third images are additional visual references.
- The local workflow passes the same images and VAE to both positive and negative Qwen conditioning. This avoids the broken colorful-noise behavior seen with incomplete conditioning.
- Do not default to `fp8_e4m3fn_fast`; on this machine it has produced colorful mosaic outputs. Use the default weight dtype unless explicitly experimenting.
- Removing labels, stickers, or large occluding objects is high risk because Qwen Image Edit may redraw brand text and object structure. For strict local preservation, use or request a mask/inpaint workflow instead.
- Qwen Image Edit needs enough free unified GPU memory. If generation appears stuck, check ComfyUI queue/logs and available VRAM.
- Do not use this skill to remove watermarks or copyright marks from third-party images. For ordinary user-owned edits, proceed normally.

## Character Generation Example

User request: "Create a Tokyo street style woman character in 1990s"

Create prompt file: `/mnt/user-data/workspace/asian-woman.json`
```json
{
  "characters": [{
    "gender": "female",
    "age": "mid-20s",
    "ethnicity": "Japanese",
    "body_type": "slender, elegant",
    "facial_features": "delicate features, expressive eyes, subtle makeup with emphasis on lips, long dark hair partially wet from rain",
    "clothing": "stylish trench coat, designer handbag, high heels, contemporary Tokyo street fashion",
    "accessories": "minimal jewelry, statement earrings, leather handbag",
    "era": "1990s"
  }],
  "negative_prompt": "blurry face, deformed, low quality, overly sharp digital look, oversaturated colors, artificial lighting, studio setting, posed, selfie angle",
  "style": "Leica M11 street photography aesthetic, film-like rendering, natural color palette with slight warmth, bokeh background blur, analog photography feel",
  "composition": "medium shot, rule of thirds, subject slightly off-center, environmental context of Tokyo street visible, shallow depth of field isolating subject",
  "lighting": "neon lights from signs and storefronts, wet pavement reflections, soft ambient city glow, natural street lighting, rim lighting from background neons",
  "color_palette": "muted naturalistic tones, warm skin tones, cool blue and magenta neon accents, desaturated compared to digital photography, film grain texture"
}
```

Execute generation:
```bash
python /mnt/skills/public/image-generation/scripts/generate.py \
  --prompt-file /mnt/user-data/workspace/asian-woman.json \
  --output-file /mnt/user-data/outputs/asian-woman-01.jpg \
  --aspect-ratio 2:3
```

With reference images:
```json
{
  "characters": [{
    "gender": "based on [Image 1]",
    "age": "based on [Image 1]",
    "ethnicity": "human from [Image 1] adapted to Star Wars universe",
    "body_type": "based on [Image 1]",
    "facial_features": "matching [Image 1] with slight weathered look from space travel",
    "clothing": "Star Wars style outfit - worn leather jacket with utility vest, cargo pants with tactical pouches, scuffed boots, belt with holster",
    "accessories": "blaster pistol on hip, comlink device on wrist, goggles pushed up on forehead, satchel with supplies, personal vehicle based on [Image 2]",
    "era": "Star Wars universe, post-Empire era"
  }],
  "prompt": "Character inspired by [Image 1] standing next to a vehicle inspired by [Image 2] on a bustling alien planet street in Star Wars universe aesthetic. Character wearing worn leather jacket with utility vest, cargo pants with tactical pouches, scuffed boots, belt with blaster holster. The vehicle adapted to Star Wars aesthetic with weathered metal panels, repulsor engines, desert dust covering, parked on the street. Exotic alien marketplace street with multi-level architecture, weathered metal structures, hanging market stalls with colorful awnings, alien species walking by as background characters. Twin suns casting warm golden light, atmospheric dust particles in air, moisture vaporators visible in distance. Gritty lived-in Star Wars aesthetic, practical effects look, film grain texture, cinematic composition.",
  "negative_prompt": "clean futuristic look, sterile environment, overly CGI appearance, fantasy medieval elements, Earth architecture, modern city",
  "style": "Star Wars original trilogy aesthetic, lived-in universe, practical effects inspired, cinematic film look, slightly desaturated with warm tones",
  "composition": "medium wide shot, character in foreground with alien street extending into background, environmental storytelling, rule of thirds",
  "lighting": "warm golden hour lighting from twin suns, rim lighting on character, atmospheric haze, practical light sources from market stalls",
  "color_palette": "warm sandy tones, ochre and sienna, dusty blues, weathered metals, muted earth colors with pops of alien market colors",
  "technical": {
    "aspect_ratio": "9:16",
    "quality": "high",
    "detail_level": "highly detailed with film-like texture"
  }
}
```
```bash
python /mnt/skills/public/image-generation/scripts/generate.py \
  --prompt-file /mnt/user-data/workspace/star-wars-scene.json \
  --reference-images /mnt/user-data/uploads/character-ref.jpg /mnt/user-data/uploads/vehicle-ref.jpg \
  --output-file /mnt/user-data/outputs/star-wars-scene-01.jpg \
  --aspect-ratio 16:9
```

## Common Scenarios

Use different JSON schemas for different scenarios.

**Character Design**:
- Physical attributes (gender, age, ethnicity, body type)
- Facial features and expressions
- Clothing and accessories
- Historical era or setting
- Pose and context

**Scene Generation**:
- Environment description
- Time of day, weather
- Mood and atmosphere
- Focal points and composition

**Product Visualization**:
- Product details and materials
- Lighting setup
- Background and context
- Presentation angle

## Specific Templates

Read the following template file only when matching the user request.

- [Doraemon Comic](templates/doraemon.md)

## Output Handling

After generation:

- Images are typically saved in `/mnt/user-data/outputs/`
- Share generated images with user using present_files tool
- Provide brief description of the generation result
- Offer to iterate if adjustments needed

## Tips: Enhancing Generation with Reference Images

For scenarios where visual accuracy is critical, **use the `image_search` tool first** to find reference images before generation.

**Recommended scenarios for using image_search tool:**
- **Character/Portrait Generation**: Search for similar poses, expressions, or styles to guide facial features and body proportions
- **Specific Objects or Products**: Find reference images of real objects to ensure accurate representation
- **Architectural or Environmental Scenes**: Search for location references to capture authentic details
- **Fashion and Clothing**: Find style references to ensure accurate garment details and styling

**Example workflow:**
1. Call the `image_search` tool to find suitable reference images:
   ```
   image_search(query="Japanese woman street photography 1990s", size="Large")
   ```
2. Download the returned image URLs to local files
3. Use the downloaded images as `--reference-images` parameter in the generation script

This approach significantly improves generation quality by providing the model with concrete visual guidance rather than relying solely on text descriptions.

## Prompt Format Guidance

Prefer `.txt` prompts for all local ComfyUI/Qwen work. JSON prompt files are still accepted for backwards compatibility and remote provider workflows, but local sampling controls must be CLI flags:

```bash
# Good: prompt meaning in text, generation controls in CLI
IMAGE_GENERATION_PROVIDER=comfy_qwen_edit python /mnt/skills/public/image-generation/scripts/generate.py \
  --prompt-file /mnt/user-data/workspace/edit.txt \
  --reference-images /mnt/user-data/uploads/input.png \
  --output-file /mnt/user-data/outputs/edit.png \
  --qwen-preset quality \
  --qwen-steps 40 \
  --qwen-cfg 4.0 \
  --qwen-denoise 0.9
```

For no-image local generation, still use Qwen and simply omit `--reference-images`:

```bash
IMAGE_GENERATION_PROVIDER=comfy_qwen_edit python /mnt/skills/public/image-generation/scripts/generate.py \
  --prompt-file /mnt/user-data/workspace/logo.txt \
  --output-file /mnt/user-data/outputs/logo.png \
  --aspect-ratio 1:1 \
  --qwen-preset quality \
  --qwen-steps 40 \
  --qwen-cfg 4.0
```

Avoid relying on JSON fields such as `technical.quality`, `steps`, `cfg`, or `denoise`; they are just text unless passed through CLI flags.

## Providers (Gemini / MiniMax / Local ComfyUI)

This skill auto-selects the provider by environment variables (no CLI change):

- `IMAGE_GENERATION_PROVIDER=comfy_qwen_edit` → use local ComfyUI Qwen Image Edit, with optional reference images.
- `IMAGE_GENERATION_PROVIDER=comfy` → compatibility alias for local Qwen Image Edit, not SD1.5.
- `IMAGE_GENERATION_PROVIDER=comfy_sd15` → explicit old SD1.5 fallback only when the user asks for it.
- `GEMINI_API_KEY` set → use Gemini when no explicit provider is set.
- Only `MINIMAX_API_KEY` set → use MiniMax (`/v1/image_generation`, model `image-01`).
- Force one explicitly with `IMAGE_GENERATION_PROVIDER=gemini|minimax|comfy_qwen_edit|comfy_sd15`. Prefer `comfy_qwen_edit` for local work.

MiniMax optional overrides: `MINIMAX_API_HOST` (default `https://api.minimaxi.com`),
`MINIMAX_IMAGE_MODEL` (default `image-01`). Reference images are sent as the MiniMax
`subject_reference` character image. The CLI and `--prompt-file` / `--reference-images`
/ `--output-file` / `--aspect-ratio` arguments are identical for both providers.

**MiniMax prompt handling (provider-internal).** Authoring is provider-agnostic — write
the same structured JSON regardless of which provider is active. MiniMax `image-01`
consumes a single text string, so the MiniMax path itself sends only the JSON `prompt`
field (the other fields such as `style` / `composition` / `negative_prompt` apply to the
Gemini path) and enables `prompt_optimizer` so MiniMax expands it server-side. MiniMax
caps that prompt at 1500 characters; if the `prompt` field is longer, the script returns
an error instead of calling the API. The Gemini path receives the full structured JSON.

## Notes

- Always use English for prompts regardless of user's language
- Prefer plain text prompts for local ComfyUI/Qwen; JSON is optional compatibility, not a quality-control interface
- Reference images enhance generation quality significantly
- Iterative refinement is normal for optimal results
- For character generation, include the detailed character object plus a consolidated prompt field
