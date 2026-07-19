# Stable Layers — Inference

Decomposes an image into ordered layers (background + separated objects) using
**Qwen-Image-Layered** with a Stable Layers LoRA.

Project Page: https://stability-ai.github.io/stable-layers.github.io/

Paper: https://arxiv.org/abs/2605.30257

Note: whilst the paper used gemini, this model was retrained using an open VLM, we found no performance loss.

---

## Recommended inference settings

> ### **Heun sampler · 50 steps · CFG 1.0 · 640 px · 4 layers**

| Setting | Value | Flag |
|---|---|---|
| **Sampler** | **Heun (2nd order)** | always used — not configurable |
| **Steps** | **50** | `--steps 50` |
| **CFG / guidance** | **1.0 (off)** | `--guidance-scale 1.0` |
| Resolution | 640 px (max dim) | `--size 640` |
| Layers | 4 | `--num-layers 4` |

**Note:** with using a high resolution or lower steps or not heun will garble the results.

---

## Install

```bash
pip install torch diffusers transformers peft pillow numpy
```

Tested with `torch 2.11`, `diffusers 0.37`, `transformers 5.5`, `peft 0.18`.
Requires **one GPU** — the base model is ~40 GB in bf16, so an 80 GB-class card
(A100-80 / H100 / H200) is comfortable.

---

## Weights

The base model is pulled from HuggingFace automatically
(`Qwen/Qwen-Image-Layered`). LoRa adapter found here: https://huggingface.co/StabilityLabs/Stable-Layers

---

## Usage

```bash
# single image
python decompose.py --input photo.png --output results/

# a directory of images
python decompose.py --input images/ --output results/

# RGBA layers with real alpha (for compositing / editors)
python decompose.py --input images/ --output results/ --transparent

# explicit LoRA location
python decompose.py --input photo.png --output results/ --lora ./checkpoint-600
```

---

## Output

```
results/<image_name>/
  source.png       # input, resized
  composite.png    # layers recomposited — compare against source as a sanity check
  layer_0.png      # background (inpainted behind the removed objects)
  layer_1.png      # object layers, back-to-front
  layer_2.png
  layer_3.png
```

Layers are ordered back-to-front: `layer_0` is the background, higher indices sit
on top. Not every image needs all 4 — unused layers come out blank, which is
normal.

By default layers are composited onto **white** (easy to eyeball). Pass
`--transparent` to get **RGBA with real alpha**, which is what you want when
importing into an editor or compositing them yourself.

---

## Notes

- **Reproducible:** noise is seeded per image as `seed + image_index`
  (`--seed 42` by default), so the same inputs give the same outputs.
- **Prompt:** decomposition is driven by the source image; the text prompt only
  nudges guidance. The default (`"a clean, well composed image"`) is fine —
  override with `--prompt` if you want.
- **Aspect ratio** is preserved; the longest side is scaled to `--size` and both
  dimensions are rounded to multiples of 16.
