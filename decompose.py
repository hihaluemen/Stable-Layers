#!/usr/bin/env python3
"""
Layer decomposition inference — Qwen-Image-Layered + GRPO LoRA.

Decomposes a source image into ordered RGBA layers (background + objects).

============================================================================
RECOMMENDED INFERENCE SETTINGS  ***  USE THESE  ***
============================================================================
    Sampler : Heun (2nd order)   <- the only sampler here; always used
    Steps   : 50                 <- --steps 50   (default)
    CFG     : 1.0 (disabled)     <- --guidance-scale 1.0 (default)
    Size    : 640 px             <- --size 640   (default)
    Layers  : 4                  <- --num-layers 4 (default)

All published results were produced with exactly these settings, and they are
the defaults below. Lowering the step count or raising CFG noticeably degrades
the decomposition. Note 50 Heun steps == ~100 model evaluations (Heun is
2nd order: two forward passes per step) — that is expected, not a bug.
============================================================================

Usage:
    python decompose.py --input photo.png --output out/
    python decompose.py --input images/ --output out/ --transparent

Output per image:
    out/<name>/source.png      resized input
    out/<name>/composite.png   layers recomposited (sanity check vs source)
    out/<name>/layer_0.png     background (inpainted behind the objects)
    out/<name>/layer_1..N.png  object layers, back-to-front
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

# Qwen-Image prompt template (must match training/inference used for the LoRA)
PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_DROP_IDX = 34
DEFAULT_PROMPT = "a clean, well composed image"
VAE_SCALE_FACTOR = 8


# ---------------------------------------------------------------------------
# Latent helpers
# ---------------------------------------------------------------------------

def rgb_to_rgba(image):
    b, _, h, w = image.shape
    alpha = torch.ones(b, 1, h, w, device=image.device, dtype=image.dtype)
    return torch.cat([image, alpha], dim=1)


def normalize_latents(latents, vae):
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    return (latents - mean) / std


def denormalize_latents(latents, vae):
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    inv_std = (1.0 / torch.tensor(vae.config.latents_std)).view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    return latents / inv_std + mean


def pack_latents(latents, batch_size, num_channels, height, width, num_frames):
    latents = latents.view(batch_size, num_frames, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4, 6)
    return latents.reshape(batch_size, num_frames * (height // 2) * (width // 2), num_channels * 4)


def unpack_latents(latents, height, width, num_layers, vae_scale_factor=VAE_SCALE_FACTOR):
    batch_size, _, channels = latents.shape
    frames = num_layers + 1
    h = 2 * (int(height) // (vae_scale_factor * 2))
    w = 2 * (int(width) // (vae_scale_factor * 2))
    latents = latents.view(batch_size, frames, h // 2, w // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 1, 4, 2, 5, 3, 6)
    latents = latents.reshape(batch_size, frames, channels // 4, h, w)
    return latents.permute(0, 2, 1, 3, 4)


@torch.no_grad()
def encode_condition_image(vae, image):
    """RGB image tensor in [-1,1], (B,3,H,W) -> packed condition latents."""
    b = image.shape[0]
    vae_input_ch = getattr(vae.config, "input_channels", 3)
    if vae_input_ch == 4 and image.shape[1] == 3:
        image = rgb_to_rgba(image)
    elif vae_input_ch == 3 and image.shape[1] == 4:
        image = image[:, :3]

    image_5d = image.unsqueeze(2).to(dtype=vae.dtype)
    dist = vae.encode(image_5d)
    if hasattr(dist, "latent_dist"):
        latents = dist.latent_dist.mode()
    elif hasattr(dist, "mode"):
        latents = dist.mode()
    else:
        latents = dist

    latents = normalize_latents(latents, vae)
    z_dim, lh, lw = latents.shape[1], latents.shape[3], latents.shape[4]
    latents = latents.permute(0, 2, 1, 3, 4)
    return pack_latents(latents, b, z_dim, lh, lw, 1).to(dtype=torch.bfloat16)


@torch.no_grad()
def decode_layers(vae, latents, height, width, num_layers):
    b = latents.shape[0]
    unpacked = unpack_latents(latents, height, width, num_layers)
    unpacked = denormalize_latents(unpacked, vae)
    layer_latents = unpacked[:, :, 1:].permute(0, 2, 1, 3, 4)  # drop base frame
    _, _, c, h, w = layer_latents.shape
    decoded = vae.decode(layer_latents.reshape(b * num_layers, c, 1, h, w).to(dtype=vae.dtype),
                         return_dict=False)[0]
    decoded = decoded.squeeze(2).float().clamp(-1.0, 1.0)
    _, c_out, h_out, w_out = decoded.shape
    return decoded.reshape(b, num_layers, c_out, h_out, w_out)


def composite_layers(layers):
    """Porter-Duff 'over', back-to-front."""
    if layers.shape[2] == 3:
        return layers.mean(dim=1)
    result = torch.zeros_like(layers[:, 0, :3])
    for i in range(layers.shape[1]):
        rgb = layers[:, i, :3]
        alpha = (layers[:, i, 3:4] + 1.0) / 2.0
        result = rgb * alpha + result * (1.0 - alpha)
    return result


def tensor_to_pil(t, mode="RGB"):
    arr = ((t.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, mode=mode)


def layer_to_pil_white_bg(layer):
    if layer.shape[0] == 4:
        rgb, alpha = layer[:3], (layer[3:4] + 1.0) / 2.0
        return tensor_to_pil(rgb * alpha + torch.ones_like(rgb) * (1.0 - alpha))
    return tensor_to_pil(layer)


def layer_to_pil_rgba(layer):
    layer = layer.clamp(-1, 1).float()
    rgba = torch.cat([(layer[:3] + 1) / 2, (layer[3:4] + 1) / 2], dim=0)
    arr = (rgba * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, mode="RGBA")


# ---------------------------------------------------------------------------
# Denoising — Heun (2nd order)
# ---------------------------------------------------------------------------

def transformer_forward(transformer, hidden_states, timestep, encoder_hidden_states,
                        img_shapes, encoder_hidden_states_mask=None, additional_t_cond=None):
    return transformer(
        hidden_states=hidden_states,
        timestep=timestep / 1000,
        encoder_hidden_states=encoder_hidden_states,
        encoder_hidden_states_mask=encoder_hidden_states_mask,
        img_shapes=img_shapes,
        guidance=None,
        additional_t_cond=additional_t_cond,
        return_dict=False,
    )[0]


def build_img_shapes(num_layers, packed_h, packed_w):
    # (num_layers + 1) generated frames + 1 condition-image entry
    return [[*[(1, packed_h, packed_w) for _ in range(num_layers + 1)], (1, packed_h, packed_w)]]


@torch.no_grad()
def denoise(transformer, scheduler, latents, timesteps, prompt_embeds, prompt_mask,
            img_shapes, condition_latents, guidance_scale,
            neg_embeds=None, neg_mask=None, additional_t_cond=None):
    """Heun's method (2nd order) over the scheduler's sigmas."""
    gen_seq_len = latents.shape[1]
    do_cfg = guidance_scale > 1.0 and neg_embeds is not None

    def velocity(lat, t_val):
        ts = t_val.expand(1).to(lat.dtype)
        model_input = lat if condition_latents is None else torch.cat([lat, condition_latents], dim=1)
        v = transformer_forward(transformer, model_input, ts, prompt_embeds,
                                img_shapes, prompt_mask, additional_t_cond)[:, :gen_seq_len]
        if do_cfg:
            v_neg = transformer_forward(transformer, model_input, ts, neg_embeds,
                                        img_shapes, neg_mask, additional_t_cond)[:, :gen_seq_len]
            v = v_neg + guidance_scale * (v - v_neg)
        return v

    sigmas = scheduler.sigmas
    for i, t in enumerate(timesteps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1] if i + 1 < len(sigmas) else torch.tensor(0.0, device=t.device)
        dt = sigma_next - sigma

        v1 = velocity(latents, t)
        latents_mid = latents + dt * v1
        if sigma_next > 0:                       # Heun corrector
            v2 = velocity(latents_mid, sigma_next * 1000)
            latents = latents + dt * 0.5 * (v1 + v2)
        else:
            latents = latents_mid
    return latents


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def compute_aspect_resize(orig_w, orig_h, max_size):
    """Keep aspect ratio, max dim = max_size, both dims divisible by 16."""
    scale = max_size / max(orig_w, orig_h)
    return (max(int(round(orig_w * scale / 16)) * 16, 16),
            max(int(round(orig_h * scale / 16)) * 16, 16))


def collect_images(input_path):
    p = Path(input_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    raise FileNotFoundError(f"input not found: {input_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Layer decomposition (Heun / 50 steps / cfg 1.0 recommended).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input", required=True, help="image file or directory of images")
    ap.add_argument("--output", required=True, help="output directory")
    ap.add_argument("--lora", default=str(Path(__file__).parent / "model"),
                    help="path to the LoRA adapter directory")
    ap.add_argument("--base-model", default="Qwen/Qwen-Image-Layered",
                    help="base model (HuggingFace id or local path)")
    # --- recommended settings (defaults) ---
    ap.add_argument("--steps", type=int, default=50, help="Heun steps (RECOMMENDED: 50)")
    ap.add_argument("--guidance-scale", type=float, default=1.0, help="CFG (RECOMMENDED: 1.0 = off)")
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--size", type=int, default=640, help="max image dimension")
    # --- misc ---
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--transparent", action="store_true",
                    help="save layers as RGBA with real alpha (for compositing) "
                         "instead of composited onto white")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", default=None, help="HuggingFace cache directory")
    args = ap.parse_args()

    if args.steps != 50 or args.guidance_scale != 1.0:
        print(f"WARNING: using steps={args.steps}, cfg={args.guidance_scale}. "
              f"Recommended is steps=50, cfg=1.0 — results may be worse.\n")

    device = torch.device(args.device)
    images = collect_images(args.input)
    if not images:
        raise SystemExit(f"no images found in {args.input}")
    os.makedirs(args.output, exist_ok=True)
    print(f"{len(images)} image(s) | Heun, {args.steps} steps, cfg {args.guidance_scale}, "
          f"{args.size}px, {args.num_layers} layers")

    # --- load base model -----------------------------------------------
    print(f"Loading base model: {args.base_model}")
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        trust_remote_code=True, cache_dir=args.cache_dir,
    )
    transformer = pipe.transformer.to(device).eval()
    vae = pipe.vae.to(device).eval()
    scheduler = pipe.scheduler
    tokenizer = getattr(pipe, "tokenizer", None)
    text_encoder = getattr(pipe, "text_encoder", None)
    del pipe
    transformer.requires_grad_(False)
    vae.requires_grad_(False)

    # --- apply LoRA ------------------------------------------------------
    print(f"Loading LoRA: {args.lora}")
    from peft import PeftModel
    transformer = PeftModel.from_pretrained(transformer, args.lora)
    transformer.eval()

    in_channels = getattr(transformer.config, "in_channels", 64)
    additional_t_cond = torch.zeros(1, dtype=torch.long, device=device)

    # --- prompt encoding -------------------------------------------------
    if text_encoder is None or tokenizer is None:
        raise SystemExit("base model has no text encoder/tokenizer")
    text_encoder = text_encoder.to(device).eval()
    text_encoder.requires_grad_(False)

    @torch.no_grad()
    def encode_prompt(text):
        tokens = tokenizer([PROMPT_TEMPLATE.format(text)], padding=True,
                           return_tensors="pt").to(text_encoder.device)
        out = text_encoder(input_ids=tokens.input_ids,
                           attention_mask=tokens.attention_mask,
                           output_hidden_states=True)
        hidden = out.hidden_states[-1]
        mask_bool = tokens.attention_mask.bool()
        lengths = mask_bool.sum(dim=1)
        chunks = torch.split(hidden[mask_bool], lengths.tolist(), dim=0)
        chunks = [c[PROMPT_TEMPLATE_DROP_IDX:] for c in chunks]   # drop system prefix
        max_len = max(c.size(0) for c in chunks)
        embeds = torch.stack([torch.cat([c, c.new_zeros(max_len - c.size(0), c.size(1))]) for c in chunks])
        mask = torch.stack([
            torch.cat([torch.ones(c.size(0), dtype=torch.long, device=c.device),
                       torch.zeros(max_len - c.size(0), dtype=torch.long, device=c.device)])
            for c in chunks
        ])
        return embeds.to(dtype=torch.bfloat16, device=device), mask.to(device)

    prompt_embeds, prompt_mask = encode_prompt(args.prompt)
    neg_embeds, neg_mask = encode_prompt("")

    # scheduler may or may not accept sigmas/mu
    import inspect
    supports_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters)

    # --- per-image loop --------------------------------------------------
    for idx, img_path in enumerate(images):
        name = img_path.stem
        out_dir = Path(args.output) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{idx + 1}/{len(images)}] {img_path.name}", flush=True)

        image = Image.open(img_path).convert("RGB")
        tw, th = compute_aspect_resize(*image.size, args.size)
        source = image.resize((tw, th), Image.LANCZOS)

        packed_h = (th // VAE_SCALE_FACTOR) // 2
        packed_w = (tw // VAE_SCALE_FACTOR) // 2

        img_t = torch.from_numpy(np.array(source)).permute(2, 0, 1).float() / 127.5 - 1.0
        img_t = img_t.unsqueeze(0).to(device)
        condition_latents = encode_condition_image(vae, img_t)

        # dynamic-shift mu from the condition sequence length
        mu = (condition_latents.shape[1] / (256 * 256 / 16 / 16)) ** 0.5
        if supports_sigmas:
            scheduler.set_timesteps(sigmas=np.linspace(1.0, 0, args.steps + 1)[:-1],
                                    device=device, mu=mu)
        else:
            scheduler.set_timesteps(num_inference_steps=args.steps, device=device)

        img_shapes = build_img_shapes(args.num_layers, packed_h, packed_w)
        seq_len = (args.num_layers + 1) * packed_h * packed_w

        gen = torch.Generator(device=device).manual_seed(args.seed + idx)
        latents = torch.randn(1, seq_len, in_channels, device=device,
                              dtype=torch.bfloat16, generator=gen)

        latents = denoise(transformer, scheduler, latents, scheduler.timesteps,
                          prompt_embeds, prompt_mask, img_shapes, condition_latents,
                          args.guidance_scale, neg_embeds, neg_mask, additional_t_cond)

        decoded = decode_layers(vae, latents, th, tw, args.num_layers)
        composite = composite_layers(decoded)

        source.save(out_dir / "source.png")
        tensor_to_pil(composite[0]).save(out_dir / "composite.png")
        for li in range(args.num_layers):
            layer = decoded[0, li]
            pil = layer_to_pil_rgba(layer) if args.transparent else layer_to_pil_white_bg(layer)
            pil.save(out_dir / f"layer_{li}.png")

    print(f"\nDone -> {args.output}/<name>/{{source,composite,layer_0..{args.num_layers - 1}}}.png")


if __name__ == "__main__":
    main()
