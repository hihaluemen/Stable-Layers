#!/usr/bin/env python3
"""Small HTTP wrapper for Stable-Layers inference.

This intentionally keeps the model runner in ``decompose.py`` so the fork's
published inference path remains the source of truth. It is suitable for a
single-GPU validation service; production deployments should keep one worker
per GPU and put authentication/TLS in a reverse proxy.
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image


ROOT = Path(__file__).resolve().parent
MODEL = os.getenv("STABLE_LAYERS_MODEL", "Qwen/Qwen-Image-Layered")
LORA = os.getenv("STABLE_LAYERS_LORA", str(ROOT / "model"))
DEVICE = os.getenv("STABLE_LAYERS_DEVICE", "cuda")
STEPS = int(os.getenv("STABLE_LAYERS_STEPS", "50"))
GUIDANCE = float(os.getenv("STABLE_LAYERS_GUIDANCE", "1.0"))
NUM_LAYERS = int(os.getenv("STABLE_LAYERS_NUM_LAYERS", "4"))
SIZE = int(os.getenv("STABLE_LAYERS_SIZE", "640"))
TIMEOUT = int(os.getenv("STABLE_LAYERS_TIMEOUT_SECONDS", "1800"))
ALPHA_THRESHOLD = max(0, min(255, int(os.getenv("STABLE_LAYERS_ALPHA_THRESHOLD", "16"))))

app = FastAPI(title="Stable-Layers inference service", version="1")


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "model": MODEL,
        "lora": LORA,
        "device": DEVICE,
        "settings": {
            "steps": STEPS,
            "guidance_scale": GUIDANCE,
            "num_layers": NUM_LAYERS,
            "size": SIZE,
            "alpha_threshold": ALPHA_THRESHOLD,
        },
    }


@app.post("/v1/layer-decomposition")
async def layer_decomposition(image: UploadFile = File(...)) -> dict:
    source = await image.read()
    if not source:
        raise HTTPException(400, "image is empty")
    if len(source) > 24 * 1024 * 1024:
        raise HTTPException(413, "image is larger than 24 MB")
    try:
        with Image.open(io.BytesIO(source)) as decoded:
            source_size = {"width": decoded.width, "height": decoded.height}
    except Exception as error:
        raise HTTPException(400, "image cannot be decoded") from error

    job = Path(tempfile.mkdtemp(prefix="stable-layers-"))
    input_path = job / "input.png"
    output_path = job / "output"
    input_path.write_bytes(source)
    command = [
        os.environ.get("PYTHON", "python"),
        str(ROOT / "decompose.py"),
        "--input", str(input_path),
        "--output", str(output_path),
        "--lora", LORA,
        "--base-model", MODEL,
        "--steps", str(STEPS),
        "--guidance-scale", str(GUIDANCE),
        "--num-layers", str(NUM_LAYERS),
        "--size", str(SIZE),
        "--device", DEVICE,
        "--transparent",
    ]
    try:
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, timeout=TIMEOUT,
        )
        if completed.returncode:
            detail = (completed.stderr or completed.stdout or "inference failed")[-2000:]
            raise HTTPException(502, detail)
        result_dir = output_path / input_path.stem
        with Image.open(result_dir / "source.png") as resized_source:
            coordinate_size = {
                "width": resized_source.width,
                "height": resized_source.height,
            }
        layers = []
        for index in range(NUM_LAYERS):
            layer_path = result_dir / f"layer_{index}.png"
            if not layer_path.exists():
                continue
            payload = layer_path.read_bytes()
            with Image.open(io.BytesIO(payload)) as layer:
                rgba = _clean_alpha(layer.convert("RGBA"))
                alpha = rgba.getchannel("A")
                bbox = alpha.getbbox()
                if bbox is None:
                    continue
                encoded = io.BytesIO()
                rgba.save(encoded, "PNG")
                layers.append({
                    "name": f"layer_{index}",
                    "description": "background" if index == 0 else "object layer",
                    "image_base64": base64.b64encode(encoded.getvalue()).decode("ascii"),
                    "bounding_box": {
                        "x": bbox[0], "y": bbox[1],
                        "w": bbox[2] - bbox[0], "h": bbox[3] - bbox[1],
                    },
                    "z_index": index,
                    "is_background": index == 0,
                })
        if not layers:
            raise HTTPException(502, "Stable-Layers returned no non-empty layers")
        return {
            "request_id": uuid.uuid4().hex,
            "model": "Stable-Layers",
            "source_size": source_size,
            "coordinate_size": coordinate_size,
            "layers": layers,
        }
    finally:
        shutil.rmtree(job, ignore_errors=True)


def _clean_alpha(rgba: Image.Image) -> Image.Image:
    """Remove near-transparent diffusion noise before bbox calculation."""
    if ALPHA_THRESHOLD <= 0:
        return rgba
    pixels = np.asarray(rgba).copy()
    alpha = pixels[:, :, 3]
    alpha[alpha < ALPHA_THRESHOLD] = 0
    pixels[:, :, 3] = alpha
    return Image.fromarray(pixels, mode="RGBA")
