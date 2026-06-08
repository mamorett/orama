"""
Shared utilities for vision model inference nodes.
"""

import gc
import io
import os
import torch
import numpy as np
from PIL import Image
from typing import Optional

# ─── Global model cache ───────────────────────────────────────────────────────
_MODEL_CACHE: dict = {}


def get_cached_model(cache_key: str):
    return _MODEL_CACHE.get(cache_key)


def set_cached_model(cache_key: str, model, processor):
    _MODEL_CACHE[cache_key] = (model, processor)


def clear_model_cache(cache_key: Optional[str] = None):
    """Free a specific model or the entire cache."""
    global _MODEL_CACHE
    if cache_key:
        if cache_key in _MODEL_CACHE:
            model, processor = _MODEL_CACHE.pop(cache_key)
            del model, processor
    else:
        for k in list(_MODEL_CACHE.keys()):
            model, processor = _MODEL_CACHE.pop(k)
            del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─── Image helpers ────────────────────────────────────────────────────────────

def comfy_image_to_pil(image_tensor) -> Image.Image:
    """
    Convert a ComfyUI IMAGE tensor (B, H, W, C float32 0-1)
    to a PIL RGB image (first frame only).
    """
    # ComfyUI images are (batch, H, W, C) in [0,1]
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor[0]          # take first in batch
    np_img = (image_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(np_img, "RGB")


def pil_to_bytes(pil_image: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt)
    return buf.getvalue()


# ─── Generation parameter helpers ─────────────────────────────────────────────

def build_generation_kwargs(
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    do_sample: bool,
) -> dict:
    """Build a clean kwargs dict for model.generate()."""
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
    }
    if do_sample:
        kwargs["temperature"] = temperature
        kwargs["top_k"] = top_k if top_k > 0 else None
        kwargs["top_p"] = top_p
    return kwargs


# ─── Device helpers ───────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_torch_dtype(dtype_str: str):
    mapping = {
        "auto":       "auto",
        "bfloat16":   torch.bfloat16,
        "float16":    torch.float16,
        "float32":    torch.float32,
    }
    return mapping.get(dtype_str, "auto")


# ─── Decode helpers ───────────────────────────────────────────────────────────

def decode_output(output_ids, input_ids, tokenizer) -> str:
    """Strip the prompt tokens and decode the generated portion."""
    prompt_len = input_ids.shape[-1]
    generated = output_ids[0][prompt_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
