"""
ComfyUI nodes for OpenBMB MiniCPM-V 4.6 models (and huihui-ai abliterated variant).

Inference pattern (from HuggingFace model card / transformers docs):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_id = "openbmb/MiniCPM-V-4.6"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )

    # Standard chat-template approach (transformers >= 5.7.0)
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": pil_image},
            {"type": "text",  "text": prompt},
        ]}
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=512)
    result = processor.decode(output_ids[0][inputs["input_ids"].shape[-1]:],
                              skip_special_tokens=True)

Notes:
- Requires transformers >= 5.7.0.
- torchvision required; optionally torchcodec (video) or PyAV.
- Supports 4x / 16x visual token compression (via image_processor backend param).
- Flash Attention 2 is recommended for multi-image / video scenarios.
"""

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

from ..model_registry import MINICPM_V_MODELS, SUPPORTED_MODELS
from ..utils import (
    comfy_image_to_pil,
    build_generation_kwargs,
    get_cached_model,
    set_cached_model,
    get_torch_dtype,
    get_model_path,
)


# ─── Loader node ──────────────────────────────────────────────────────────────

class MiniCPMVModelLoader:
    """Downloads / loads a MiniCPM-V model and caches it in memory."""

    CATEGORY = "orama/MiniCPM-V"
    FUNCTION = "load_model"
    RETURN_TYPES = ("MINICPM_V_MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": (MINICPM_V_MODELS, {
                    "default": MINICPM_V_MODELS[0],
                }),
                "dtype": (["auto", "bfloat16", "float16", "float32"], {
                    "default": "auto",
                }),
                "device_map": (["auto", "cuda", "cpu", "mps"], {
                    "default": "auto",
                }),
                "use_flash_attention_2": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Recommended for multi-image tasks. Requires flash-attn package.",
                }),
                "visual_token_compression": (["16x (fast)", "4x (detailed)"], {
                    "default": "16x (fast)",
                    "tooltip": "16x = fewer tokens, faster. 4x = more detail.",
                }),
            },
        }

    def load_model(
        self,
        model_id: str,
        dtype: str,
        device_map: str,
        use_flash_attention_2: bool,
        visual_token_compression: str,
    ):
        compress_mode = "16x" if "16x" in visual_token_compression else "4x"
        cache_key = f"minicpm_v::{model_id}::{dtype}::{compress_mode}"
        cached = get_cached_model(cache_key)
        if cached:
            print(f"[MiniCPM-V] Using cached model: {model_id}")
            return ({"model": cached[0], "processor": cached[1],
                     "model_id": model_id, "cache_key": cache_key},)

        print(f"[MiniCPM-V] Loading {model_id} …")
        torch_dtype = get_torch_dtype(dtype)

        # Ensure weights are in ComfyUI/models/vision_models/, download if needed
        local_path = get_model_path(model_id)

        # Processor: optionally switch image backend for 4x compression
        processor = AutoProcessor.from_pretrained(local_path)
        if compress_mode == "4x":
            from transformers import AutoImageProcessor
            processor.image_processor = AutoImageProcessor.from_pretrained(
                local_path, compress_ratio=4
            )

        load_kwargs = dict(device_map=device_map)
        if torch_dtype != "auto":
            load_kwargs["torch_dtype"] = torch_dtype
        else:
            load_kwargs["torch_dtype"] = "auto"
        if use_flash_attention_2:
            load_kwargs["attn_implementation"] = "flash_attention_2"

        model = AutoModelForImageTextToText.from_pretrained(local_path, **load_kwargs)
        model.eval()

        set_cached_model(cache_key, model, processor)
        print(f"[MiniCPM-V] Model loaded.")

        return ({"model": model, "processor": processor,
                 "model_id": model_id, "cache_key": cache_key},)


# ─── Inference node ───────────────────────────────────────────────────────────

class MiniCPMVNode:
    """
    Run inference with a loaded MiniCPM-V model.
    Accepts an image + text prompt, plus an optional system prompt.
    """

    CATEGORY = "orama/MiniCPM-V"
    FUNCTION = "run_inference"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":  ("MINICPM_V_MODEL",),
                "image":  ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Describe this image in detail.",
                }),
                "max_new_tokens": ("INT", {
                    "default": 512, "min": 1, "max": 8192, "step": 1,
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01,
                }),
                "top_k": ("INT", {
                    "default": 50, "min": 0, "max": 1000, "step": 1,
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.0, "min": 1.0, "max": 2.0, "step": 0.01,
                }),
                "do_sample": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Optional system prompt.",
                }),
                "settings": ("VISION_SETTINGS",),
            },
        }

    def run_inference(
        self,
        model: dict,
        image,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        system_prompt: str = "",
        settings: dict = None,
    ):
        if settings:
            max_new_tokens     = settings.get("max_new_tokens",   max_new_tokens)
            temperature        = settings.get("temperature",      temperature)
            top_k              = settings.get("top_k",            top_k)
            top_p              = settings.get("top_p",            top_p)
            repetition_penalty = settings.get("repetition_penalty", repetition_penalty)
            do_sample          = settings.get("do_sample",        do_sample)

        mdl = model["model"]
        processor = model["processor"]

        pil_image = comfy_image_to_pil(image)

        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text",  "text": prompt},
            ],
        })

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(mdl.device)

        gen_kwargs = build_generation_kwargs(
            max_new_tokens, temperature, top_k, top_p, repetition_penalty, do_sample
        )

        with torch.no_grad():
            output_ids = mdl.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[-1]
        generated = output_ids[0][prompt_len:]
        result = processor.decode(generated, skip_special_tokens=True).strip()

        return (result,)
