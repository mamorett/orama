"""
ComfyUI nodes for StepFun Step3-VL models (and huihui-ai abliterated variant).

Inference pattern (from official README):
    from transformers import AutoProcessor, AutoModelForCausalLM

    key_mapping = {
        "^vision_model": "model.vision_model",
        r"^model(?!\.(language_model|vision_model))": "model.language_model",
        "vit_large_projector": "model.vit_large_projector",
    }

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto",
        key_mapping=key_mapping,
    ).eval()

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=1024)
    result = processor.decode(output_ids[0][inputs["input_ids"].shape[-1]:],
                              skip_special_tokens=True)

Notes:
- Only bf16 is officially supported.
- trust_remote_code=True is mandatory.
- key_mapping handles checkpoint key renaming at load time.
"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM

from ..model_registry import STEP3_VL_MODELS, SUPPORTED_MODELS
from ..utils import (
    comfy_image_to_pil,
    build_generation_kwargs,
    get_cached_model,
    set_cached_model,
    get_model_path,
)


# ─── Loader node ──────────────────────────────────────────────────────────────

class Step3VLModelLoader:
    """Downloads / loads a Step3-VL model and caches it in memory."""

    CATEGORY = "orama/Step3-VL"
    FUNCTION = "load_model"
    RETURN_TYPES = ("STEP3_VL_MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": (STEP3_VL_MODELS, {
                    "default": STEP3_VL_MODELS[0],
                }),
                "device_map": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                }),
                "use_flash_attention_2": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Speeds up inference on supported GPUs. Requires flash-attn package.",
                }),
            },
        }

    def load_model(self, model_id: str, device_map: str, use_flash_attention_2: bool):
        cache_key = f"step3_vl::{model_id}"
        cached = get_cached_model(cache_key)
        if cached:
            print(f"[Step3-VL] Using cached model: {model_id}")
            return ({"model": cached[0], "processor": cached[1],
                     "model_id": model_id, "cache_key": cache_key},)

        print(f"[Step3-VL] Loading {model_id} …")
        meta = SUPPORTED_MODELS[model_id]

        # Ensure weights are in ComfyUI/models/vision_models/, download if needed
        local_path = get_model_path(model_id)

        key_mapping = meta.get("key_mapping", {})

        load_kwargs = dict(
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch.bfloat16,  # bf16 is the only officially supported dtype
        )
        if key_mapping:
            load_kwargs["key_mapping"] = key_mapping
        if use_flash_attention_2:
            load_kwargs["attn_implementation"] = "flash_attention_2"

        processor = AutoProcessor.from_pretrained(local_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(local_path, **load_kwargs).eval()

        set_cached_model(cache_key, model, processor)
        print(f"[Step3-VL] Model loaded.")

        return ({"model": model, "processor": processor,
                 "model_id": model_id, "cache_key": cache_key},)


# ─── Inference node ───────────────────────────────────────────────────────────

class Step3VLNode:
    """
    Run inference with a loaded Step3-VL model.
    Accepts an image + text prompt, plus an optional system prompt.
    """

    CATEGORY = "orama/Step3-VL"
    FUNCTION = "run_inference"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":  ("STEP3_VL_MODEL",),
                "image":  ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Describe this image in detail.",
                }),
                "max_new_tokens": ("INT", {
                    "default": 1024, "min": 1, "max": 65536, "step": 1,
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
        max_new_tokens: int = 1024,
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
