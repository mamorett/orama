"""
Registry of all supported vision models with their HuggingFace IDs and metadata.
"""

SUPPORTED_MODELS = {
    # --- LFM2.5-VL family ---
    "LiquidAI/LFM2.5-VL-1.6B-Extract": {
        "family": "lfm2_vl",
        "display_name": "LFM2.5-VL 1.6B Extract (Liquid AI)",
        "model_class": "AutoModelForImageTextToText",
        "processor_class": "AutoProcessor",
        "trust_remote_code": False,
        "dtype": "auto",
        "supports_system_prompt": True,
        "chat_template": "apply_chat_template",
        "notes": "Liquid AI's efficient 1.6B VL extraction model",
    },

    # --- Step3-VL family ---
    "stepfun-ai/Step3-VL-10B": {
        "family": "step3_vl",
        "display_name": "Step3-VL 10B (StepFun)",
        "model_class": "AutoModelForCausalLM",
        "processor_class": "AutoProcessor",
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "supports_system_prompt": True,
        "chat_template": "apply_chat_template",
        "key_mapping": {
            "^vision_model": "model.vision_model",
            r"^model(?!\.(language_model|vision_model))": "model.language_model",
            "vit_large_projector": "model.vit_large_projector",
        },
        "notes": "StepFun's 10B frontier VL model — bf16 only",
    },
    "huihui-ai/Huihui-Step3-VL-10B-abliterated": {
        "family": "step3_vl",
        "display_name": "Huihui Step3-VL 10B Abliterated",
        "model_class": "AutoModelForCausalLM",
        "processor_class": "AutoProcessor",
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "supports_system_prompt": True,
        "chat_template": "apply_chat_template",
        "key_mapping": {
            "^vision_model": "model.vision_model",
            r"^model(?!\.(language_model|vision_model))": "model.language_model",
            "vit_large_projector": "model.vit_large_projector",
        },
        "notes": "Uncensored Step3-VL 10B variant by huihui-ai",
    },

    # --- MiniCPM-V family ---
    "openbmb/MiniCPM-V-4.6": {
        "family": "minicpm_v",
        "display_name": "MiniCPM-V 4.6 (OpenBMB)",
        "model_class": "AutoModelForImageTextToText",
        "processor_class": "AutoProcessor",
        "trust_remote_code": False,
        "dtype": "auto",
        "supports_system_prompt": True,
        "chat_template": "apply_chat_template",
        "notes": "OpenBMB's edge-friendly VL model based on SigLIP2 + Qwen3.5-0.8B",
    },
    "huihui-ai/Huihui-MiniCPM-V-4.6-abliterated": {
        "family": "minicpm_v",
        "display_name": "Huihui MiniCPM-V 4.6 Abliterated",
        "model_class": "AutoModelForImageTextToText",
        "processor_class": "AutoProcessor",
        "trust_remote_code": False,
        "dtype": "auto",
        "supports_system_prompt": True,
        "chat_template": "apply_chat_template",
        "notes": "Uncensored MiniCPM-V 4.6 variant by huihui-ai",
    },
}

LFM2_VL_MODELS = [k for k, v in SUPPORTED_MODELS.items() if v["family"] == "lfm2_vl"]
STEP3_VL_MODELS = [k for k, v in SUPPORTED_MODELS.items() if v["family"] == "step3_vl"]
MINICPM_V_MODELS = [k for k, v in SUPPORTED_MODELS.items() if v["family"] == "minicpm_v"]
ALL_MODELS = list(SUPPORTED_MODELS.keys())
