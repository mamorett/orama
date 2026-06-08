"""
Common utility nodes: image input adapter and shared settings block.
"""

import torch


class VisionModelImageInput:
    """
    Passes a ComfyUI IMAGE tensor through unchanged, with an optional
    resize so it can be plugged into any vision model node.
    """

    CATEGORY = "orama"
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "max_side_px": ("INT", {
                    "default": 0, "min": 0, "max": 4096, "step": 64,
                    "tooltip": "Resize longest side to this value (0 = no resize)",
                }),
            },
        }

    def process(self, image, max_side_px: int = 0):
        if max_side_px > 0:
            import torchvision.transforms.functional as TF
            # image is (B, H, W, C) float32
            b, h, w, c = image.shape
            scale = max_side_px / max(h, w)
            if scale < 1.0:
                new_h = int(h * scale)
                new_w = int(w * scale)
                # permute to (B, C, H, W) for torchvision, then back
                img_bchw = image.permute(0, 3, 1, 2)
                img_bchw = TF.resize(img_bchw, [new_h, new_w], antialias=True)
                image = img_bchw.permute(0, 2, 3, 1)
        return (image,)


class VisionModelSettings:
    """
    A reusable settings block that can be wired into any vision model node.
    Exposes generation parameters as a VISION_SETTINGS passthrough.
    """

    CATEGORY = "orama"
    FUNCTION = "build_settings"
    RETURN_TYPES = ("VISION_SETTINGS",)
    RETURN_NAMES = ("settings",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_new_tokens": ("INT", {
                    "default": 512, "min": 1, "max": 8192, "step": 1,
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01,
                }),
                "top_k": ("INT", {
                    "default": 50, "min": 0, "max": 1000, "step": 1,
                    "tooltip": "0 = disabled",
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.0, "min": 1.0, "max": 2.0, "step": 0.01,
                }),
                "do_sample": ("BOOLEAN", {"default": True}),
            },
        }

    def build_settings(
        self,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        do_sample: bool,
    ):
        settings = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
        }
        return (settings,)
