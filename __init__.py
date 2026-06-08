"""
ComfyUI Vision Models - Custom nodes for LFM2.5-VL, Step3-VL, and MiniCPM-V models
"""

# Patch transformers Lfm2VlImageProcessor / Lfm2VlImageProcessorFast naming mismatch
import transformers
if not hasattr(transformers, "Lfm2VlImageProcessorFast") and hasattr(transformers, "Lfm2VlImageProcessor"):
    transformers.Lfm2VlImageProcessorFast = transformers.Lfm2VlImageProcessor

from .nodes.lfm2_vl_node import LFM25VLNode, LFM25VLModelLoader
from .nodes.step3_vl_node import Step3VLNode, Step3VLModelLoader
from .nodes.minicpm_v_node import MiniCPMVNode, MiniCPMVModelLoader
from .nodes.common_nodes import VisionModelImageInput, VisionModelSettings

NODE_CLASS_MAPPINGS = {
    # Loaders
    "LFM25VLModelLoader": LFM25VLModelLoader,
    "Step3VLModelLoader": Step3VLModelLoader,
    "MiniCPMVModelLoader": MiniCPMVModelLoader,
    # Inference nodes
    "LFM25VLNode": LFM25VLNode,
    "Step3VLNode": Step3VLNode,
    "MiniCPMVNode": MiniCPMVNode,
    # Utility nodes
    "VisionModelImageInput": VisionModelImageInput,
    "VisionModelSettings": VisionModelSettings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Loaders
    "LFM25VLModelLoader":   "Load LFM2.5-VL Model",
    "Step3VLModelLoader":   "Load Step3-VL Model",
    "MiniCPMVModelLoader":  "Load MiniCPM-V Model",
    # Inference
    "LFM25VLNode":          "LFM2.5-VL (Liquid AI)",
    "Step3VLNode":          "Step3-VL (StepFun)",
    "MiniCPMVNode":         "MiniCPM-V (OpenBMB)",
    # Utility
    "VisionModelImageInput": "Vision Model Image Input",
    "VisionModelSettings":  "Vision Model Settings",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
