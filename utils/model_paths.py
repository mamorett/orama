"""
model_paths.py
~~~~~~~~~~~~~~
Resolves the ComfyUI ``models/`` directory at runtime and provides helpers
that force every HuggingFace ``from_pretrained`` call to use it as the
download cache — completely bypassing the system HF_HOME / HF_HUB_CACHE.

Resolution order (first one that exists wins):
  1. COMFYUI_MODEL_DIR environment variable
  2. ``folder_paths.models_dir`` from ComfyUI's own path registry
  3. Walk up from this file's location until we find a ``models/`` sibling
     of a directory that also contains ``main.py`` (i.e. the ComfyUI root)
  4. Fallback: ``~/ComfyUI/models``

The resolved path is stored in ``VISION_MODELS_DIR`` and every HF model
is saved under ``<VISION_MODELS_DIR>/vision_models/<org>/<repo>``.
"""

import os
import pathlib

# ─── 1. Resolve ComfyUI models/ root ─────────────────────────────────────────

def _resolve_comfyui_models_dir() -> pathlib.Path:
    # Explicit env override
    env = os.environ.get("COMFYUI_MODEL_DIR")
    if env:
        p = pathlib.Path(env)
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ComfyUI's own folder_paths module (always available inside ComfyUI)
    try:
        import folder_paths  # type: ignore
        p = pathlib.Path(folder_paths.models_dir)
        if p.exists():
            return p
    except ImportError:
        pass

    # Walk up from this file searching for ComfyUI root (has main.py + models/)
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "main.py").exists() and (parent / "models").is_dir():
            return parent / "models"

    # Last resort
    fallback = pathlib.Path.home() / "ComfyUI" / "models"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


# Public constant — import this everywhere
COMFYUI_MODELS_DIR: pathlib.Path = _resolve_comfyui_models_dir()

# All VLM weights live under models/vision_models/
VISION_MODELS_DIR: pathlib.Path = COMFYUI_MODELS_DIR / "vision_models"
VISION_MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ─── 2. Per-model local path ──────────────────────────────────────────────────

def get_local_model_dir(model_id: str) -> pathlib.Path:
    """
    Return the local directory where a HuggingFace model should be stored.

    ``model_id`` is the HF repo ID, e.g. ``"stepfun-ai/Step3-VL-10B"``.
    This maps to ``<VISION_MODELS_DIR>/stepfun-ai/Step3-VL-10B``.
    """
    # model_id may contain "/" — keep the org/repo hierarchy
    local = VISION_MODELS_DIR / model_id.replace("\\", "/")
    local.mkdir(parents=True, exist_ok=True)
    return local


def model_is_downloaded(model_id: str) -> bool:
    """
    Return True if the model directory looks populated (contains at least
    one .safetensors or .bin file, or a config.json).
    """
    local = get_local_model_dir(model_id)
    for ext in ("*.safetensors", "*.bin", "config.json"):
        if list(local.glob(ext)):
            return True
    return False


# ─── 3. Download helper ───────────────────────────────────────────────────────

def ensure_model_downloaded(model_id: str, token: str = None) -> pathlib.Path:
    """
    Download ``model_id`` from HuggingFace into ``VISION_MODELS_DIR`` if not
    already present.  Returns the local directory path.

    We use ``huggingface_hub.snapshot_download`` which:
    - Skips files that already exist (resume-safe).
    - Respects ``HF_TOKEN`` / ``token`` for gated models.
    - Does NOT touch the system HF cache.
    """
    local_dir = get_local_model_dir(model_id)

    if model_is_downloaded(model_id):
        print(f"[VisionModels] Model already present: {local_dir}")
        return local_dir

    print(f"[VisionModels] Downloading {model_id} → {local_dir} …")
    print(f"[VisionModels] This may take a while on first run.")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for automatic model download. "
            "Run: pip install huggingface_hub"
        )

    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,   # copy files, no symlinks into HF cache
        token=token,
        ignore_patterns=["*.gguf", "*.pt", "flax_model*", "tf_model*",
                         "rust_model*", "onnx/*"],
    )

    print(f"[VisionModels] Download complete: {local_dir}")
    return local_dir


# ─── 4. Convenience: get path to pass to from_pretrained ─────────────────────

def get_model_path(model_id: str, hf_token: str = None) -> str:
    """
    Ensure model is downloaded and return its local path as a string,
    ready to pass directly to ``AutoProcessor.from_pretrained(path)`` etc.
    """
    local_dir = ensure_model_downloaded(model_id, token=hf_token)
    return str(local_dir)
