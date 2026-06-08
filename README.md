# ComfyUI Vision Models

A ComfyUI custom node package for running state-of-the-art open-source vision-language models locally.

---

## Supported Models

| Model | Family | Parameters | Notes |
|---|---|---|---|
| `LiquidAI/LFM2.5-VL-1.6B-Extract` | LFM2.5-VL | 1.6 B | Liquid AI extraction-focused VLM |
| `stepfun-ai/Step3-VL-10B` | Step3-VL | 10 B | StepFun frontier VLM — bf16 only |
| `huihui-ai/Huihui-Step3-VL-10B-abliterated` | Step3-VL | 10 B | Uncensored Step3-VL variant |
| `openbmb/MiniCPM-V-4.6` | MiniCPM-V | ~1 B | Edge-efficient VLM (SigLIP2 + Qwen3.5) |
| `huihui-ai/Huihui-MiniCPM-V-4.6-abliterated` | MiniCPM-V | ~1 B | Uncensored MiniCPM-V variant |

---

## Installation

1. Clone this repo into `ComfyUI/custom_nodes/`:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/your-repo/comfyui-vision-models
   ```

2. Install Python dependencies:
   ```bash
   pip install -r ComfyUI/custom_nodes/comfyui-vision-models/requirements.txt
   ```

3. **Step3-VL only** — requires `transformers>=4.57.0`:
   ```bash
   pip install "transformers>=4.57.0"
   ```

4. **MiniCPM-V & LFM2.5-VL** — requires `transformers>=5.7.0`:
   ```bash
   pip install "transformers[torch]>=5.7.0" torchvision
   # optionally for video:
   pip install av   # or torchcodec (CUDA 13.1+)
   ```

5. Restart ComfyUI.

---

## Model Storage

Models are **automatically downloaded on first use** and stored inside ComfyUI's own `models/` directory — completely separate from the system HuggingFace cache (`~/.cache/huggingface`).

```
ComfyUI/
└── models/
    └── vision_models/
        ├── LiquidAI/
        │   └── LFM2.5-VL-1.6B-Extract/   ← weights land here
        ├── stepfun-ai/
        │   └── Step3-VL-10B/
        ├── huihui-ai/
        │   ├── Huihui-Step3-VL-10B-abliterated/
        │   └── Huihui-MiniCPM-V-4.6-abliterated/
        └── openbmb/
            └── MiniCPM-V-4.6/
```

The resolution order for finding `ComfyUI/models/`:

1. `COMFYUI_MODEL_DIR` environment variable (explicit override)
2. `folder_paths.models_dir` — ComfyUI's own runtime path registry
3. Walk up the directory tree from the custom node until a folder with `main.py` + `models/` is found
4. Fallback: `~/ComfyUI/models`

Downloads use `huggingface_hub.snapshot_download` with `local_dir_use_symlinks=False`, so every file is a real copy in your `models/` folder with no dependency on the system HF cache at all. Subsequent loads skip the download check and go straight to the local path.

### Gated models / HF token

If a model requires HuggingFace authentication, set your token before starting ComfyUI:
```bash
export HF_TOKEN=hf_...
```
The token is picked up automatically from the environment.

---

## Node Reference

### Model Loaders
Each model family has a dedicated loader that downloads the model from HuggingFace (first run) and caches it in GPU memory for subsequent runs.

| Node | Output type | Key options |
|---|---|---|
| **Load LFM2.5-VL Model** | `LFM2_VL_MODEL` | model_id, dtype, device_map, flash_attn |
| **Load Step3-VL Model** | `STEP3_VL_MODEL` | model_id, device_map, flash_attn |
| **Load MiniCPM-V Model** | `MINICPM_V_MODEL` | model_id, dtype, device_map, flash_attn, visual compression |

### Inference Nodes
Connect a loaded model + a ComfyUI IMAGE + your prompt text.

| Node | Inputs | Output |
|---|---|---|
| **LFM2.5-VL (Liquid AI)** | model, image, prompt, gen settings, *system_prompt* | STRING |
| **Step3-VL (StepFun)** | model, image, prompt, gen settings, *system_prompt* | STRING |
| **MiniCPM-V (OpenBMB)** | model, image, prompt, gen settings, *system_prompt* | STRING |

### Utility Nodes
| Node | Purpose |
|---|---|
| **Vision Model Image Input** | Resize / forward a ComfyUI IMAGE tensor to a vision model |
| **Vision Model Settings** | Reusable generation parameters block (connect to any inference node) |

---

## Generation Parameters (all inference nodes)

| Parameter | Default | Description |
|---|---|---|
| `max_new_tokens` | 512 | Maximum tokens to generate |
| `temperature` | 0.7 | Sampling temperature (0 = greedy) |
| `top_k` | 50 | Top-K sampling (0 = disabled) |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `repetition_penalty` | 1.0 | Penalise repeated tokens (>1 = stronger) |
| `do_sample` | True | Enable stochastic sampling (False = greedy) |

All parameters can also be set once via the **Vision Model Settings** node and wired to multiple inference nodes.

---

## Typical Workflow

```
[Load Image] → [Vision Model Image Input] ──┐
                                            ↓
[Load Step3-VL Model] ──────────────→ [Step3-VL Node] → [Show Text]
                                            ↑
[Vision Model Settings] ────────────────────┘
```

---

## Hardware Requirements

| Model | Min VRAM | Recommended |
|---|---|---|
| LFM2.5-VL-1.6B | ~4 GB | 6 GB |
| MiniCPM-V-4.6 | ~3 GB | 6 GB |
| Step3-VL-10B | ~20 GB | 24 GB (bf16) |

For Step3-VL on consumer hardware, use the FP8 quantized checkpoint: `stepfun-ai/Step3-VL-10B-FP8`.

---

## Model-Specific Notes

### LFM2.5-VL
- Requires `transformers>=5.7.0` (for `Lfm2VlProcessor` and `TokenizersBackend`).
- Standard `AutoModelForImageTextToText` / `apply_chat_template` pattern.
- No `trust_remote_code` needed.

### Step3-VL
- **Only bf16 dtype is officially supported** — the node enforces this.
- `trust_remote_code=True` is mandatory (custom architecture code).
- A `key_mapping` dict is applied at load time to rename checkpoint keys to match the registered architecture.
- For infinite-generation issues, see [Discussion #9](https://huggingface.co/stepfun-ai/Step3-VL-10B/discussions/9).

### MiniCPM-V
- Requires `transformers >= 5.7.0`.
- Supports **4x / 16x visual token compression**:
  - **16x (fast)** — fewer visual tokens, faster inference (default).
  - **4x (detailed)** — more visual tokens, better for text-heavy / detail tasks.
- Flash Attention 2 is especially beneficial for multi-image scenarios.
- The abliterated variant (`Huihui-MiniCPM-V-4.6-abliterated`) uses identical inference code.

---

## License

This node package is MIT licensed. Each model has its own license — please review before commercial use:
- LFM2.5-VL: LFM-1.0 license
- Step3-VL: Apache 2.0
- MiniCPM-V: Apache 2.0
