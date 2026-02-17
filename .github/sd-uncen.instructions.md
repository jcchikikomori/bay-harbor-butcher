---
description: This file provides instructions for setting up and running the Stable Diffusion Uncensored project
applyTo: "**"
---

# Project Instructions

## Setup

### Python Version

- `.python-version` file ensures `pyenv` uses Python 3.11.4
- Create a virtual environment: `python -m venv venv` or `pyenv virtualenv 3.11.4 sd-uncen`

### Dependencies

**CUDA (default):**

```bash
pip install -r requirements.txt
```

**ROCm (AMD GPU):**

```bash
pip install -r requirements-rocm.txt
```

**CPU only:**
Install normally; the app detects GPU availability at runtime.

## Running the App

### Basic Usage

```bash
python app.py --help                                    # View all options
python app.py                                           # Use .env settings
python app.py --prompt "your prompt here"              # Override prompt
```

### Text-to-Image

```bash
python app.py --prompt "a cat wearing sunglasses"
```

### Image-to-Image

```bash
python app.py --init-image path/to/image.jpg --strength 0.7
```

### With LoRA Weights

```bash
python app.py --lora-path ./my_loras --lora-weight-name model.safetensors
```

## Environment Variables (.env)

Copy `.env.example` to `.env` and customize:

### Core Settings

- `SD_PROMPT` - Text prompt for generation
- `SD_NEGATIVE_PROMPT` - Terms to exclude from output
- `SD_IMAGE_HEIGHT` - Output height (default: 512)
- `SD_IMAGE_WIDTH` - Output width (default: 512)
- `SD_NUM_INFERENCE_STEPS` - Quality/speed tradeoff (default: 50)

### Model & Pipeline

- `SD_MODEL` - HuggingFace model ID (default: `runwayml/stable-diffusion-v1-5`)
- `SD_USE_AUTO_PIPELINE` - Use `AutoPipelineForText2Image` instead of standard pipeline (true/false)
- `SD_TORCH_DTYPE` - Model precision: `float16` (GPU default), `float32` (CPU default), or `bfloat16` (recommended for FLUX)

### Image-to-Image

- `SD_STRENGTH` - How much to change image (0.0-1.0, default: 0.7)
- `SD_GUIDANCE_SCALE` - Prompt adherence strength (default: 7.5)

### LoRA Weights

- `SD_LORA_PATH` - Local path or HuggingFace model ID
  - Local: `./loras/my_style` or `../path/to/loras`
  - HuggingFace: `ostris/OpenXL-LoRA-library` or `username/my-lora`
- `SD_LORA_WEIGHT_NAME` - Specific weight file name (e.g., `pytorch_lora_weights.safetensors`)
- `SD_LORA_ADAPTER_NAME` - Adapter name for multi-LoRA management (optional)

### HuggingFace Authentication

- `HF_TOKEN` - Your HuggingFace API token (for gated models)
  - Get from: https://huggingface.co/settings/tokens
- `HF_SKIP_LOGIN` - Skip login (set to 1 for public models only)

## Output

- Generated images are saved to `output/`
- Files auto-increment: `output_1.png`, `output_2.png`, etc. (no overwrites)
- Run multiple times to generate variations

## Device Selection

**Priority order:**

1. CUDA (NVIDIA)
2. ROCm (AMD)
3. CPU (with fallback warning)

The app detects GPU availability automatically.

## Advanced: LoRA Examples

### Local LoRA

```bash
# In .env:
SD_LORA_PATH="./loras/my_style"
SD_LORA_WEIGHT_NAME="pytorch_lora_weights.safetensors"

# Or via CLI:
python app.py --lora-path ./loras/my_style --lora-weight-name pytorch_lora_weights.safetensors
```

### HuggingFace LoRA

```bash
# In .env:
SD_LORA_PATH="ostris/OpenXL-LoRA-library"

# Or via CLI:
python app.py --lora-path "ostris/OpenXL-LoRA-library"
```

## Advanced: AutoPipeline with FLUX

```bash
# In .env:
SD_USE_AUTO_PIPELINE=true
SD_MODEL="black-forest-labs/FLUX.1-dev"
SD_LORA_PATH="./loras/flux_style"

# Run:
python app.py
```

## Testing

```bash
python -m unittest
```

## Troubleshooting

**HuggingFace Login Issues:**

- Browser won't open? Visit https://huggingface.co/settings/tokens manually
- Copy your API token and add to `.env`: `HF_TOKEN="hf_your_token_here"`

**Dependencies Conflicts:**

- Run: `pip install -r requirements.txt --upgrade`
- Or clear venv: `rm -rf venv && python -m venv venv && pip install -r requirements.txt`

**GPU Not Detected:**

- Verify drivers: `nvidia-smi` (CUDA) or `rocm-smi` (ROCm)
- Falls back to CPU automatically; results will be slow

**LoRA Not Loading:**

- Check path exists: `ls SD_LORA_PATH`
- Check weight file name is correct
- Check error message for format compatibility
