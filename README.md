# The Bay Harbor Butcher

James Doakes we're right after all.
I just needed a uncensored version of Stable Diffusion.

Generate images using Stable Diffusion with support for custom models, LoRA weights, and various GPU backends.

## Quick Setup

### 1. Clone & Install Dependencies

```bash
# Create virtual environment (Python 3.11.4)
pyenv virtualenv 3.11.4 sd-uncen
pyenv local sd-uncen

# Install dependencies
pip install -r requirements.txt
```

**For ROCm (AMD GPU):**

```bash
pip install -r requirements-rocm.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and customize settings:

```bash
cp .env.example .env
```

**Read `.env.example` for all available options:**

- `SD_PROMPT` - Your generation prompt
- `SD_MODEL` - HuggingFace model ID
- `SD_TORCH_DTYPE` - Precision (float16, float32, bfloat16)
- `SD_LORA_PATH` - LoRA weights (local or HuggingFace)
- `HF_TOKEN` - For gated models (optional)
- And more!

### 3. Run

```bash
python app.py --help                          # View all options
python app.py                                 # Use .env settings
python app.py --prompt "your prompt here"   # Override prompt
```

## Examples

### Text-to-Image

```bash
python app.py --prompt "a cat wearing sunglasses"
```

### Image-to-Image

```bash
python app.py --init-image path/to/image.jpg --strength 0.7
```

### With LoRA

```bash
python app.py --lora-path ./my_loras --lora-weight-name model.safetensors
```

### FLUX with bfloat16

```bash
python app.py --model "black-forest-labs/FLUX.1-dev" --torch-dtype bfloat16
```

## Output

- Images saved to `output/` folder
- Auto-increments: `output_1.png`, `output_2.png`, etc.
- No overwrites

## Device Support

- **CUDA** (NVIDIA) - Preferred
- **ROCm** (AMD, Steam Deck, etc.) - Via `requirements-rocm.txt`
- **CPU** - Fallback with warning

Auto-detected at runtime.

## Full Documentation

For detailed environment variables and advanced configuration, see [.env.example](.env.example) and [.github/sd-uncen.instructions.md](.github/sd-uncen.instructions.md).

## Troubleshooting

### HuggingFace Login

- Visit: <https://huggingface.co/settings/tokens>
- Copy token to `.env`: `HF_TOKEN="hf_xxx"`

### Dependencies Issues

```bash
pip install -r requirements.txt --upgrade
```

### GPU Not Detected

```bash
nvidia-smi      # CUDA
rocm-smi        # ROCm
```

## Testing

```bash
python -m unittest
```
