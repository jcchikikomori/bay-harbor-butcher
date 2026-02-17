"""FLUX model pipeline building."""

import torch
from diffusers import AutoPipelineForText2Image

from utils.lora import load_lora_weights


def is_flux_model(model_id: str) -> bool:
    """Detect if the model is a FLUX variant."""
    flux_keywords = ["flux", "flux.1"]
    return any(kw in model_id.lower() for kw in flux_keywords)


def build_flux_pipeline(
    model_id: str,
    device: str,
    dtype: torch.dtype,
    lora_path: str | None = None,
    lora_weight_name: str | None = None,
    lora_adapter_name: str | None = None,
) -> AutoPipelineForText2Image:
    """
    Build FLUX text-to-image pipeline.

    Args:
        model_id: HuggingFace model ID (must be FLUX variant)
        device: Target device ('cuda' or 'cpu')
        dtype: torch dtype for model
        lora_path: Optional LoRA path
        lora_weight_name: Optional LoRA weight file name
        lora_adapter_name: Optional LoRA adapter name

    Returns:
        AutoPipelineForText2Image: Configured FLUX pipeline
    """
    print(f"✓ Detected FLUX model: {model_id}")

    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )
    pipe = pipe.to(device)

    # FLUX doesn't support safety_checker
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    load_lora_weights(
        pipe,
        lora_path,
        lora_weight_name,
        lora_adapter_name,
    )

    return pipe
