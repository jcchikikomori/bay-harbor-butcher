"""Stable Diffusion model pipeline building."""

import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline

from utils.lora import load_lora_weights


def build_stable_diffusion_pipeline(
    model_id: str,
    device: str,
    dtype: torch.dtype,
    use_img2img: bool = False,
    lora_path: str | None = None,
    lora_weight_name: str | None = None,
    lora_adapter_name: str | None = None,
) -> StableDiffusionPipeline | StableDiffusionImg2ImgPipeline:
    """
    Build Stable Diffusion text-to-image or image-to-image pipeline.

    Args:
        model_id: HuggingFace model ID
        device: Target device ('cuda' or 'cpu')
        dtype: torch dtype for model
        use_img2img: If True, build Img2ImgPipeline; else text-to-image
        lora_path: Optional LoRA path
        lora_weight_name: Optional LoRA weight file name
        lora_adapter_name: Optional LoRA adapter name

    Returns:
        StableDiffusionPipeline or StableDiffusionImg2ImgPipeline
    """
    if use_img2img:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )

    pipe = pipe.to(device)

    # Disable safety_checker for Stable Diffusion
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    load_lora_weights(
        pipe,
        lora_path,
        lora_weight_name,
        lora_adapter_name,
    )

    return pipe
