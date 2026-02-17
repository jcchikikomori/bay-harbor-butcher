"""Stable Diffusion & FLUX image generation with auto-detection."""

import argparse
import os
import warnings

import torch
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    StableDiffusionImg2ImgPipeline,
)
from dotenv import load_dotenv
from huggingface_hub import login
from PIL import Image

from pipelines.flux import build_flux_pipeline, is_flux_model
from pipelines.stable_diffusion import build_stable_diffusion_pipeline
from utils.device import get_device, get_dtype

DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "output"


def get_next_output_path(output_dir: str, prefix: str = "output") -> str:
    """Generate next auto-incremented output path."""
    os.makedirs(output_dir, exist_ok=True)
    index = 1
    while True:
        output_path = os.path.join(output_dir, f"{prefix}_{index}.png")
        if not os.path.exists(output_path):
            return output_path
        index += 1


def ensure_hf_login() -> None:
    """Ensure HuggingFace authentication for gated models."""
    skip_login = os.getenv("HF_SKIP_LOGIN", "0").lower() == "1"
    if skip_login:
        return
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)
    else:
        try:
            login(add_to_git_credential=False)
        except Exception as e:
            print("\n" + "=" * 60)
            print("HuggingFace Login Required for Gated Models")
            print("=" * 60)
            print(
                "\nBrowser login failed. Please authenticate manually:\n"
                "1. Visit: https://huggingface.co/settings/tokens\n"
                "2. Create or copy an existing API token\n"
                '3. Add to .env: HF_TOKEN="hf_your_token_here"\n'
                "4. Restart this script\n"
            )
            print("=" * 60 + "\n")
            warnings.warn(f"HuggingFace login skipped: {e}", stacklevel=2)


def load_init_image(image_path: str, width: int, height: int) -> Image.Image:
    """Load and resize init image for img2img pipeline."""
    image = Image.open(image_path).convert("RGB")
    return image.resize((width, height))


def get_env_str(name: str, default: str) -> str:
    """Get string environment variable with default."""
    value = os.getenv(name)
    return value if value is not None else default


def get_env_optional_str(name: str) -> str | None:
    """Get optional string environment variable."""
    value = os.getenv(name)
    return value if value else None


def get_env_int(name: str, default: int) -> int:
    """Get integer environment variable with default."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        warnings.warn(f"Invalid int for {name}; using default.", stacklevel=2)
        return default


def get_env_float(name: str, default: float) -> float:
    """Get float environment variable with default."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        warnings.warn(f"Invalid float for {name}; using default.", stacklevel=2)
        return default


def get_env_bool(name: str, default: bool = False) -> bool:
    """Get boolean environment variable with default."""
    value = os.getenv(name, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    if value in ("false", "0", "no", "off"):
        return False
    return default


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments with dotenv defaults."""
    load_dotenv()
    default_prompt = get_env_str("SD_PROMPT", "a photo of an astronaut riding a horse on mars")
    default_negative_prompt = get_env_optional_str("SD_NEGATIVE_PROMPT")
    default_height = get_env_int("SD_IMAGE_HEIGHT", 512)
    default_width = get_env_int("SD_IMAGE_WIDTH", 512)
    default_model = get_env_str("SD_MODEL", DEFAULT_MODEL_ID)
    default_strength = get_env_float("SD_STRENGTH", 0.6)
    default_guidance_scale = get_env_float("SD_GUIDANCE_SCALE", 7.5)
    default_num_steps = get_env_int("SD_NUM_INFERENCE_STEPS", 50)
    default_lora_path = get_env_optional_str("SD_LORA_PATH")
    default_lora_weight_name = get_env_optional_str("SD_LORA_WEIGHT_NAME")
    default_lora_adapter_name = get_env_optional_str("SD_LORA_ADAPTER_NAME")
    default_torch_dtype = get_env_optional_str("SD_TORCH_DTYPE")

    parser = argparse.ArgumentParser(description="Stable Diffusion & FLUX image generation")
    parser.add_argument("--prompt", default=default_prompt)
    parser.add_argument("--negative-prompt", dest="negative_prompt", default=default_negative_prompt)
    parser.add_argument("--init-image", dest="init_image", default=None)
    parser.add_argument("--height", type=int, default=default_height)
    parser.add_argument("--width", type=int, default=default_width)
    parser.add_argument("--strength", type=float, default=default_strength)
    parser.add_argument("--guidance-scale", dest="guidance_scale", type=float, default=default_guidance_scale)
    parser.add_argument(
        "--num-inference-steps",
        dest="num_inference_steps",
        type=int,
        default=default_num_steps,
    )
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--lora-path", dest="lora_path", default=default_lora_path)
    parser.add_argument("--lora-weight-name", dest="lora_weight_name", default=default_lora_weight_name)
    parser.add_argument("--lora-adapter-name", dest="lora_adapter_name", default=default_lora_adapter_name)
    parser.add_argument("--torch-dtype", dest="torch_dtype", default=default_torch_dtype)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Main image generation pipeline."""
    args = parse_args()
    ensure_hf_login()

    # Detect device and determine dtype
    device = get_device()
    dtype = get_dtype(args.model, device, args.torch_dtype)

    # Build appropriate pipeline
    use_img2img = args.init_image is not None

    pipe: AutoPipelineForText2Image | StableDiffusionPipeline | StableDiffusionImg2ImgPipeline
    if is_flux_model(args.model):
        if dtype == torch.float16:
            print("✓ FLUX model detected: using bfloat16 for better quality")
            dtype = torch.bfloat16

        pipe = build_flux_pipeline(
            args.model,
            device,
            dtype,
            lora_path=args.lora_path,
            lora_weight_name=args.lora_weight_name,
            lora_adapter_name=args.lora_adapter_name,
        )
    else:
        pipe = build_stable_diffusion_pipeline(
            args.model,
            device,
            dtype,
            use_img2img=use_img2img,
            lora_path=args.lora_path,
            lora_weight_name=args.lora_weight_name,
            lora_adapter_name=args.lora_adapter_name,
        )

    pipe.enable_attention_slicing()

    # Prepare generator for seeding
    generator: torch.Generator | None = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    # Generate image
    if use_img2img:
        init_image = load_init_image(args.init_image, args.width, args.height)
        image = pipe(  # type: ignore
            args.prompt,
            image=init_image,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            negative_prompt=args.negative_prompt,
            generator=generator,
        ).images[0]  # type: ignore
    else:
        image = pipe(  # type: ignore
            args.prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            negative_prompt=args.negative_prompt,
            generator=generator,
        ).images[0]  # type: ignore

    # Save output
    output_path = get_next_output_path(OUTPUT_DIR)
    image.save(output_path)  # type: ignore
    print(f"✓ Image saved: {output_path}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
