import argparse
import os
import warnings
from typing import Optional

import torch
from diffusers import AutoPipelineForText2Image, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from dotenv import load_dotenv
from huggingface_hub import login
from PIL import Image

DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "output"


def get_next_output_path(output_dir: str, prefix: str = "output") -> str:
    os.makedirs(output_dir, exist_ok=True)
    index = 1
    while True:
        output_path = os.path.join(output_dir, f"{prefix}_{index}.png")
        if not os.path.exists(output_path):
            return output_path
        index += 1


def ensure_hf_login() -> None:
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
                "3. Add to .env: HF_TOKEN=\"hf_your_token_here\"\n"
                "4. Restart this script\n"
            )
            print("=" * 60 + "\n")
            warnings.warn(f"HuggingFace login skipped: {e}")


def load_init_image(image_path: str, width: int, height: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    return image.resize((width, height))


def build_pipeline(
    model_id: str,
    use_img2img: bool,
    device: str,
    dtype: torch.dtype,
    use_auto_pipeline: bool = False,
    lora_path: Optional[str] = None,
    lora_weight_name: Optional[str] = None,
    lora_adapter_name: Optional[str] = None,
):
    if use_auto_pipeline:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )
    elif use_img2img:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
        )
    
    pipe = pipe.to(device)
    
    if lora_path:
        # Support both local paths and HuggingFace model IDs
        is_local = os.path.exists(lora_path)
        try:
            pipe.load_lora_weights(
                lora_path,
                weight_name=lora_weight_name,
                adapter_name=lora_adapter_name,
            )
            source = "local" if is_local else "HuggingFace"
            print(f"✓ Loaded LoRA from {source}: {lora_path}")
        except Exception as e:
            warnings.warn(f"Failed to load LoRA '{lora_path}': {e}")
    
    return pipe


def get_env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None else default


def get_env_optional_str(name: str) -> Optional[str]:
    value = os.getenv(name)
    return value if value else None


def get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        warnings.warn(f"Invalid int for {name}; using default.")
        return default


def get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        warnings.warn(f"Invalid float for {name}; using default.")
        return default


def get_env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    if value in ("false", "0", "no", "off"):
        return False
    return default


def parse_torch_dtype(dtype_str: Optional[str]) -> Optional[torch.dtype]:
    """Parse torch dtype from string (e.g., 'float16', 'bfloat16', 'float32')."""
    if not dtype_str:
        return None
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "half": torch.float16,
        "full": torch.float32,
    }
    dtype = dtype_map.get(dtype_str.lower())
    if dtype is None:
        warnings.warn(f"Unknown dtype '{dtype_str}'; using auto-detection.")
    return dtype


def parse_args() -> argparse.Namespace:
    load_dotenv()
    default_prompt = get_env_str("SD_PROMPT", "a photo of an astronaut riding a horse on mars")
    default_negative_prompt = get_env_optional_str("SD_NEGATIVE_PROMPT")
    default_height = get_env_int("SD_IMAGE_HEIGHT", 512)
    default_width = get_env_int("SD_IMAGE_WIDTH", 512)
    default_model = get_env_str("SD_MODEL", DEFAULT_MODEL_ID)
    default_strength = get_env_float("SD_STRENGTH", 0.6)
    default_guidance_scale = get_env_float("SD_GUIDANCE_SCALE", 7.5)
    default_num_steps = get_env_int("SD_NUM_INFERENCE_STEPS", 50)
    default_use_auto_pipeline = get_env_bool("SD_USE_AUTO_PIPELINE", False)
    default_lora_path = get_env_optional_str("SD_LORA_PATH")
    default_lora_weight_name = get_env_optional_str("SD_LORA_WEIGHT_NAME")
    default_lora_adapter_name = get_env_optional_str("SD_LORA_ADAPTER_NAME")
    default_torch_dtype = get_env_optional_str("SD_TORCH_DTYPE")
    
    parser = argparse.ArgumentParser(description="Stable Diffusion image generation")
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
    parser.add_argument("--use-auto-pipeline", dest="use_auto_pipeline", action="store_true", default=default_use_auto_pipeline)
    parser.add_argument("--lora-path", dest="lora_path", default=default_lora_path)
    parser.add_argument("--lora-weight-name", dest="lora_weight_name", default=default_lora_weight_name)
    parser.add_argument("--lora-adapter-name", dest="lora_adapter_name", default=default_lora_adapter_name)
    parser.add_argument("--torch-dtype", dest="torch_dtype", default=default_torch_dtype)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_hf_login()
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.version.hip is not None:
        device = "cuda"
    else:
        warnings.warn("CUDA/ROCm not available; falling back to CPU.")
        device = "cpu"
    
    # Determine dtype: explicit > env/cli > auto-detect
    if args.torch_dtype:
        dtype = parse_torch_dtype(args.torch_dtype)
        if dtype is None:
            dtype = torch.float16 if device == "cuda" else torch.float32
    else:
        dtype = torch.float16 if device == "cuda" else torch.float32

    use_img2img = args.init_image is not None
    pipe = build_pipeline(
        args.model,
        use_img2img,
        device,
        dtype,
        use_auto_pipeline=args.use_auto_pipeline,
        lora_path=args.lora_path,
        lora_weight_name=args.lora_weight_name,
        lora_adapter_name=args.lora_adapter_name,
    )
    pipe.enable_attention_slicing()

    generator: Optional[torch.Generator] = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    if use_img2img:
        init_image = load_init_image(args.init_image, args.width, args.height)
        image = pipe(
            args.prompt,
            image=init_image,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            negative_prompt=args.negative_prompt,
            generator=generator,
        ).images[0]
    else:
        image = pipe(
            args.prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            negative_prompt=args.negative_prompt,
            generator=generator,
        ).images[0]

    output_path = get_next_output_path(OUTPUT_DIR)
    image.save(output_path)


if __name__ == "__main__":
    main()
