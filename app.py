import argparse
import os
import warnings
from typing import Optional

import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from dotenv import load_dotenv
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


def load_init_image(image_path: str, width: int, height: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    return image.resize((width, height))


def build_pipeline(model_id: str, use_img2img: bool, device: str, dtype: torch.dtype):
    if use_img2img:
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
    return pipe.to(device)


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
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.version.hip is not None:
        device = "cuda"
    else:
        warnings.warn("CUDA/ROCm not available; falling back to CPU.")
        device = "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    use_img2img = args.init_image is not None
    pipe = build_pipeline(args.model, use_img2img, device, dtype)
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
