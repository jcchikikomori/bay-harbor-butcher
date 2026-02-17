import argparse
import os
import warnings
from typing import Optional

import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from PIL import Image

MODEL_ID = "runwayml/stable-diffusion-v1-5"
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


def build_pipeline(use_img2img: bool, device: str, dtype: torch.dtype):
    if use_img2img:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            safety_checker=None,
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            safety_checker=None,
        )
    return pipe.to(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stable Diffusion image generation")
    parser.add_argument("--prompt", default="a photo of an astronaut riding a horse on mars")
    parser.add_argument("--init-image", dest="init_image", default=None)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--strength", type=float, default=0.6)
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
    pipe = build_pipeline(use_img2img, device, dtype)
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
            generator=generator,
        ).images[0]
    else:
        image = pipe(
            args.prompt,
            height=args.height,
            width=args.width,
            generator=generator,
        ).images[0]

    output_path = get_next_output_path(OUTPUT_DIR)
    image.save(output_path)


if __name__ == "__main__":
    main()
