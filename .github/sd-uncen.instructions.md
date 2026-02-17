---
description: This file provides instructions for setting up and running the Stable Diffusion Uncensored project
applyTo: **/*
---

# Project Instructions

## Run Modes

- Use `python app.py --help` to view all CLI options.
- Default prompt and other settings are read from `.env` via `python-dotenv`.
- CLI args override `.env` values when provided.

## Virtual Environment

- Create with `pyenv virtualenv sd`, or `pyenv virtualenv 3.12.0 sd-uncen`.
- Activate with `pyenv local sd`.

## Environment Variables (.env)

- `SD_PROMPT`
- `SD_NEGATIVE_PROMPT`
- `SD_IMAGE_HEIGHT`
- `SD_IMAGE_WIDTH`
- `SD_MODEL`
- `SD_STRENGTH`
- `SD_GUIDANCE_SCALE`
- `SD_NUM_INFERENCE_STEPS`

## Generation

- Text-to-image: run without `--init-image`.
- Image-to-image: pass `--init-image path/to.jpg`.
- Output is always saved to `output/` and auto-increments (`output_1.png`, `output_2.png`, ...).

## Device Selection

- CUDA is preferred when available.
- ROCm is supported (uses `torch.version.hip` and device `cuda`).
- Falls back to CPU with a warning.

## Tests

- Run: `python -m unittest`.
