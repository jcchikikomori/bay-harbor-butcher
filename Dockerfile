FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

RUN pip install --upgrade pip

RUN pip install \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    xformers \
    torchvision==0.17.0

WORKDIR /app
