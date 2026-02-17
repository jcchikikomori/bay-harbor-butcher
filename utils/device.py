"""Device detection and configuration."""

import warnings

import torch


def get_device() -> str:
    """Detect available device: CUDA, ROCm, or CPU."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.version.hip is not None:
        return "cuda"  # ROCm uses cuda device type
    else:
        warnings.warn(
            "CUDA/ROCm not available; falling back to CPU.",
            stacklevel=2,
        )
        return "cpu"


def _is_flux_model(model_id: str) -> bool:
    """Check if model is a FLUX variant (local check, no imports)."""
    flux_keywords = ["flux", "flux.1"]
    return any(kw in model_id.lower() for kw in flux_keywords)


def get_dtype(
    model_id: str,
    device: str,
    dtype_override: str | None = None,
) -> torch.dtype:
    """
    Determine optimal dtype based on model and device.

    Args:
        model_id: HuggingFace model ID
        device: Target device ('cuda' or 'cpu')
        dtype_override: User-specified dtype string

    Returns:
        torch.dtype: Optimal dtype
    """
    if dtype_override:
        dtype = _parse_torch_dtype(dtype_override)
        if dtype is not None:
            return dtype

    # Auto-select based on model and device
    if _is_flux_model(model_id) and device == "cuda":
        return torch.bfloat16  # FLUX prefers bfloat16 on GPU

    # Stable Diffusion defaults
    return torch.float16 if device == "cuda" else torch.float32


def _parse_torch_dtype(dtype_str: str | None) -> torch.dtype | None:
    """Parse torch dtype from string."""
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
        warnings.warn(
            f"Unknown dtype '{dtype_str}'; using auto-detection.",
            stacklevel=2,
        )
    return dtype
