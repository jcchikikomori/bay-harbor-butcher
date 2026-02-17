"""LoRA weight loading utilities."""

import os
import warnings


def load_lora_weights(
    pipe,
    lora_path: str | None,
    lora_weight_name: str | None = None,
    lora_adapter_name: str | None = None,
) -> None:
    """
    Load LoRA weights into pipeline.

    Args:
        pipe: Diffusion pipeline
        lora_path: Local path or HuggingFace model ID
        lora_weight_name: Specific weight file name
        lora_adapter_name: Adapter name for multi-LoRA
    """
    if not lora_path:
        return

    try:
        # Check if PEFT is available for LoRA loading
        import peft  # noqa: F401

        is_local = os.path.exists(lora_path)
        pipe.load_lora_weights(
            lora_path,
            weight_name=lora_weight_name,
            adapter_name=lora_adapter_name,
        )
        source = "local" if is_local else "HuggingFace"
        print(f"✓ Loaded LoRA from {source}: {lora_path}")
    except ImportError:
        warnings.warn(
            f"PEFT library required for LoRA. Install with: pip install peft\nSkipping LoRA: {lora_path}",
            stacklevel=2,
        )
    except Exception as e:
        warnings.warn(f"Failed to load LoRA '{lora_path}': {e}", stacklevel=2)
