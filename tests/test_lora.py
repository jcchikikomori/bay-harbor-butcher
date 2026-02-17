"""Unit tests for utils.lora module."""

import warnings
from unittest import TestCase
from unittest.mock import MagicMock, patch

from utils.lora import load_lora_weights


class TestLoadLoraWeights(TestCase):
    """Test LoRA weight loading."""

    def test_load_lora_skip_when_none(self) -> None:
        """Test that None lora_path is skipped."""
        mock_pipe = MagicMock()
        load_lora_weights(mock_pipe, None)
        mock_pipe.load_lora_weights.assert_not_called()

    def test_load_lora_skip_when_empty(self) -> None:
        """Test that empty string lora_path is skipped."""
        mock_pipe = MagicMock()
        load_lora_weights(mock_pipe, "")
        mock_pipe.load_lora_weights.assert_not_called()

    @patch("os.path.exists")
    def test_load_lora_peft_missing(self, mock_exists) -> None:
        """Test graceful handling when PEFT is missing."""
        mock_exists.return_value = True
        mock_pipe = MagicMock()

        # Simulate ImportError when trying to import peft
        def import_side_effect(name, *args, **kwargs):
            if name == "peft":
                raise ImportError("No module named 'peft'")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_side_effect):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                load_lora_weights(mock_pipe, "./loras/test")
                # Should warn about missing PEFT
                assert len(w) >= 1

    @patch("os.path.exists")
    def test_load_lora_local_path(self, mock_exists) -> None:
        """Test loading LoRA from local path."""
        mock_exists.return_value = True
        mock_pipe = MagicMock()

        with patch("builtins.__import__", return_value=MagicMock()):
            load_lora_weights(
                mock_pipe,
                "./loras/my_style",
                lora_weight_name="weights.safetensors",
            )
            # Verify load_lora_weights was called on pipe
            mock_pipe.load_lora_weights.assert_called_once()

    @patch("os.path.exists")
    def test_load_lora_hf_path(self, mock_exists) -> None:
        """Test loading LoRA from HuggingFace model ID."""
        mock_exists.return_value = False
        mock_pipe = MagicMock()

        with patch("builtins.__import__", return_value=MagicMock()):
            load_lora_weights(
                mock_pipe,
                "ostris/OpenXL-LoRA-library",
                lora_weight_name="weights.safetensors",
            )
            # Verify load_lora_weights was called on pipe
            mock_pipe.load_lora_weights.assert_called_once()

    def test_load_lora_exception_handling(self) -> None:
        """Test handling of LoRA loading exceptions."""
        mock_pipe = MagicMock()
        mock_pipe.load_lora_weights.side_effect = RuntimeError("Failed to load")

        with patch("os.path.exists", return_value=True):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                load_lora_weights(mock_pipe, "./loras/test")
                # Should warn about failure
                assert len(w) >= 1
                assert "Failed to load LoRA" in str(w[0].message)
