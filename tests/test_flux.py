"""Unit tests for pipelines.flux module."""

from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch

from pipelines.flux import build_flux_pipeline, is_flux_model


class TestIsFluxModel(TestCase):
    """Test FLUX model detection in pipelines.flux."""

    def test_flux_dev_detected(self) -> None:
        """Test detection of FLUX.1-dev model."""
        assert is_flux_model("black-forest-labs/FLUX.1-dev") is True

    def test_flux_schnell_detected(self) -> None:
        """Test detection of FLUX.1-schnell model."""
        assert is_flux_model("black-forest-labs/FLUX.1-schnell") is True

    def test_stable_diffusion_not_detected(self) -> None:
        """Test that Stable Diffusion models are not detected as FLUX."""
        assert is_flux_model("runwayml/stable-diffusion-v1-5") is False


class TestBuildFluxPipeline(TestCase):
    """Test FLUX pipeline building."""

    @patch("pipelines.flux.AutoPipelineForText2Image")
    def test_build_flux_pipeline_basic(self, mock_autopipeline) -> None:
        """Test basic FLUX pipeline building."""
        mock_pipe = MagicMock()
        mock_autopipeline.from_pretrained.return_value = mock_pipe

        with patch("builtins.print") as mock_print:
            build_flux_pipeline(
                "black-forest-labs/FLUX.1-dev",
                "cuda",
                torch.bfloat16,
            )

            # Verify AutoPipeline.from_pretrained was called
            mock_autopipeline.from_pretrained.assert_called_once()
            call_args = mock_autopipeline.from_pretrained.call_args
            assert call_args[0][0] == "black-forest-labs/FLUX.1-dev"
            assert call_args[1]["torch_dtype"] == torch.bfloat16

            # Verify detection message printed
            mock_print.assert_called()
            assert "Detected FLUX model" in str(mock_print.call_args)

    @patch("pipelines.flux.AutoPipelineForText2Image")
    def test_build_flux_pipeline_device_placement(self, mock_autopipeline) -> None:
        """Test that pipeline is moved to correct device."""
        mock_pipe = MagicMock()
        mock_autopipeline.from_pretrained.return_value = mock_pipe

        build_flux_pipeline(
            "black-forest-labs/FLUX.1-dev",
            "cuda",
            torch.float16,
        )

        # Verify to() was called with device
        mock_pipe.to.assert_called_once_with("cuda")

    @patch("pipelines.flux.AutoPipelineForText2Image")
    def test_build_flux_pipeline_safety_checker_removed(self, mock_autopipeline) -> None:
        """Test that safety_checker is disabled for FLUX."""
        mock_pipe = MagicMock()
        mock_autopipeline.from_pretrained.return_value = mock_pipe
        mock_pipe.to.return_value = mock_pipe  # Ensure to() returns the pipe for chaining

        build_flux_pipeline(
            "black-forest-labs/FLUX.1-dev",
            "cuda",
            torch.float16,
        )

        # Verify safety_checker was set to None
        assert mock_pipe.safety_checker is None

    @patch("pipelines.flux.load_lora_weights")
    @patch("pipelines.flux.AutoPipelineForText2Image")
    def test_build_flux_pipeline_with_lora(self, mock_autopipeline, mock_load_lora) -> None:
        """Test FLUX pipeline building with LoRA."""
        mock_pipe = MagicMock()
        mock_autopipeline.from_pretrained.return_value = mock_pipe
        mock_pipe.to.return_value = mock_pipe  # Ensure to() returns the pipe

        build_flux_pipeline(
            "black-forest-labs/FLUX.1-dev",
            "cuda",
            torch.float16,
            lora_path="./loras/style",
            lora_weight_name="weights.safetensors",
        )

        # Verify load_lora_weights was called with the returned pipe
        mock_load_lora.assert_called_once()
        call_args = mock_load_lora.call_args[0]
        assert call_args[1] == "./loras/style"
        assert call_args[2] == "weights.safetensors"

    @patch("pipelines.flux.AutoPipelineForText2Image")
    def test_build_flux_pipeline_cpu(self, mock_autopipeline) -> None:
        """Test FLUX pipeline on CPU device."""
        mock_pipe = MagicMock()
        mock_autopipeline.from_pretrained.return_value = mock_pipe

        build_flux_pipeline(
            "black-forest-labs/FLUX.1-dev",
            "cpu",
            torch.float32,
        )

        # Verify device placement
        mock_pipe.to.assert_called_once_with("cpu")
