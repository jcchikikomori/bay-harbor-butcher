"""Unit tests for pipelines.stable_diffusion module."""

from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch

from pipelines.stable_diffusion import build_stable_diffusion_pipeline


class TestBuildStableDiffusionPipeline(TestCase):
    """Test Stable Diffusion pipeline building."""

    @patch("pipelines.stable_diffusion.StableDiffusionPipeline")
    def test_build_text2img_pipeline(self, mock_sd_pipeline) -> None:
        """Test building text-to-image pipeline."""
        mock_pipe = MagicMock()
        mock_sd_pipeline.from_pretrained.return_value = mock_pipe

        build_stable_diffusion_pipeline(
            "runwayml/stable-diffusion-v1-5",
            "cuda",
            torch.float16,
            use_img2img=False,
        )

        # Verify StableDiffusionPipeline was called
        mock_sd_pipeline.from_pretrained.assert_called_once()
        call_args = mock_sd_pipeline.from_pretrained.call_args
        assert call_args[0][0] == "runwayml/stable-diffusion-v1-5"
        assert call_args[1]["torch_dtype"] == torch.float16

    @patch("pipelines.stable_diffusion.StableDiffusionImg2ImgPipeline")
    def test_build_img2img_pipeline(self, mock_img2img_pipeline) -> None:
        """Test building image-to-image pipeline."""
        mock_pipe = MagicMock()
        mock_img2img_pipeline.from_pretrained.return_value = mock_pipe

        build_stable_diffusion_pipeline(
            "runwayml/stable-diffusion-v1-5",
            "cuda",
            torch.float16,
            use_img2img=True,
        )

        # Verify Img2ImgPipeline was called
        mock_img2img_pipeline.from_pretrained.assert_called_once()

    @patch("pipelines.stable_diffusion.StableDiffusionPipeline")
    def test_build_sd_pipeline_device_placement(self, mock_sd_pipeline) -> None:
        """Test device placement for SD pipeline."""
        mock_pipe = MagicMock()
        mock_sd_pipeline.from_pretrained.return_value = mock_pipe

        build_stable_diffusion_pipeline(
            "runwayml/stable-diffusion-v1-5",
            "cuda",
            torch.float16,
        )

        # Verify to() was called with device
        mock_pipe.to.assert_called_once_with("cuda")

    @patch("pipelines.stable_diffusion.StableDiffusionPipeline")
    def test_build_sd_pipeline_safety_checker_disabled(self, mock_sd_pipeline) -> None:
        """Test that safety_checker is disabled for SD."""
        mock_pipe = MagicMock()
        mock_sd_pipeline.from_pretrained.return_value = mock_pipe
        mock_pipe.to.return_value = mock_pipe  # Ensure to() returns the pipe

        build_stable_diffusion_pipeline(
            "runwayml/stable-diffusion-v1-5",
            "cuda",
            torch.float16,
        )

        # Verify safety_checker was set to None
        assert mock_pipe.safety_checker is None

    @patch("pipelines.stable_diffusion.load_lora_weights")
    @patch("pipelines.stable_diffusion.StableDiffusionPipeline")
    def test_build_sd_pipeline_with_lora(self, mock_sd_pipeline, mock_load_lora) -> None:
        """Test SD pipeline building with LoRA."""
        mock_pipe = MagicMock()
        mock_sd_pipeline.from_pretrained.return_value = mock_pipe
        mock_pipe.to.return_value = mock_pipe  # Ensure to() returns the pipe

        build_stable_diffusion_pipeline(
            "runwayml/stable-diffusion-v1-5",
            "cuda",
            torch.float16,
            lora_path="./loras/style",
            lora_weight_name="weights.safetensors",
            lora_adapter_name="my_adapter",
        )

        # Verify load_lora_weights was called with correct args
        mock_load_lora.assert_called_once()
        call_args = mock_load_lora.call_args[0]
        assert call_args[1] == "./loras/style"
        assert call_args[2] == "weights.safetensors"
        assert call_args[3] == "my_adapter"

    @patch("pipelines.stable_diffusion.StableDiffusionPipeline")
    def test_build_sd_pipeline_cpu(self, mock_sd_pipeline) -> None:
        """Test SD pipeline on CPU device."""
        mock_pipe = MagicMock()
        mock_sd_pipeline.from_pretrained.return_value = mock_pipe

        build_stable_diffusion_pipeline(
            "runwayml/stable-diffusion-v1-5",
            "cpu",
            torch.float32,
        )

        # Verify device placement
        mock_pipe.to.assert_called_once_with("cpu")

    @patch("pipelines.stable_diffusion.StableDiffusionPipeline")
    def test_build_sd_pipeline_float32(self, mock_sd_pipeline) -> None:
        """Test SD pipeline with float32 dtype."""
        mock_pipe = MagicMock()
        mock_sd_pipeline.from_pretrained.return_value = mock_pipe

        build_stable_diffusion_pipeline(
            "runwayml/stable-diffusion-v1-5",
            "cpu",
            torch.float32,
        )

        # Verify dtype was used
        call_args = mock_sd_pipeline.from_pretrained.call_args
        assert call_args[1]["torch_dtype"] == torch.float32
