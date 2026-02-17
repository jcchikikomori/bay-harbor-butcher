"""Unit tests for utils.device module."""

import warnings
from unittest import TestCase
from unittest.mock import patch

import torch

from utils.device import _is_flux_model, _parse_torch_dtype, get_device, get_dtype


class TestIsFluxModel(TestCase):
    """Test FLUX model detection."""

    def test_flux_dev_detected(self) -> None:
        """Test detection of FLUX.1-dev model."""
        assert _is_flux_model("black-forest-labs/FLUX.1-dev") is True

    def test_flux_schnell_detected(self) -> None:
        """Test detection of FLUX.1-schnell model."""
        assert _is_flux_model("black-forest-labs/FLUX.1-schnell") is True

    def test_flux_lowercase_detected(self) -> None:
        """Test detection with lowercase."""
        assert _is_flux_model("some-org/flux_model") is True

    def test_stable_diffusion_not_detected(self) -> None:
        """Test that Stable Diffusion models are not detected as FLUX."""
        assert _is_flux_model("runwayml/stable-diffusion-v1-5") is False

    def test_custom_model_not_detected(self) -> None:
        """Test that custom models are not detected as FLUX."""
        assert _is_flux_model("username/custom-model") is False


class TestParseTorchDtype(TestCase):
    """Test torch dtype string parsing."""

    def test_parse_float16(self) -> None:
        """Test parsing float16."""
        assert _parse_torch_dtype("float16") == torch.float16

    def test_parse_float32(self) -> None:
        """Test parsing float32."""
        assert _parse_torch_dtype("float32") == torch.float32

    def test_parse_bfloat16(self) -> None:
        """Test parsing bfloat16."""
        assert _parse_torch_dtype("bfloat16") == torch.bfloat16

    def test_parse_half(self) -> None:
        """Test parsing 'half' alias."""
        assert _parse_torch_dtype("half") == torch.float16

    def test_parse_full(self) -> None:
        """Test parsing 'full' alias."""
        assert _parse_torch_dtype("full") == torch.float32

    def test_parse_case_insensitive(self) -> None:
        """Test case-insensitive parsing."""
        assert _parse_torch_dtype("FLOAT16") == torch.float16
        assert _parse_torch_dtype("BFloat16") == torch.bfloat16

    def test_parse_none_returns_none(self) -> None:
        """Test parsing None returns None."""
        assert _parse_torch_dtype(None) is None

    def test_parse_empty_returns_none(self) -> None:
        """Test parsing empty string returns None."""
        assert _parse_torch_dtype("") is None

    def test_parse_invalid_warns(self) -> None:
        """Test parsing invalid dtype warns and returns None."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _parse_torch_dtype("invalid_dtype")
            assert result is None
            assert len(w) == 1
            assert "Unknown dtype" in str(w[0].message)


class TestGetDevice(TestCase):
    """Test device detection."""

    @patch("torch.cuda.is_available")
    def test_cuda_available(self, mock_cuda_available) -> None:
        """Test CUDA device selection when available."""
        mock_cuda_available.return_value = True
        assert get_device() == "cuda"

    @patch("torch.version.hip", None)
    @patch("torch.cuda.is_available")
    def test_rocm_fallback(self, mock_cuda_available) -> None:
        """Test ROCm fallback when CUDA unavailable."""
        mock_cuda_available.return_value = False
        # Note: torch.version.hip would need to be non-None for ROCm detection
        # For now just test CPU fallback
        with patch("torch.version.hip", None):
            assert get_device() == "cpu"

    @patch("torch.version.hip", None)
    @patch("torch.cuda.is_available")
    def test_cpu_fallback(self, mock_cuda_available) -> None:
        """Test CPU fallback when no GPU available."""
        mock_cuda_available.return_value = False
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            device = get_device()
            assert device == "cpu"
            assert len(w) == 1
            assert "falling back to CPU" in str(w[0].message)


class TestGetDtype(TestCase):
    """Test dtype selection."""

    @patch("utils.device._parse_torch_dtype")
    def test_explicit_override(self, mock_parse) -> None:
        """Test explicit dtype override."""
        mock_parse.return_value = torch.float32
        result = get_dtype("any-model", "cuda", "float32")
        assert result == torch.float32

    def test_flux_bfloat16_on_cuda(self) -> None:
        """Test FLUX models use bfloat16 on CUDA."""
        result = get_dtype("black-forest-labs/FLUX.1-dev", "cuda", None)
        assert result == torch.bfloat16

    def test_stable_diffusion_float16_on_cuda(self) -> None:
        """Test Stable Diffusion uses float16 on CUDA."""
        result = get_dtype("runwayml/stable-diffusion-v1-5", "cuda", None)
        assert result == torch.float16

    def test_stable_diffusion_float32_on_cpu(self) -> None:
        """Test Stable Diffusion uses float32 on CPU."""
        result = get_dtype("runwayml/stable-diffusion-v1-5", "cpu", None)
        assert result == torch.float32

    def test_flux_float32_on_cpu(self) -> None:
        """Test FLUX uses float32 on CPU (fallback)."""
        result = get_dtype("black-forest-labs/FLUX.1-dev", "cpu", None)
        assert result == torch.float32
