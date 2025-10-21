"""Tests for GPU configuration module."""

import pytest
from src.gpu_config import GPUConfig, get_optimal_device


class TestGPUConfig:
    """Test cases for GPUConfig class."""
    
    def test_initialization(self):
        """Test GPU configuration initialization."""
        config = GPUConfig()
        assert config.device in ["cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"]
        assert isinstance(config.gpu_id, int)
    
    def test_device_info(self):
        """Test device information retrieval."""
        config = GPUConfig()
        info = config.get_device_info()
        
        assert "device" in info
        assert "cuda_available" in info
        assert "pytorch_installed" in info
        assert isinstance(info["pytorch_installed"], bool)
    
    def test_cpu_fallback(self):
        """Test CPU fallback when GPU is not requested."""
        config = GPUConfig(device="cpu")
        assert config.device == "cpu"
    
    def test_clear_cache(self):
        """Test GPU cache clearing (should not raise error)."""
        config = GPUConfig()
        config.clear_cache()  # Should work regardless of GPU availability
    
    def test_optimal_device(self):
        """Test optimal device selection."""
        device = get_optimal_device(prefer_gpu=True)
        assert device in ["cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"]
        
        device_cpu = get_optimal_device(prefer_gpu=False)
        assert device_cpu == "cpu"
    
    def test_memory_fraction(self):
        """Test memory fraction setting (should not raise error)."""
        config = GPUConfig()
        config.set_memory_fraction(0.5)  # Should work regardless of GPU availability


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
