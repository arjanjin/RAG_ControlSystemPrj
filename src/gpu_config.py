"""GPU configuration and detection module for RAG Control System."""

import os
import logging
from typing import Optional, Dict, Any

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)


class GPUConfig:
    """Manages GPU configuration and device selection for the RAG system."""
    
    def __init__(self, device: Optional[str] = None, gpu_id: int = 0):
        """
        Initialize GPU configuration.
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            gpu_id: GPU device ID to use if multiple GPUs are available
        """
        self.gpu_id = gpu_id
        self._device = self._detect_device(device)
        self._log_device_info()
    
    def _detect_device(self, device: Optional[str] = None) -> str:
        """
        Detect and configure the appropriate device.
        
        Args:
            device: Requested device or None for auto-detection
            
        Returns:
            Device string ('cuda:N' or 'cpu')
        """
        if torch is None:
            logger.warning("PyTorch not installed. Falling back to CPU.")
            return "cpu"
        
        if device is not None:
            if device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning(f"CUDA requested but not available. Falling back to CPU.")
                return "cpu"
            return device
        
        # Auto-detect
        if torch.cuda.is_available():
            if self.gpu_id >= torch.cuda.device_count():
                logger.warning(
                    f"GPU {self.gpu_id} not available. "
                    f"Only {torch.cuda.device_count()} GPU(s) detected. Using GPU 0."
                )
                self.gpu_id = 0
            return f"cuda:{self.gpu_id}"
        
        logger.info("No GPU detected. Using CPU.")
        return "cpu"
    
    def _log_device_info(self):
        """Log information about the selected device."""
        if torch is None:
            return
            
        logger.info(f"Using device: {self._device}")
        
        if self.is_cuda_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"Available GPUs: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
            
            if self._device.startswith("cuda"):
                current_gpu = int(self._device.split(":")[-1])
                logger.info(f"Selected GPU {current_gpu}: {torch.cuda.get_device_name(current_gpu)}")
    
    @property
    def device(self) -> str:
        """Get the configured device string."""
        return self._device
    
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return torch is not None and torch.cuda.is_available()
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the current device.
        
        Returns:
            Dictionary with device information
        """
        info = {
            "device": self._device,
            "cuda_available": self.is_cuda_available(),
            "pytorch_installed": torch is not None,
        }
        
        if torch is not None and torch.cuda.is_available():
            info.update({
                "gpu_count": torch.cuda.device_count(),
                "gpu_id": self.gpu_id,
                "gpu_name": torch.cuda.get_device_name(self.gpu_id),
                "cuda_version": torch.version.cuda,
                "pytorch_version": torch.__version__,
            })
            
            # Get memory info for current GPU
            if self._device.startswith("cuda"):
                gpu_id = int(self._device.split(":")[-1])
                props = torch.cuda.get_device_properties(gpu_id)
                info["gpu_memory_total_gb"] = props.total_memory / 1024**3
                info["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated(gpu_id) / 1024**3
                info["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved(gpu_id) / 1024**3
        
        return info
    
    def clear_cache(self):
        """Clear GPU cache if available."""
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
    
    def set_memory_fraction(self, fraction: float = 0.8):
        """
        Set the fraction of GPU memory to use.
        
        Args:
            fraction: Fraction of GPU memory to allocate (0.0 to 1.0)
        """
        if torch is not None and torch.cuda.is_available():
            if 0.0 < fraction <= 1.0:
                torch.cuda.set_per_process_memory_fraction(fraction, device=self.gpu_id)
                logger.info(f"Set GPU memory fraction to {fraction}")
            else:
                logger.warning(f"Invalid memory fraction {fraction}. Must be between 0 and 1.")


def get_optimal_device(prefer_gpu: bool = True) -> str:
    """
    Get the optimal device for computation.
    
    Args:
        prefer_gpu: Whether to prefer GPU if available
        
    Returns:
        Device string ('cuda:0' or 'cpu')
    """
    config = GPUConfig()
    if not prefer_gpu:
        return "cpu"
    return config.device


if __name__ == "__main__":
    # Test GPU configuration
    logging.basicConfig(level=logging.INFO)
    
    config = GPUConfig()
    print("\n=== GPU Configuration ===")
    for key, value in config.get_device_info().items():
        print(f"{key}: {value}")
