# Quick Start Guide - GPU-Accelerated RAG System

This guide will help you get started with the GPU-accelerated RAG Control System in just a few minutes.

## Prerequisites

Before you begin, ensure you have:
- Python 3.8 or higher
- NVIDIA GPU with CUDA support (optional, will fall back to CPU)
- CUDA Toolkit 11.7+ installed (for GPU support)

## Quick Setup (5 minutes)

### 1. Clone and Navigate

```bash
git clone https://github.com/arjanjin/RAG_ControlSystemPrj.git
cd RAG_ControlSystemPrj
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: This installs GPU-enabled packages by default. If you don't have a GPU or CUDA, the system will automatically fall back to CPU.

### 4. Verify GPU Setup

```bash
python src/gpu_config.py
```

Expected output (with GPU):
```
INFO:__main__:Using device: cuda:0
INFO:__main__:Available GPUs: 1
INFO:__main__:  GPU 0: NVIDIA GeForce RTX 3080 (10.00 GB)
INFO:__main__:Selected GPU 0: NVIDIA GeForce RTX 3080

=== GPU Configuration ===
device: cuda:0
cuda_available: True
pytorch_installed: True
gpu_count: 1
gpu_id: 0
gpu_name: NVIDIA GeForce RTX 3080
...
```

Expected output (without GPU):
```
WARNING:__main__:No GPU detected. Using CPU.

=== GPU Configuration ===
device: cpu
cuda_available: False
pytorch_installed: True
```

### 5. Run the Example

```bash
python example.py
```

This will:
1. Check GPU configuration
2. Initialize the RAG system
3. Add sample documents
4. Perform sample searches
5. Display system information

## Basic Usage

### Simple Document Search

```python
from src.rag_system import RAGSystem

# Initialize system (GPU auto-detected)
rag = RAGSystem()

# Add documents
documents = [
    "Document 1: Information about examinations",
    "Document 2: GPU acceleration benefits",
    "Document 3: Assessment policies"
]
rag.add_documents(documents)

# Search
results = rag.search("Tell me about examinations", top_k=2)
for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Text: {result['text']}\n")
```

### Force CPU Usage

```python
# Use CPU even if GPU is available
rag = RAGSystem(device="cpu")
```

### Select Specific GPU

```python
# Use GPU 1 (if you have multiple GPUs)
rag = RAGSystem(device="cuda:1", gpu_id=1)
```

### Adjust GPU Memory

```python
# Use 50% of GPU memory
rag = RAGSystem(memory_fraction=0.5)
```

## Checking System Status

```python
from src.gpu_config import GPUConfig

# Get GPU information
config = GPUConfig()
info = config.get_device_info()

print(f"Device: {info['device']}")
print(f"GPU Available: {info['cuda_available']}")
if info['cuda_available']:
    print(f"GPU Name: {info['gpu_name']}")
    print(f"GPU Memory: {info['gpu_memory_total_gb']:.2f} GB")
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_gpu_config.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Common Issues

### Issue: "CUDA not available"

**Solution**: 
1. Verify CUDA installation: `nvidia-smi`
2. Check PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
3. If needed, reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### Issue: "Out of memory" error

**Solutions**:
1. Reduce memory fraction: `RAGSystem(memory_fraction=0.5)`
2. Clear GPU cache: `rag.clear_cache()`
3. Use a smaller embedding model
4. Process documents in smaller batches

### Issue: Slow performance

**Checks**:
1. Verify GPU is being used: `nvidia-smi`
2. Check device in use: `print(rag.gpu_config.device)`
3. Ensure CUDA drivers are up to date

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [config.yaml.example](config.yaml.example) for advanced configuration
- Explore the source code in `src/` directory
- Run the example script: `python example.py`

## Need Help?

- Check the [README.md](README.md) for troubleshooting
- Review the code documentation in source files
- Open an issue on GitHub

## Performance Tips

1. **Batch operations**: Add multiple documents at once
2. **Reuse models**: Initialize RAGSystem once and reuse
3. **Clear cache**: Call `rag.clear_cache()` after large operations
4. **Choose right model**: Balance between speed and accuracy
5. **Monitor memory**: Use `nvidia-smi` to watch GPU memory usage

---

**Time to first result: ~2 minutes**
**Full setup with examples: ~5 minutes**
