# GPU Optimization Guide

This guide provides detailed information about optimizing GPU usage for the RAG Control System.

## Table of Contents

1. [Understanding GPU Acceleration](#understanding-gpu-acceleration)
2. [Hardware Requirements](#hardware-requirements)
3. [Memory Management](#memory-management)
4. [Performance Tuning](#performance-tuning)
5. [Monitoring and Debugging](#monitoring-and-debugging)
6. [Multi-GPU Setup](#multi-gpu-setup)
7. [Cloud GPU Deployment](#cloud-gpu-deployment)

## Understanding GPU Acceleration

### Why GPU?

GPUs accelerate RAG systems by:
- **Parallel Processing**: Process thousands of embeddings simultaneously
- **Matrix Operations**: Optimized for transformer model computations
- **Faster Inference**: 10-100x speedup compared to CPU
- **Batch Processing**: Efficiently handle multiple documents at once

### What Gets Accelerated?

- ✅ **Embedding Generation**: Creating vector representations
- ✅ **Similarity Computation**: Finding similar documents
- ✅ **Neural Network Inference**: Model predictions
- ❌ **Data Loading**: Still CPU-bound
- ❌ **Text Preprocessing**: Minimal GPU benefit

## Hardware Requirements

### Minimum Requirements

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| GPU | GTX 1060 (6GB) | RTX 3060 (12GB) | RTX 4090 (24GB) |
| CUDA Cores | 1280 | 3584 | 16384 |
| Memory | 6GB | 12GB | 24GB+ |
| CUDA Version | 11.7 | 12.0+ | 12.0+ |
| PCIe | 3.0 x16 | 4.0 x16 | 4.0 x16 |

### Model Size vs GPU Memory

| Model | Parameters | GPU Memory | Documents/Batch |
|-------|------------|------------|----------------|
| MiniLM-L6 | 22M | 1GB | 1000+ |
| MPNet-Base | 110M | 2GB | 500+ |
| Large Models | 300M+ | 4GB+ | 100+ |

## Memory Management

### Setting Memory Fraction

Control how much GPU memory the system uses:

```python
from src.rag_system import RAGSystem

# Use 80% of GPU memory (recommended)
rag = RAGSystem(memory_fraction=0.8)

# Conservative: Use 50% for shared GPU
rag = RAGSystem(memory_fraction=0.5)

# Aggressive: Use 95% for dedicated GPU
rag = RAGSystem(memory_fraction=0.95)
```

### Clearing GPU Cache

Free memory after operations:

```python
# Clear cache manually
rag.clear_cache()
```

### Memory Monitoring

Monitor GPU memory usage:

```python
# Get memory statistics
info = rag.get_system_info()
print(info['gpu_info'])
```

## Performance Tuning

### Model Selection

Choose embedding models based on your needs:

```python
# Fast and lightweight (best for most cases)
rag = RAGSystem(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# Better quality, more memory
rag = RAGSystem(
    embedding_model="sentence-transformers/all-mpnet-base-v2"
)

# Specialized for Q&A
rag = RAGSystem(
    embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
)
```

## Monitoring and Debugging

### Real-time Monitoring

Use `nvidia-smi` for monitoring:

```bash
# Watch GPU usage every 1 second
watch -n 1 nvidia-smi

# Or detailed query
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 1
```

### Check GPU Configuration

```python
from src.gpu_config import GPUConfig

config = GPUConfig()
info = config.get_device_info()

for key, value in info.items():
    print(f"{key}: {value}")
```

## Multi-GPU Setup

### Using Specific GPU

```python
from src.rag_system import RAGSystem

# Use GPU 0
rag = RAGSystem(device="cuda:0", gpu_id=0)

# Use GPU 1
rag = RAGSystem(device="cuda:1", gpu_id=1)
```

## Cloud GPU Deployment

### AWS EC2 GPU Instances

Recommended instances:
- **g4dn.xlarge**: T4 GPU, 16GB, ~$0.50/hr (development)
- **g5.xlarge**: A10G GPU, 24GB, ~$1.00/hr (production)
- **p3.2xlarge**: V100 GPU, 16GB, ~$3.00/hr (high-performance)

### Google Cloud Platform

Recommended instances:
- **n1-standard-4 + T4**: ~$0.35/hr (development)
- **n1-standard-8 + V100**: ~$2.50/hr (production)

### Docker Deployment

```dockerfile
FROM nvidia/cuda:12.0-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Run application
CMD ["python3", "example.py"]
```

Run with GPU:
```bash
docker run --gpus all my-rag-system
```

## Best Practices Summary

1. ✅ **Start Conservative**: Use 70-80% memory fraction initially
2. ✅ **Monitor First**: Profile before optimizing
3. ✅ **Batch Processing**: Process multiple documents together
4. ✅ **Clear Cache**: Regularly free unused memory
5. ✅ **Choose Right Model**: Balance speed vs accuracy
6. ✅ **Test Thoroughly**: Verify GPU is actually being used
7. ✅ **Have Fallback**: Always support CPU mode

## Troubleshooting Common Issues

### GPU Not Being Used

```python
# Verify GPU is selected
print(rag.gpu_config.device)  # Should show "cuda:X"
```

### Slow Performance

1. Check if CUDA drivers are current
2. Monitor GPU utilization (should be >80%)
3. Verify model is on GPU

### Out of Memory

1. Reduce `memory_fraction`
2. Use smaller embedding model
3. Clear GPU cache: `rag.clear_cache()`

---

For more information, see:
- [README.md](../README.md) - General documentation
- [QUICKSTART.md](../QUICKSTART.md) - Getting started guide
