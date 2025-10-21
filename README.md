# RAG Control System Project

A GPU-accelerated Retrieval Augmented Generation (RAG) system for examination control and management.

## Overview

This project implements a high-performance RAG system that leverages GPU acceleration for fast document retrieval and semantic search. It's designed specifically for examination control systems where quick access to relevant information is crucial.

## Features

- **GPU Acceleration**: Full support for CUDA-enabled GPUs for faster embeddings and inference
- **Automatic Device Detection**: Automatically detects and uses available GPU, falls back to CPU if needed
- **Flexible Configuration**: Easy configuration through YAML files
- **Scalable Architecture**: Designed to handle large document collections efficiently
- **Vector Search**: Fast similarity search using FAISS with GPU support

## GPU Requirements

### Supported Configurations

- **GPU**: NVIDIA GPU with CUDA Compute Capability 3.5 or higher
- **CUDA**: Version 11.7 or higher recommended
- **Memory**: At least 4GB GPU memory recommended (8GB+ for larger models)
- **CPU Fallback**: Automatic fallback to CPU if GPU is not available

### Recommended GPUs

- NVIDIA RTX 3060 or better (12GB+ VRAM)
- NVIDIA Tesla T4 or better for cloud deployments
- NVIDIA A100 for production workloads

## Installation

### Prerequisites

1. **Install CUDA Toolkit** (for GPU support):
   ```bash
   # Visit https://developer.nvidia.com/cuda-downloads
   # Follow instructions for your operating system
   ```

2. **Verify CUDA installation**:
   ```bash
   nvcc --version
   nvidia-smi
   ```

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/arjanjin/RAG_ControlSystemPrj.git
   cd RAG_ControlSystemPrj
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   # For GPU support (recommended)
   pip install -r requirements.txt
   
   # For CPU-only installation
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```

4. **Configure the system**:
   ```bash
   cp config.yaml.example config.yaml
   # Edit config.yaml to match your setup
   ```

## Usage

### Quick Start

```python
from src.rag_system import RAGSystem

# Initialize with GPU (automatic detection)
rag = RAGSystem()

# Add documents
documents = [
    "Examination policies and procedures",
    "Student assessment guidelines",
    "Grading rubrics and criteria"
]
rag.add_documents(documents)

# Search for relevant documents
results = rag.search("What are the grading criteria?", top_k=3)
for result in results:
    print(f"Score: {result['score']:.4f} - {result['text']}")
```

### GPU Configuration

```python
from src.rag_system import RAGSystem

# Specify GPU device
rag = RAGSystem(device="cuda:0", gpu_id=0, memory_fraction=0.8)

# Check GPU status
info = rag.get_system_info()
print(info['gpu_info'])

# Clear GPU cache when needed
rag.clear_cache()
```

### Testing GPU Setup

```bash
# Test GPU configuration
python -m src.gpu_config

# Test RAG system
python -m src.rag_system
```

## GPU Performance Optimization

### Tips for Better Performance

1. **Batch Processing**: Process multiple documents at once
   ```python
   rag.add_documents(large_document_list)  # Processes in batches
   ```

2. **Memory Management**: Adjust memory fraction based on your GPU
   ```python
   rag = RAGSystem(memory_fraction=0.7)  # Use 70% of GPU memory
   ```

3. **Model Selection**: Choose appropriate embedding models
   - Small/Fast: `all-MiniLM-L6-v2` (80MB, 384 dim)
   - Medium: `all-mpnet-base-v2` (420MB, 768 dim)
   - Large: `multi-qa-mpnet-base-dot-v1` (420MB, 768 dim)

4. **Clear Cache**: Free GPU memory when switching tasks
   ```python
   rag.clear_cache()
   ```

## Configuration Options

Edit `config.yaml` to customize:

- **GPU Settings**: Device selection, memory allocation
- **Embedding Model**: Choose different models for your needs
- **Vector Store**: Configure FAISS or other vector stores
- **RAG Parameters**: Top-k results, score thresholds
- **API Settings**: Host, port, documentation

## Troubleshooting

### GPU Not Detected

```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU visibility
nvidia-smi
```

### Out of Memory Errors

- Reduce `memory_fraction` in configuration
- Use a smaller embedding model
- Process documents in smaller batches
- Clear GPU cache regularly

### Slow Performance

- Ensure GPU drivers are up to date
- Check if GPU is being used: `nvidia-smi`
- Consider using mixed precision (FP16)
- Verify CUDA and PyTorch versions are compatible

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with GPU
CUDA_VISIBLE_DEVICES=0 pytest tests/
```

### Adding New Features

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues, questions, or contributions, please open an issue on GitHub.

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/) for GPU acceleration
- Uses [Sentence Transformers](https://www.sbert.net/) for embeddings
- Powered by [FAISS](https://github.com/facebookresearch/faiss) for vector search

