# RAG Control System

A Retrieval-Augmented Generation (RAG) system for examination control and management.

## Overview

This project implements a RAG-based system designed to answer questions and provide information about examination control systems. It combines document retrieval with natural language generation to provide accurate, context-aware responses.

## Features

- **Document Retrieval**: Efficiently retrieve relevant documents using vector similarity search
- **Response Generation**: Generate contextual responses using large language models
- **Flexible Configuration**: Easy configuration through environment variables and config files
- **Multiple Data Formats**: Support for JSON, CSV, and text file formats
- **Interactive Mode**: Command-line interface for interactive querying
- **Extensible Architecture**: Modular design for easy customization and extension

## Installation

1. Clone the repository:
```bash
git clone https://github.com/arjanjin/RAG_ControlSystemPrj.git
cd RAG_ControlSystemPrj
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### Basic Usage

```python
from rag_control_system import RAGEngine

# Initialize the engine
engine = RAGEngine()

# Index some documents
documents = [
    {"id": "1", "content": "Examination rules and regulations..."},
    {"id": "2", "content": "Grading policies..."},
]
engine.index_documents(documents)

# Query the system
result = engine.query("What are the examination rules?")
print(result["answer"])
```

### Command Line Usage

Process a single query:
```bash
python main.py --index data/sample_documents.json --query "What is the examination control system?"
```

Interactive mode:
```bash
python main.py --index data/sample_documents.json --interactive
```

### Run Example

```bash
python examples/basic_usage.py
```

## Project Structure

```
RAG_ControlSystemPrj/
├── rag_control_system/         # Main package
│   ├── core/                   # Core modules
│   │   ├── rag_engine.py      # Main RAG orchestrator
│   │   ├── retriever.py       # Document retriever
│   │   └── generator.py       # Response generator
│   ├── models/                 # Data models
│   │   ├── document.py        # Document model
│   │   └── query.py           # Query and response models
│   ├── config/                 # Configuration
│   │   └── settings.py        # Settings management
│   └── utils/                  # Utility functions
│       ├── logger.py          # Logging utilities
│       └── data_loader.py     # Data loading utilities
├── tests/                      # Test suite
├── examples/                   # Usage examples
├── data/                       # Data directory
│   ├── sample_documents.json  # Sample documents
│   ├── raw/                   # Raw data
│   └── processed/             # Processed data
├── main.py                    # Main entry point
├── requirements.txt           # Dependencies
└── setup.py                   # Package setup

```

## Configuration

Configuration can be set through environment variables with the `RAG_` prefix:

```bash
export RAG_GENERATOR__MODEL_NAME=gpt-4
export RAG_GENERATOR__TEMPERATURE=0.7
export RAG_RETRIEVER__TOP_K=5
export OPENAI_API_KEY=your_api_key_here
```

Or create a `.env` file in the project root.

## Testing

Run tests with pytest:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=rag_control_system tests/
```

## Development

### Code Style

The project uses:
- `black` for code formatting
- `flake8` for linting
- `mypy` for type checking

Run checks:
```bash
black rag_control_system/
flake8 rag_control_system/
mypy rag_control_system/
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 
