# RAG Control System Project

A Retrieval-Augmented Generation (RAG) system for examining and learning about control systems. This application uses LangChain, ChromaDB, and OpenAI to provide an intelligent question-answering interface for control system topics.

## Features

- 📚 Document ingestion and processing for control system materials
- 🔍 Semantic search using vector embeddings
- 🤖 Intelligent question-answering powered by GPT-3.5
- 💾 Persistent vector database using ChromaDB
- 📖 Comprehensive control system knowledge base included

## Knowledge Base

The system comes with pre-loaded documents covering:
- Control Systems Fundamentals (open-loop, closed-loop, components)
- PID Controllers (theory, tuning methods, applications)
- Stability Analysis (Routh-Hurwitz, Nyquist, Bode plots, root locus)

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Setup

1. Clone the repository:
```bash
git clone https://github.com/arjanjin/RAG_ControlSystemPrj.git
cd RAG_ControlSystemPrj
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Usage

### Running the Application

Start the interactive question-answering interface:

```bash
python main.py
```

### Example Questions

Try asking questions like:
- "What is a PID controller?"
- "Explain the difference between open-loop and closed-loop control systems"
- "How do I tune a PID controller using the Ziegler-Nichols method?"
- "What is the Routh-Hurwitz stability criterion?"
- "What are the advantages and disadvantages of feedback control?"

### Adding Your Own Documents

To add your own control system documents:

1. Place your text files (`.txt`) in the `documents/` directory
2. Run the application with force reload to reindex:

```python
from src.rag_system import ControlSystemRAG

rag = ControlSystemRAG()
rag.initialize(force_reload=True)
```

## Project Structure

```
RAG_ControlSystemPrj/
├── src/
│   ├── __init__.py
│   ├── document_loader.py    # Document loading and chunking
│   ├── vector_store.py        # Vector database management
│   └── rag_system.py          # Main RAG system
├── documents/                  # Control system documents
│   ├── control_systems_basics.txt
│   ├── pid_controllers.txt
│   └── stability_analysis.txt
├── main.py                     # Interactive CLI application
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## Dependencies

- **langchain**: Framework for building LLM applications
- **chromadb**: Vector database for embeddings storage
- **openai**: OpenAI API client
- **python-dotenv**: Environment variable management
- **tiktoken**: Tokenizer for OpenAI models
- **pypdf**: PDF document processing support

## Architecture

The RAG system follows this workflow:

1. **Document Loading**: Control system documents are loaded from the `documents/` directory
2. **Text Chunking**: Documents are split into manageable chunks with overlap
3. **Embedding**: Text chunks are converted to vector embeddings using OpenAI
4. **Vector Storage**: Embeddings are stored in ChromaDB for efficient retrieval
5. **Query Processing**: User questions are embedded and used to find relevant chunks
6. **Answer Generation**: Retrieved context is passed to GPT-3.5 to generate answers

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Customization

You can customize the RAG system by modifying:

- **Chunk size**: Edit `chunk_size` in `src/document_loader.py`
- **Number of retrieved documents**: Modify `k` parameter in retrieval
- **LLM model**: Change the model in `src/rag_system.py`
- **Temperature**: Adjust for more creative or factual responses

## Advanced Usage

### Programmatic Usage

```python
from src.rag_system import ControlSystemRAG

# Initialize the system
rag = ControlSystemRAG(
    documents_dir="documents",
    persist_directory="chroma_db"
)
rag.initialize()

# Ask a question
result = rag.query("What is a PID controller?")
print(result["answer"])

# View source documents
for doc in result["sources"]:
    print(doc.page_content)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Built with [LangChain](https://langchain.com/)
- Vector storage by [ChromaDB](https://www.trychroma.com/)
- Powered by [OpenAI](https://openai.com/) 
