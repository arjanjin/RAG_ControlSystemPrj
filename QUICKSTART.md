# Quick Start Guide

Get up and running with the Control System RAG in 5 minutes!

## Prerequisites

- Python 3.8+
- OpenAI API key

## Installation (3 steps)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Create a `.env` file:

```bash
echo "OPENAI_API_KEY=your-key-here" > .env
```

Replace `your-key-here` with your actual OpenAI API key from https://platform.openai.com/api-keys

### 3. Run the Application

```bash
python main.py
```

That's it! The system will initialize and you can start asking questions.

## First Questions to Try

Once the application starts, try these questions:

1. **Basic Understanding:**
   ```
   What is a control system?
   ```

2. **Technical Details:**
   ```
   How does a PID controller work?
   ```

3. **Practical Application:**
   ```
   How do I tune a PID controller?
   ```

4. **Advanced Topics:**
   ```
   What is the Routh-Hurwitz stability criterion?
   ```

## Example Session

```
Control System RAG - Examination Assistant
================================================================================

Initializing RAG system...
Creating vector store with 93 document chunks...
Vector store created successfully with 93 chunks
RAG system initialized successfully!
You can now ask questions about control systems.
Type 'quit' or 'exit' to stop.

================================================================================

Your question: What is a PID controller?

Searching for answer...
================================================================================
Answer:
A PID controller is a control algorithm that uses three terms: Proportional (P),
Integral (I), and Derivative (D) to control a process. The proportional term 
provides immediate response to error, the integral term eliminates steady-state 
error, and the derivative term improves stability and reduces overshoot...

Sources:
1. documents/pid_controllers.txt
   Content preview: PID Controllers

Introduction
PID (Proportional-Integral-Derivative) controllers are the most widely used 
control algorithm...

================================================================================
```

## Programmatic Usage

If you want to integrate the RAG system into your own code:

```python
from src.rag_system import ControlSystemRAG

# Initialize
rag = ControlSystemRAG()
rag.initialize()

# Ask a question
result = rag.query("What is a PID controller?")
print(result["answer"])

# Access sources
for doc in result["sources"]:
    print(f"Source: {doc.metadata['source']}")
```

## Common Issues

### "OPENAI_API_KEY not set"

**Solution:** Make sure you created the `.env` file with your API key:

```bash
cat .env  # Should show: OPENAI_API_KEY=your-key-here
```

### "ModuleNotFoundError"

**Solution:** Install dependencies:

```bash
pip install -r requirements.txt
```

### Using virtual environment (recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt

# Run application
python main.py
```

## Next Steps

1. ✅ Try the example questions above
2. 📚 Read [FEATURES.md](FEATURES.md) for capabilities
3. 🏗️ Check [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
4. 📖 See [README.md](README.md) for comprehensive guide
5. ➕ Add your own documents to `documents/` folder

## Adding Your Own Documents

1. Create or copy `.txt` files into the `documents/` folder
2. Delete the `chroma_db/` folder (if it exists)
3. Run `python main.py` again
4. Your documents will be indexed automatically!

## Tips

- 💡 Ask specific questions for better answers
- 📝 The system cites sources for every answer
- 🔄 First run takes ~1 minute to create embeddings
- ⚡ Subsequent runs start in ~5 seconds
- 💰 Each query costs ~$0.001 (very cheap!)

## Getting Help

1. Check [SETUP.md](SETUP.md) for detailed installation
2. Read [FEATURES.md](FEATURES.md) for feature documentation
3. Review [ARCHITECTURE.md](ARCHITECTURE.md) for system design
4. Run `python example.py` to see working examples

## Support

For issues or questions:
- Check existing documentation files
- Review error messages carefully
- Ensure API key is valid and has credits
- Verify internet connection for API calls

---

**Ready to learn about control systems? Start asking questions! 🚀**
