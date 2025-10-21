# Setup Guide

This guide will help you set up and run the Control System RAG application.

## Step 1: Install Python

Ensure you have Python 3.8 or higher installed on your system.

Check your Python version:
```bash
python --version
```

If you need to install Python, download it from [python.org](https://www.python.org/downloads/).

## Step 2: Clone the Repository

```bash
git clone https://github.com/arjanjin/RAG_ControlSystemPrj.git
cd RAG_ControlSystemPrj
```

## Step 3: Create Virtual Environment

It's recommended to use a virtual environment to isolate dependencies:

### On Linux/Mac:
```bash
python -m venv venv
source venv/bin/activate
```

### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

## Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- LangChain and related packages
- ChromaDB for vector storage
- OpenAI client
- Other utilities

## Step 5: Get OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign up or log in
3. Create a new API key
4. Copy the key (you won't be able to see it again)

## Step 6: Configure Environment

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit the `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

**Important:** Never commit your `.env` file to version control!

## Step 7: Run the Application

```bash
python main.py
```

The first time you run the application, it will:
1. Load documents from the `documents/` directory
2. Create embeddings (this may take a minute)
3. Store embeddings in the `chroma_db/` directory
4. Start the interactive interface

## Step 8: Ask Questions

Once the application is running, you can ask questions about control systems:

```
Your question: What is a PID controller?
```

The system will:
1. Search for relevant information in the knowledge base
2. Generate an answer using GPT-3.5
3. Show the sources used

## Troubleshooting

### ImportError: No module named 'langchain'

Make sure you've activated your virtual environment and installed dependencies:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Error: OPENAI_API_KEY not set

Ensure you've created the `.env` file with your API key:
```bash
cat .env  # Check if the file exists and has your key
```

### ChromaDB initialization error

If you get errors with ChromaDB, try deleting the `chroma_db/` directory and running again:
```bash
rm -rf chroma_db/  # On Linux/Mac
# or manually delete the chroma_db folder on Windows
python main.py
```

### API rate limit errors

If you hit OpenAI API rate limits:
1. Wait a few minutes before trying again
2. Consider upgrading your OpenAI plan
3. Reduce the number of documents or chunk size

## Adding Custom Documents

To add your own control system documents:

1. Place `.txt` files in the `documents/` directory
2. Delete the `chroma_db/` directory to force reindexing
3. Run the application again

The system will automatically process and index your new documents.

## Next Steps

- Explore the example questions in the README
- Add your own control system documents
- Customize the system by modifying the source code
- Experiment with different LLM parameters

For more information, see the main [README.md](README.md).
