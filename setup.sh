#!/bin/bash
# Quick Setup Script for RAG Control System
# สคริปต์สำหรับติดตั้งและเริ่มใช้งาน

echo "🚀 RAG Control System - Quick Setup"
echo "===================================="

# Check if in virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment is active: $VIRTUAL_ENV"
else
    echo "⚠️  Virtual environment not detected. Activating..."
    source venv/bin/activate
fi

# Check Ollama status
echo "🔍 Checking Ollama status..."
if curl -s http://localhost:11434 > /dev/null; then
    echo "✅ Ollama is running"
else
    echo "❌ Ollama is not running. Please start Ollama first:"
    echo "   ollama serve"
    exit 1
fi

# Check if required models exist
echo "🔍 Checking required models..."
if ollama list | grep -q "llama3:latest"; then
    echo "✅ llama3:latest model found"
else
    echo "📥 Downloading llama3:latest model..."
    ollama pull llama3:latest
fi

echo ""
echo "🎯 USAGE OPTIONS:"
echo "=================="
echo ""
echo "1. 🗣️  Interactive Q&A:"
echo "   python main.py --query"
echo ""
echo "2. 📝 Grade Exam:"
echo "   python main.py --grade examples/sample_exam.json --output results.json"
echo ""
echo "3. 🌐 Web Interface:"
echo "   streamlit run web_interface.py"
echo ""
echo "4. 🧪 Test Individual Modules:"
echo "   python src/document_loader.py"
echo "   python src/vector_store.py"
echo "   python src/rag_engine.py"
echo "   python src/exam_grader.py"
echo ""
echo "✅ Setup complete! Your RAG Control System is ready to use."
echo "📚 Check README.md for detailed instructions."