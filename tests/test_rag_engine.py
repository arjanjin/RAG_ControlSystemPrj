"""
Tests for the RAG engine module.
"""

import pytest
from rag_control_system.core.rag_engine import RAGEngine
from rag_control_system.core.retriever import DocumentRetriever
from rag_control_system.core.generator import ResponseGenerator


class TestRAGEngine:
    """Tests for RAGEngine class."""

    def test_initialization(self):
        """Test engine initialization."""
        engine = RAGEngine()
        assert engine.retriever is not None
        assert engine.generator is not None

    def test_initialization_with_components(self):
        """Test engine initialization with custom components."""
        retriever = DocumentRetriever(top_k=10)
        generator = ResponseGenerator(temperature=0.5)

        engine = RAGEngine(retriever=retriever, generator=generator)
        assert engine.retriever.top_k == 10
        assert engine.generator.temperature == 0.5

    def test_index_documents(self):
        """Test document indexing."""
        engine = RAGEngine()
        documents = [
            {"id": "1", "content": "Test document 1"},
            {"id": "2", "content": "Test document 2"},
        ]

        engine.index_documents(documents)
        assert engine.retriever.vectorstore is not None

    def test_add_documents(self):
        """Test adding documents."""
        engine = RAGEngine()
        initial_docs = [{"id": "1", "content": "Doc 1"}]
        engine.index_documents(initial_docs)

        new_docs = [{"id": "2", "content": "Doc 2"}]
        engine.add_documents(new_docs)

        assert len(engine.retriever.vectorstore["documents"]) == 2

    def test_query_without_documents(self):
        """Test querying without indexed documents."""
        engine = RAGEngine()
        result = engine.query("test question")

        assert result["question"] == "test question"
        assert "couldn't find" in result["answer"].lower()
        assert result["documents"] == []
        assert result["metadata"]["documents_retrieved"] == 0

    def test_query_with_documents(self):
        """Test querying with indexed documents."""
        engine = RAGEngine()
        documents = [
            {"id": "1", "content": "This is about examinations"},
            {"id": "2", "content": "This is about testing"},
        ]
        engine.index_documents(documents)

        result = engine.query("What about examinations?")

        assert result["question"] == "What about examinations?"
        assert isinstance(result["answer"], str)
        assert len(result["documents"]) > 0
        assert result["metadata"]["documents_retrieved"] > 0

    def test_clear_index(self):
        """Test clearing the document index."""
        engine = RAGEngine()
        documents = [{"id": "1", "content": "Test"}]
        engine.index_documents(documents)

        assert engine.retriever.vectorstore is not None

        engine.clear_index()
        assert engine.retriever.vectorstore is None

    def test_get_config(self):
        """Test getting configuration."""
        engine = RAGEngine()
        config = engine.get_config()

        assert "retriever" in config
        assert "generator" in config
        assert "embedding_model" in config["retriever"]
        assert "model_name" in config["generator"]
