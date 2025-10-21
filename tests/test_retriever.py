"""
Tests for the document retriever module.
"""

import pytest
from rag_control_system.core.retriever import DocumentRetriever


class TestDocumentRetriever:
    """Tests for DocumentRetriever class."""

    def test_initialization(self):
        """Test retriever initialization."""
        retriever = DocumentRetriever()
        assert retriever.embedding_model == "all-MiniLM-L6-v2"
        assert retriever.collection_name == "examination_docs"
        assert retriever.top_k == 5

    def test_custom_initialization(self):
        """Test retriever with custom parameters."""
        retriever = DocumentRetriever(
            embedding_model="custom-model", collection_name="custom_collection", top_k=10
        )
        assert retriever.embedding_model == "custom-model"
        assert retriever.collection_name == "custom_collection"
        assert retriever.top_k == 10

    def test_initialize_vectorstore(self):
        """Test vector store initialization."""
        retriever = DocumentRetriever()
        documents = [
            {"id": "1", "content": "Test document 1"},
            {"id": "2", "content": "Test document 2"},
        ]

        retriever.initialize_vectorstore(documents)
        assert retriever.vectorstore is not None
        assert retriever.vectorstore["indexed"] is True
        assert len(retriever.vectorstore["documents"]) == 2

    def test_retrieve_without_vectorstore(self):
        """Test retrieve when vectorstore is not initialized."""
        retriever = DocumentRetriever()
        results = retriever.retrieve("test query")
        assert results == []

    def test_retrieve_with_documents(self):
        """Test retrieving documents."""
        retriever = DocumentRetriever(top_k=2)
        documents = [
            {"id": "1", "content": "Test document 1"},
            {"id": "2", "content": "Test document 2"},
            {"id": "3", "content": "Test document 3"},
        ]

        retriever.initialize_vectorstore(documents)
        results = retriever.retrieve("test query")

        assert len(results) == 2
        assert results[0]["id"] == "1"

    def test_add_documents(self):
        """Test adding documents to existing vectorstore."""
        retriever = DocumentRetriever()
        initial_docs = [{"id": "1", "content": "Doc 1"}]
        retriever.initialize_vectorstore(initial_docs)

        new_docs = [{"id": "2", "content": "Doc 2"}]
        retriever.add_documents(new_docs)

        assert len(retriever.vectorstore["documents"]) == 2

    def test_clear_vectorstore(self):
        """Test clearing the vectorstore."""
        retriever = DocumentRetriever()
        documents = [{"id": "1", "content": "Test document"}]
        retriever.initialize_vectorstore(documents)

        assert retriever.vectorstore is not None

        retriever.clear()
        assert retriever.vectorstore is None
