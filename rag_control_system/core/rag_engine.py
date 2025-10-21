"""
Main RAG engine that orchestrates retrieval and generation.
"""

from typing import List, Dict, Any, Optional
import logging

from rag_control_system.core.retriever import DocumentRetriever
from rag_control_system.core.generator import ResponseGenerator

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Main orchestrator for the RAG Control System.
    Combines document retrieval and response generation.
    """

    def __init__(
        self,
        retriever: Optional[DocumentRetriever] = None,
        generator: Optional[ResponseGenerator] = None,
    ):
        """
        Initialize the RAG engine.

        Args:
            retriever: Document retriever instance
            generator: Response generator instance
        """
        self.retriever = retriever or DocumentRetriever()
        self.generator = generator or ResponseGenerator()
        logger.info("Initialized RAGEngine")

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline.

        Args:
            question: The user's question
            top_k: Number of documents to retrieve
            system_prompt: Optional system prompt for generation

        Returns:
            Dictionary containing the answer and metadata
        """
        logger.info(f"Processing query: {question[:100]}...")

        # Retrieve relevant documents
        documents = self.retriever.retrieve(question, top_k=top_k)

        if not documents:
            logger.warning("No documents retrieved")
            return {
                "question": question,
                "answer": "I couldn't find any relevant information to answer your question.",
                "documents": [],
                "metadata": {"documents_retrieved": 0},
            }

        # Generate response
        answer = self.generator.generate(
            query=question,
            context_documents=documents,
            system_prompt=system_prompt,
        )

        result = {
            "question": question,
            "answer": answer,
            "documents": documents,
            "metadata": {
                "documents_retrieved": len(documents),
                "model": self.generator.model_name,
            },
        }

        logger.info("Query processed successfully")
        return result

    def index_documents(self, documents: List[Dict[str, Any]]):
        """
        Index documents into the retriever.

        Args:
            documents: List of documents to index
        """
        logger.info(f"Indexing {len(documents)} documents")
        self.retriever.initialize_vectorstore(documents)

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add new documents to the existing index.

        Args:
            documents: List of documents to add
        """
        logger.info(f"Adding {len(documents)} documents")
        self.retriever.add_documents(documents)

    def clear_index(self):
        """Clear all indexed documents."""
        logger.info("Clearing document index")
        self.retriever.clear()

    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.

        Returns:
            Dictionary of current settings
        """
        return {
            "retriever": {
                "embedding_model": self.retriever.embedding_model,
                "collection_name": self.retriever.collection_name,
                "top_k": self.retriever.top_k,
            },
            "generator": {
                "model_name": self.generator.model_name,
                "temperature": self.generator.temperature,
                "max_tokens": self.generator.max_tokens,
            },
        }
