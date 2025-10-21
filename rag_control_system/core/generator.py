"""
Response generator module for generating answers using LLM.
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    Generates responses using retrieved documents and a language model.
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 500,
    ):
        """
        Initialize the response generator.

        Args:
            model_name: Name of the LLM to use
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in generated response
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(f"Initialized ResponseGenerator with model: {self.model_name}")

    def generate(
        self,
        query: str,
        context_documents: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a response based on query and retrieved documents.

        Args:
            query: The user's query
            context_documents: Retrieved relevant documents
            system_prompt: Optional system prompt for the LLM

        Returns:
            Generated response string
        """
        if not context_documents:
            logger.warning("No context documents provided for generation")
            return "I don't have enough information to answer this question."

        # Build context from documents
        context = self._build_context(context_documents)

        # Build prompt
        prompt = self._build_prompt(query, context, system_prompt)

        logger.info(f"Generating response for query: {query[:50]}...")

        # Placeholder for actual LLM call
        # In production, this would call OpenAI or similar API
        response = self._mock_generate(prompt)

        return response

    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Build context string from retrieved documents.

        Args:
            documents: List of document dictionaries

        Returns:
            Formatted context string
        """
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", doc.get("text", ""))
            context_parts.append(f"[Document {i}]\n{content}\n")

        return "\n".join(context_parts)

    def _build_prompt(
        self, query: str, context: str, system_prompt: Optional[str] = None
    ) -> str:
        """
        Build the full prompt for the LLM.

        Args:
            query: User query
            context: Context from documents
            system_prompt: Optional system prompt

        Returns:
            Complete prompt string
        """
        default_system = (
            "You are a helpful assistant for an examination control system. "
            "Answer questions based on the provided context."
        )

        system = system_prompt or default_system

        prompt = f"""{system}

Context:
{context}

Question: {query}

Answer:"""

        return prompt

    def _mock_generate(self, prompt: str) -> str:
        """
        Mock response generation for testing.

        Args:
            prompt: The prompt to generate from

        Returns:
            Mock response
        """
        # This is a placeholder implementation
        # In production, this would call an actual LLM API
        return (
            "This is a mock response. Please configure a real LLM API "
            "to get actual generated responses."
        )

    def set_parameters(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Update generation parameters.

        Args:
            temperature: New temperature value
            max_tokens: New max tokens value
        """
        if temperature is not None:
            self.temperature = temperature
            logger.info(f"Updated temperature to {temperature}")

        if max_tokens is not None:
            self.max_tokens = max_tokens
            logger.info(f"Updated max_tokens to {max_tokens}")
