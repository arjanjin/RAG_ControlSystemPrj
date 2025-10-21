"""
Tests for the response generator module.
"""

import pytest
from rag_control_system.core.generator import ResponseGenerator


class TestResponseGenerator:
    """Tests for ResponseGenerator class."""

    def test_initialization(self):
        """Test generator initialization."""
        generator = ResponseGenerator()
        assert generator.model_name == "gpt-3.5-turbo"
        assert generator.temperature == 0.7
        assert generator.max_tokens == 500

    def test_custom_initialization(self):
        """Test generator with custom parameters."""
        generator = ResponseGenerator(
            model_name="gpt-4", temperature=0.5, max_tokens=1000
        )
        assert generator.model_name == "gpt-4"
        assert generator.temperature == 0.5
        assert generator.max_tokens == 1000

    def test_generate_without_context(self):
        """Test generation without context documents."""
        generator = ResponseGenerator()
        result = generator.generate("test query", [])
        assert "don't have enough information" in result.lower()

    def test_generate_with_context(self):
        """Test generation with context documents."""
        generator = ResponseGenerator()
        documents = [
            {"content": "This is test content 1"},
            {"content": "This is test content 2"},
        ]

        result = generator.generate("test query", documents)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_build_context(self):
        """Test context building from documents."""
        generator = ResponseGenerator()
        documents = [
            {"content": "Content 1"},
            {"content": "Content 2"},
        ]

        context = generator._build_context(documents)
        assert "[Document 1]" in context
        assert "[Document 2]" in context
        assert "Content 1" in context
        assert "Content 2" in context

    def test_build_prompt(self):
        """Test prompt building."""
        generator = ResponseGenerator()
        query = "What is the answer?"
        context = "Context information here"

        prompt = generator._build_prompt(query, context)
        assert query in prompt
        assert context in prompt
        assert "Context:" in prompt

    def test_build_prompt_with_system(self):
        """Test prompt building with custom system prompt."""
        generator = ResponseGenerator()
        query = "What is the answer?"
        context = "Context information"
        system_prompt = "You are a helpful assistant."

        prompt = generator._build_prompt(query, context, system_prompt)
        assert system_prompt in prompt
        assert query in prompt

    def test_set_parameters(self):
        """Test updating generation parameters."""
        generator = ResponseGenerator()

        generator.set_parameters(temperature=0.3)
        assert generator.temperature == 0.3

        generator.set_parameters(max_tokens=800)
        assert generator.max_tokens == 800

        generator.set_parameters(temperature=0.9, max_tokens=1200)
        assert generator.temperature == 0.9
        assert generator.max_tokens == 1200
