"""
Configuration settings using Pydantic.
"""

from typing import Optional
from pydantic import BaseModel, Field
from functools import lru_cache
import os


class RetrieverConfig(BaseModel):
    """Configuration for the document retriever."""

    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Embedding model for document retrieval",
    )
    collection_name: str = Field(
        default="examination_docs", description="Vector store collection name"
    )
    top_k: int = Field(default=5, description="Number of documents to retrieve")


class GeneratorConfig(BaseModel):
    """Configuration for the response generator."""

    model_name: str = Field(default="gpt-3.5-turbo", description="LLM model name")
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Generation temperature"
    )
    max_tokens: int = Field(
        default=500, gt=0, description="Maximum tokens in response"
    )
    api_key: Optional[str] = Field(default=None, description="API key for LLM service")


class Settings(BaseModel):
    """Main application settings."""

    # Application
    app_name: str = Field(default="RAG Control System")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # Components
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)

    # Data paths
    data_dir: str = Field(default="data")
    raw_data_dir: str = Field(default="data/raw")
    processed_data_dir: str = Field(default="data/processed")

    class Config:
        env_prefix = "RAG_"
        env_nested_delimiter = "__"

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        # Load API key from environment if available
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("RAG_GENERATOR__API_KEY")

        settings = cls()
        if api_key:
            settings.generator.api_key = api_key

        return settings


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings instance
    """
    return Settings.from_env()
