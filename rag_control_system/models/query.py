"""
Query and response data models.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class Query(BaseModel):
    """Represents a user query."""

    question: str = Field(..., description="The user's question")
    top_k: Optional[int] = Field(default=None, description="Number of documents to retrieve")
    system_prompt: Optional[str] = Field(
        default=None, description="Optional system prompt"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional query metadata"
    )


class QueryResponse(BaseModel):
    """Represents a response to a query."""

    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="Generated answer")
    documents: List[Dict[str, Any]] = Field(
        default_factory=list, description="Retrieved documents"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Response metadata"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Response timestamp"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "question": self.question,
            "answer": self.answer,
            "documents": self.documents,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }
