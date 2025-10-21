"""Main RAG system implementation with GPU acceleration."""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError:
    SentenceTransformer = None
    torch = None

from .gpu_config import GPUConfig

logger = logging.getLogger(__name__)


class RAGSystem:
    """RAG (Retrieval Augmented Generation) system for examination control with GPU support."""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        gpu_id: int = 0,
        memory_fraction: float = 0.8
    ):
        """
        Initialize the RAG system with GPU support.
        
        Args:
            embedding_model: Name of the embedding model to use
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            gpu_id: GPU device ID if multiple GPUs are available
            memory_fraction: Fraction of GPU memory to use (0.0 to 1.0)
        """
        # Initialize GPU configuration
        self.gpu_config = GPUConfig(device=device, gpu_id=gpu_id)
        self.gpu_config.set_memory_fraction(memory_fraction)
        
        logger.info(f"Initializing RAG system on {self.gpu_config.device}")
        
        # Initialize embedding model
        self.embedding_model_name = embedding_model
        self.embeddings = None
        self._initialize_embeddings()
        
        # Initialize document store (placeholder for now)
        self.documents: List[Dict[str, Any]] = []
        self.document_embeddings: Optional[Any] = None
        
    def _initialize_embeddings(self):
        """Initialize the embedding model with GPU support."""
        if SentenceTransformer is None:
            logger.error("sentence-transformers not installed. Cannot initialize embeddings.")
            return
        
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embeddings = SentenceTransformer(
                self.embedding_model_name,
                device=self.gpu_config.device
            )
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        Add documents to the RAG system.
        
        Args:
            documents: List of document texts
            metadata: Optional list of metadata dictionaries for each document
        """
        if not documents:
            logger.warning("No documents provided")
            return
        
        if metadata is None:
            metadata = [{} for _ in documents]
        
        logger.info(f"Adding {len(documents)} documents to RAG system")
        
        # Store documents
        for i, (doc, meta) in enumerate(zip(documents, metadata)):
            self.documents.append({
                "id": len(self.documents),
                "text": doc,
                "metadata": meta
            })
        
        # Generate embeddings
        if self.embeddings is not None:
            logger.info("Generating embeddings for documents")
            embeddings = self.embeddings.encode(
                documents,
                show_progress_bar=True,
                convert_to_tensor=True,
                device=self.gpu_config.device
            )
            
            if self.document_embeddings is None:
                self.document_embeddings = embeddings
            else:
                if torch is not None:
                    self.document_embeddings = torch.cat([self.document_embeddings, embeddings])
            
            logger.info(f"Generated embeddings with shape: {self.document_embeddings.shape}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using the query.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of relevant documents with scores
        """
        if not self.documents:
            logger.warning("No documents in the system")
            return []
        
        if self.embeddings is None:
            logger.error("Embeddings not initialized")
            return []
        
        # Encode query
        query_embedding = self.embeddings.encode(
            query,
            convert_to_tensor=True,
            device=self.gpu_config.device
        )
        
        # Calculate similarity scores
        if torch is not None and self.document_embeddings is not None:
            # Cosine similarity
            scores = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0),
                self.document_embeddings
            )
            
            # Get top-k results
            top_k = min(top_k, len(self.documents))
            top_scores, top_indices = torch.topk(scores, k=top_k)
            
            results = []
            for score, idx in zip(top_scores.cpu().tolist(), top_indices.cpu().tolist()):
                doc = self.documents[idx].copy()
                doc["score"] = score
                results.append(doc)
            
            return results
        
        return []
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the RAG system configuration.
        
        Returns:
            Dictionary with system information
        """
        info = {
            "embedding_model": self.embedding_model_name,
            "document_count": len(self.documents),
            "gpu_info": self.gpu_config.get_device_info()
        }
        
        if self.document_embeddings is not None:
            info["embeddings_shape"] = str(self.document_embeddings.shape)
        
        return info
    
    def clear_cache(self):
        """Clear GPU cache to free memory."""
        self.gpu_config.clear_cache()


if __name__ == "__main__":
    # Test RAG system
    logging.basicConfig(level=logging.INFO)
    
    # Initialize system
    rag = RAGSystem()
    
    # Print system info
    print("\n=== RAG System Information ===")
    for key, value in rag.get_system_info().items():
        print(f"{key}: {value}")
    
    # Test with sample documents
    sample_docs = [
        "The examination control system manages student assessments.",
        "GPU acceleration improves the performance of neural networks.",
        "RAG systems combine retrieval and generation for better responses."
    ]
    
    rag.add_documents(sample_docs)
    
    # Test search
    results = rag.search("How does GPU help with neural networks?", top_k=2)
    print("\n=== Search Results ===")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['score']:.4f}")
        print(f"   Text: {result['text']}")
