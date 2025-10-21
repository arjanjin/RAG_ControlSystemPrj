"""Vector store management for RAG system."""
import os
from typing import List, Optional
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


class VectorStoreManager:
    """Manages the vector database for semantic search."""
    
    def __init__(self, persist_directory: str = "chroma_db", api_key: Optional[str] = None):
        """Initialize the vector store manager.
        
        Args:
            persist_directory: Directory to persist the vector database
            api_key: OpenAI API key for embeddings
        """
        self.persist_directory = persist_directory
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.vectorstore = None
    
    def create_vectorstore(self, documents: List[Document]) -> None:
        """Create or update the vector store with documents.
        
        Args:
            documents: List of document chunks to add to the vector store
        """
        if not documents:
            print("Warning: No documents provided to create vector store")
            return
        
        print(f"Creating vector store with {len(documents)} document chunks...")
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        print(f"Vector store created successfully with {len(documents)} chunks")
    
    def load_vectorstore(self) -> bool:
        """Load existing vector store from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(self.persist_directory):
            print(f"Vector store directory '{self.persist_directory}' does not exist")
            return False
        
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print("Vector store loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search on the vector store.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of most similar documents
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Create or load a vector store first.")
        
        return self.vectorstore.similarity_search(query, k=k)
    
    def get_retriever(self, k: int = 4):
        """Get a retriever for the vector store.
        
        Args:
            k: Number of results to retrieve
            
        Returns:
            Retriever object
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Create or load a vector store first.")
        
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
