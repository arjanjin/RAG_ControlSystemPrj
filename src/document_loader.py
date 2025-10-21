"""Document loader for control system materials."""
import os
from typing import List
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class ControlSystemDocumentLoader:
    """Loads and processes control system documents."""
    
    def __init__(self, documents_dir: str = "documents"):
        """Initialize the document loader.
        
        Args:
            documents_dir: Directory containing control system documents
        """
        self.documents_dir = documents_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def load_documents(self) -> List[Document]:
        """Load all documents from the documents directory.
        
        Returns:
            List of loaded documents
        """
        if not os.path.exists(self.documents_dir):
            print(f"Warning: Documents directory '{self.documents_dir}' does not exist")
            return []
        
        documents = []
        
        # Load text files
        try:
            loader = DirectoryLoader(
                self.documents_dir,
                glob="**/*.txt",
                loader_cls=TextLoader,
                show_progress=True
            )
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading documents: {e}")
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        return self.text_splitter.split_documents(documents)
