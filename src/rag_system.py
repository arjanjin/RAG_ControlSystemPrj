"""RAG System for Control System Examination."""
import os
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from src.document_loader import ControlSystemDocumentLoader
from src.vector_store import VectorStoreManager


class ControlSystemRAG:
    """RAG system for examining control systems."""
    
    def __init__(self, documents_dir: str = "documents", 
                 persist_directory: str = "chroma_db",
                 api_key: Optional[str] = None):
        """Initialize the RAG system.
        
        Args:
            documents_dir: Directory containing control system documents
            persist_directory: Directory to persist the vector database
            api_key: OpenAI API key
        """
        self.documents_dir = documents_dir
        self.persist_directory = persist_directory
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        self.doc_loader = ControlSystemDocumentLoader(documents_dir)
        self.vector_store = VectorStoreManager(persist_directory, self.api_key)
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=self.api_key
        )
        self.qa_chain = None
    
    def initialize(self, force_reload: bool = False) -> None:
        """Initialize the RAG system by loading or creating the vector store.
        
        Args:
            force_reload: If True, recreate the vector store from documents
        """
        if force_reload or not self.vector_store.load_vectorstore():
            print("Creating new vector store from documents...")
            documents = self.doc_loader.load_documents()
            
            if not documents:
                raise ValueError("No documents found. Please add documents to the documents directory.")
            
            print(f"Loaded {len(documents)} documents")
            chunks = self.doc_loader.split_documents(documents)
            print(f"Split into {len(chunks)} chunks")
            
            self.vector_store.create_vectorstore(chunks)
        
        self._setup_qa_chain()
    
    def _setup_qa_chain(self) -> None:
        """Setup the question-answering chain."""
        prompt_template = """You are an expert in control systems. Use the following pieces of context to answer the question about control systems. 
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer: """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        retriever = self.vector_store.get_retriever(k=4)
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
    
    def query(self, question: str) -> dict:
        """Query the RAG system with a question.
        
        Args:
            question: Question about control systems
            
        Returns:
            Dictionary with answer and source documents
        """
        if not self.qa_chain:
            raise ValueError("RAG system not initialized. Call initialize() first.")
        
        result = self.qa_chain.invoke({"query": question})
        
        return {
            "answer": result["result"],
            "sources": result["source_documents"]
        }
    
    def ask(self, question: str) -> str:
        """Ask a question and get the answer.
        
        Args:
            question: Question about control systems
            
        Returns:
            Answer string
        """
        result = self.query(question)
        return result["answer"]
