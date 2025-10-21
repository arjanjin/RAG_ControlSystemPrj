"""
RAG Engine Module
โมดูลหลักสำหรับ Retrieval-Augmented Generation
"""

from typing import List, Dict, Optional, Any
import logging
import json

from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain

# Handle imports for both module and standalone usage
try:
    import config
    from src.vector_store import VectorStoreManager
    from src.document_loader import DocumentLoader
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    import config
    from src.vector_store import VectorStoreManager
    from src.document_loader import DocumentLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEngine:
    """
    คลาสหลักสำหรับ RAG Pipeline
    รวม document retrieval และ generation
    """

    def __init__(
        self,
        vector_store_manager: Optional[VectorStoreManager] = None,
        llm_provider: str = None
    ):
        """
        Initialize RAG Engine

        Args:
            vector_store_manager: Vector store manager (ถ้าไม่ระบุจะสร้างใหม่)
            llm_provider: LLM provider ("ollama" or "openai")
        """
        self.vector_store = vector_store_manager or VectorStoreManager()
        self.llm_provider = llm_provider or config.LLM_PROVIDER

        # สร้าง LLM
        self._setup_llm()

    def _setup_llm(self):
        """ตั้งค่า Language Model"""
        try:
            if self.llm_provider == "ollama":
                logger.info(f"Using Ollama model: {config.OLLAMA_MODEL}")
                self.llm = Ollama(
                    model=config.OLLAMA_MODEL,
                    base_url=config.OLLAMA_BASE_URL
                )
            elif self.llm_provider == "openai":
                from langchain_community.llms import OpenAI
                logger.info("Using OpenAI model")
                self.llm = OpenAI(
                    api_key=config.OPENAI_API_KEY,
                    temperature=0.7
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

            logger.info("LLM initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise

    def retrieve_context(
        self,
        query: str,
        k: int = None
    ) -> List[Document]:
        """
        ค้นหาความรู้ที่เกี่ยวข้อง

        Args:
            query: คำค้นหา
            k: จำนวนผลลัพธ์

        Returns:
            List[Document]: เอกสารที่เกี่ยวข้อง
        """
        k = k or config.TOP_K_RESULTS
        return self.vector_store.similarity_search(query, k=k)

    def format_context(self, documents: List[Document]) -> str:
        """
        จัดรูปแบบเอกสารเป็น context string

        Args:
            documents: รายการเอกสาร

        Returns:
            str: context ที่จัดรูปแบบแล้ว
        """
        if not documents:
            return "ไม่พบข้อมูลที่เกี่ยวข้อง"

        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[เอกสาร {i}]\n{doc.page_content}\n")

        return "\n".join(context_parts)

    def query(
        self,
        question: str,
        k: int = None,
        return_context: bool = False
    ) -> Dict[str, Any]:
        """
        Query RAG system

        Args:
            question: คำถาม
            k: จำนวน context ที่จะค้นหา
            return_context: ส่งคืน context ด้วยหรือไม่

        Returns:
            Dict: ผลลัพธ์
        """
        # ค้นหา context
        retrieved_docs = self.retrieve_context(question, k=k)
        context = self.format_context(retrieved_docs)

        # สร้าง prompt
        prompt = f"""
คุณเป็นผู้เชี่ยวชาญด้านระบบควบคุม (Control System) และต้องตอบเป็นภาษาไทยเท่านั้น

คำถาม: {question}

ข้อมูลที่เกี่ยวข้อง:
{context}

คำแนะนำ:
- ตอบคำถามโดยอ้างอิงจากข้อมูลที่ให้มา
- หากไม่พบข้อมูลที่เกี่ยวข้อง ให้บอกว่าไม่มีข้อมูล
- ตอบเป็นภาษาไทยเท่านั้น
- ให้คำตอบที่ชัดเจนและถูกต้อง

คำตอบ:
"""

        # Generate answer
        try:
            answer = self.llm.invoke(prompt)

            result = {
                "question": question,
                "answer": answer,
                "num_sources": len(retrieved_docs)
            }

            if return_context:
                result["context"] = context
                result["sources"] = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in retrieved_docs
                ]

            return result

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "question": question,
                "answer": f"เกิดข้อผิดพลาด: {str(e)}",
                "error": True
            }

    def load_knowledge_base(self, directory: str = None):
        """
        โหลดฐานความรู้จากโฟลเดอร์

        Args:
            directory: path ของโฟลเดอร์ (default: knowledge_base)
        """
        directory = directory or str(config.KNOWLEDGE_BASE_DIR)

        logger.info(f"Loading knowledge base from {directory}")
        loader = DocumentLoader()
        documents = loader.load_directory(directory)

        if documents:
            self.vector_store.add_documents(documents)
            logger.info(f"Loaded {len(documents)} documents into vector store")
        else:
            logger.warning("No documents found in knowledge base directory")

    def get_status(self) -> Dict[str, Any]:
        """
        ดูสถานะของ RAG Engine

        Returns:
            Dict: สถานะ
        """
        vector_info = self.vector_store.get_collection_info()

        return {
            "llm_provider": self.llm_provider,
            "vector_store": vector_info,
            "embedding_model": self.vector_store.embedding_model_name
        }


def create_rag_engine() -> RAGEngine:
    """
    สร้าง RAG Engine พร้อมโหลดฐานความรู้

    Returns:
        RAGEngine: RAG engine ที่พร้อมใช้งาน
    """
    engine = RAGEngine()
    engine.load_knowledge_base()
    return engine


def main():
    """Test function for standalone usage"""
    print("🧪 Testing RAG Engine...")
    
    try:
        # Test 1: Initialize RAG engine
        vector_store = VectorStoreManager()
        rag_engine = RAGEngine(vector_store_manager=vector_store)
        print("✓ RAG engine initialized")
        
        # Test 2: Test status
        status = rag_engine.get_status()
        print(f"✓ LLM Provider: {status['llm_provider']}")
        print(f"✓ Vector Store: {status['vector_store']['name']}")
        
        # Test 3: Test query
        test_questions = [
            "ระบบควบคุมแบบวงเปิดคืออะไร",
            "PID Controller ประกอบด้วยอะไรบ้าง",
            "Transfer Function คืออะไร"
        ]
        
        print("\n--- Testing Queries ---")
        for i, question in enumerate(test_questions, 1):
            print(f"\nTest {i}: {question}")
            result = rag_engine.query(question)
            print(f"✓ Answer: {result['answer'][:100]}...")
            print(f"✓ Sources: {result['num_sources']}")
        
        print("\n✅ RAG Engine Tests Completed!")
        
    except Exception as e:
        print(f"❌ Error in RAG engine test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
