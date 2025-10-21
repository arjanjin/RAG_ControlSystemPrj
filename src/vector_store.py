"""
Vector Store Module
โมดูลสำหรับจัดการ Vector Database
"""

from typing import List, Optional
import logging
from pathlib import Path

from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb

# Handle imports for both module and standalone usage
try:
    import config
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    คลาสสำหรับจัดการ Vector Store
    ใช้ ChromaDB เป็น vector database
    """

    def __init__(
        self,
        collection_name: str = None,
        embedding_model: str = None,
        persist_directory: str = None
    ):
        """
        Initialize vector store manager

        Args:
            collection_name: ชื่อ collection (default จาก config)
            embedding_model: ชื่อโมเดล embedding (default จาก config)
            persist_directory: โฟลเดอร์สำหรับเก็บ vector database
        """
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.embedding_model_name = embedding_model or config.EMBEDDING_MODEL
        self.persist_directory = persist_directory or str(config.VECTOR_DB_DIR)

        # สร้างโฟลเดอร์ถ้ายังไม่มี
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # สร้าง embeddings
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # สร้าง/โหลด vector store
        self.vector_store = None
        
        # Disable ChromaDB telemetry to avoid errors
        import os
        os.environ["ANONYMIZED_TELEMETRY"] = "False"
        os.environ["CHROMA_TELEMETRY"] = "False"
        
        self._load_or_create_vector_store()

    def _load_or_create_vector_store(self):
        """โหลด vector store หรือสร้างใหม่ถ้ายังไม่มี"""
        try:
            # ลองโหลด existing vector store
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            logger.info(f"Loaded existing vector store: {self.collection_name}")
        except Exception as e:
            logger.info(f"Creating new vector store: {self.collection_name}")
            self.vector_store = None

    def add_documents(self, documents: List[Document]) -> bool:
        """
        เพิ่มเอกสารเข้า vector store

        Args:
            documents: รายการเอกสาร

        Returns:
            bool: สำเร็จหรือไม่
        """
        try:
            if not documents:
                logger.warning("No documents to add")
                return False

            if self.vector_store is None:
                # สร้าง vector store ใหม่
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    collection_name=self.collection_name,
                    persist_directory=self.persist_directory
                )
            else:
                # เพิ่มเข้า existing vector store
                self.vector_store.add_documents(documents)

            logger.info(f"Added {len(documents)} documents to vector store")
            return True

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False

    def similarity_search(
        self,
        query: str,
        k: int = None,
        filter_dict: Optional[dict] = None
    ) -> List[Document]:
        """
        ค้นหาเอกสารที่คล้ายกับ query

        Args:
            query: คำค้นหา
            k: จำนวนผลลัพธ์ (default จาก config)
            filter_dict: เงื่อนไขการกรอง

        Returns:
            List[Document]: รายการเอกสารที่เกี่ยวข้อง
        """
        if self.vector_store is None:
            logger.warning("Vector store is empty")
            return []

        k = k or config.TOP_K_RESULTS

        try:
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter_dict
            )
            logger.info(f"Found {len(results)} similar documents")
            return results

        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []

    def similarity_search_with_score(
        self,
        query: str,
        k: int = None,
        filter_dict: Optional[dict] = None
    ) -> List[tuple]:
        """
        ค้นหาเอกสารพร้อมคะแนนความคล้าย

        Args:
            query: คำค้นหา
            k: จำนวนผลลัพธ์
            filter_dict: เงื่อนไขการกรอง

        Returns:
            List[tuple]: รายการ (Document, score)
        """
        if self.vector_store is None:
            logger.warning("Vector store is empty")
            return []

        k = k or config.TOP_K_RESULTS

        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict
            )
            logger.info(f"Found {len(results)} similar documents with scores")
            return results

        except Exception as e:
            logger.error(f"Error in similarity search with score: {e}")
            return []

    def delete_collection(self):
        """ลบ collection ทั้งหมด"""
        try:
            if self.vector_store is not None:
                self.vector_store.delete_collection()
                self.vector_store = None
                logger.info(f"Deleted collection: {self.collection_name}")
                return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False

    def get_collection_info(self) -> dict:
        """
        ดูข้อมูล collection

        Returns:
            dict: ข้อมูล collection
        """
        try:
            if self.vector_store is None:
                return {"status": "empty"}

            collection = self.vector_store._collection
            return {
                "name": self.collection_name,
                "count": collection.count(),
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}


def create_vector_store_from_documents(documents: List[Document]) -> VectorStoreManager:
    """
    สร้าง vector store จากเอกสาร

    Args:
        documents: รายการเอกสาร

    Returns:
        VectorStoreManager: vector store manager
    """
    manager = VectorStoreManager()
    manager.add_documents(documents)
    return manager


def main():
    """Test function for standalone usage"""
    print("🧪 Testing Vector Store...")
    
    try:
        # Test 1: Initialize vector store
        vector_store = VectorStoreManager()
        print("✓ Vector store initialized")
        
        # Test 2: Check collection info
        info = vector_store.get_collection_info()
        print(f"✓ Collection: {info.get('name', 'N/A')}")
        print(f"✓ Document count: {info.get('count', 0)}")
        
        # Test 3: Test similarity search
        if info.get('count', 0) > 0:
            print("\n--- Testing Similarity Search ---")
            test_queries = [
                "ระบบควบคุมแบบวงเปิด",
                "PID Controller",
                "Transfer Function"
            ]
            
            for query in test_queries:
                results = vector_store.similarity_search(query, k=3)
                print(f"✓ Query '{query}': Found {len(results)} results")
                if results:
                    print(f"  Sample: {results[0].page_content[:50]}...")
        else:
            print("⚠️ No documents in vector store for testing search")
        
        print("\n✅ Vector Store Tests Completed!")
        
    except Exception as e:
        print(f"❌ Error in vector store test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
