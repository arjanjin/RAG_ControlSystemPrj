"""
Document Loader Module
โมดูลสำหรับโหลดและประมวลผลเอกสารความรู้
"""

from typing import List, Optional
from pathlib import Path
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    DirectoryLoader
)
from langchain.schema import Document

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


class DocumentLoader:
    """
    คลาสสำหรับโหลดเอกสารจากหลายแหล่ง
    รองรับ PDF, TXT, DOCX
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize document loader

        Args:
            chunk_size: ขนาดของแต่ละ chunk (default จาก config)
            chunk_overlap: ความทับซ้อนของ chunks (default จาก config)
        """
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""],
            length_function=len
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        """
        โหลดไฟล์ PDF

        Args:
            file_path: path ของไฟล์ PDF

        Returns:
            List[Document]: รายการเอกสารที่ถูกแบ่ง chunk
        """
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from {file_path}")
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            return []

    def load_text(self, file_path: str) -> List[Document]:
        """
        โหลดไฟล์ TXT

        Args:
            file_path: path ของไฟล์ TXT

        Returns:
            List[Document]: รายการเอกสารที่ถูกแบ่ง chunk
        """
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            logger.info(f"Loaded text from {file_path}")
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            logger.error(f"Error loading text {file_path}: {e}")
            return []

    def load_docx(self, file_path: str) -> List[Document]:
        """
        โหลดไฟล์ DOCX

        Args:
            file_path: path ของไฟล์ DOCX

        Returns:
            List[Document]: รายการเอกสารที่ถูกแบ่ง chunk
        """
        try:
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            logger.info(f"Loaded docx from {file_path}")
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            logger.error(f"Error loading docx {file_path}: {e}")
            return []

    def load_document(self, file_path: str) -> List[Document]:
        """
        โหลดเอกสาร ตรวจสอบนามสกุลไฟล์อัตโนมัติ

        Args:
            file_path: path ของไฟล์

        Returns:
            List[Document]: รายการเอกสารที่ถูกแบ่ง chunk
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == '.pdf':
            return self.load_pdf(file_path)
        elif suffix == '.txt':
            return self.load_text(file_path)
        elif suffix in ['.docx', '.doc']:
            return self.load_docx(file_path)
        else:
            logger.warning(f"Unsupported file type: {suffix}")
            return []

    def load_directory(self, directory_path: str, glob_pattern: str = "**/*") -> List[Document]:
        """
        โหลดเอกสารทั้งหมดในโฟลเดอร์

        Args:
            directory_path: path ของโฟลเดอร์
            glob_pattern: pattern สำหรับค้นหาไฟล์

        Returns:
            List[Document]: รายการเอกสารทั้งหมด
        """
        all_documents = []
        directory = Path(directory_path)

        if not directory.exists():
            logger.error(f"Directory not found: {directory_path}")
            return []

        # โหลดไฟล์แต่ละประเภท
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                docs = self.load_document(str(file_path))
                all_documents.extend(docs)

        logger.info(f"Loaded {len(all_documents)} chunks from {directory_path}")
        return all_documents


def load_knowledge_base() -> List[Document]:
    """
    โหลดฐานความรู้ทั้งหมดจากโฟลเดอร์ knowledge_base

    Returns:
        List[Document]: รายการเอกสารทั้งหมด
    """
    loader = DocumentLoader()
    documents = loader.load_directory(str(config.KNOWLEDGE_BASE_DIR))
    return documents


def main():
    """Test function for standalone usage"""
    print("🧪 Testing Document Loader...")
    
    # Test 1: Initialize document loader
    loader = DocumentLoader()
    print("✓ Document loader initialized")
    
    # Test 2: Load knowledge base
    print(f"Loading documents from: {config.KNOWLEDGE_BASE_DIR}")
    documents = load_knowledge_base()
    print(f"✓ Loaded {len(documents)} document chunks")
    
    # Test 3: Show sample documents
    if documents:
        print("\nSample documents:")
        for i, doc in enumerate(documents[:3]):
            print(f"  Document {i+1}: {doc.page_content[:100]}...")
            print(f"    Metadata: {doc.metadata}")
    
    print("✅ Document Loader Tests Completed!")


if __name__ == "__main__":
    main()
