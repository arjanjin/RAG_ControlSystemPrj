#!/usr/bin/env python3
"""
Main Script for RAG Control System Exam Grader
ระบบตรวจข้อสอบวิชาระบบควบคุมด้วย RAG
"""

import sys
import json
import argparse
import logging
from pathlib import Path

from src.rag_engine import RAGEngine
from src.exam_grader import ExamGrader
from src.document_loader import DocumentLoader
from src.vector_store import VectorStoreManager
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_knowledge_base():
    """โหลดฐานความรู้เข้า vector database"""
    logger.info("Loading knowledge base...")

    # สร้าง document loader
    loader = DocumentLoader()
    documents = loader.load_directory(str(config.KNOWLEDGE_BASE_DIR))

    if not documents:
        logger.warning("No documents found in knowledge base!")
        return None

    # สร้าง vector store และเพิ่มเอกสาร
    vector_store = VectorStoreManager()
    vector_store.add_documents(documents)

    logger.info(f"Successfully loaded {len(documents)} document chunks")
    return vector_store


def query_mode(vector_store: VectorStoreManager):
    """โหมดถามตอบ"""
    logger.info("Starting Query Mode - Type 'exit' to quit")

    rag_engine = RAGEngine(vector_store_manager=vector_store)

    print("\n" + "=" * 60)
    print("RAG Control System Q&A")
    print("=" * 60)

    while True:
        try:
            question = input("\nคำถาม: ").strip()

            if question.lower() in ['exit', 'quit', 'q']:
                break

            if not question:
                continue

            # Query RAG system
            result = rag_engine.query(question, return_context=True)

            print("\nคำตอบ:")
            print("-" * 60)
            print(result['answer'])
            print("-" * 60)
            print(f"จำนวนแหล่งอ้างอิง: {result['num_sources']}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error: {e}")

    print("\nGoodbye!")


def grade_exam_mode(vector_store: VectorStoreManager, exam_file: str, output_file: str = None):
    """โหมดตรวจข้อสอบ"""
    logger.info(f"Grading exam from {exam_file}")

    # โหลดข้อสอบ
    try:
        with open(exam_file, 'r', encoding='utf-8') as f:
            exam_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading exam file: {e}")
        return

    # สร้าง exam grader
    rag_engine = RAGEngine(vector_store_manager=vector_store)
    grader = ExamGrader(rag_engine)

    # ตรวจข้อสอบ
    print("\n" + "=" * 60)
    print("กำลังตรวจข้อสอบ...")
    print("=" * 60)

    results = grader.grade_exam(exam_data)

    # สร้างรายงาน
    report = grader.generate_feedback_report(results)
    print(report)

    # บันทึกผลลัพธ์
    if output_file:
        try:
            # บันทึกผลลัพธ์ JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {output_file}")

            # บันทึกรายงาน TXT
            report_file = output_file.replace('.json', '_report.txt')
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Report saved to {report_file}")

        except Exception as e:
            logger.error(f"Error saving results: {e}")


def initialize_system():
    """เริ่มต้นระบบ"""
    print("\n" + "=" * 60)
    print("RAG Control System Exam Grader")
    print("ระบบตรวจข้อสอบวิชาระบบควบคุมด้วย RAG")
    print("=" * 60)
    print()

    # ตรวจสอบโฟลเดอร์
    for directory in [config.KNOWLEDGE_BASE_DIR, config.EXAMS_DIR, config.VECTOR_DB_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)

    # โหลดฐานความรู้
    vector_store = load_knowledge_base()

    if vector_store is None:
        logger.error("Failed to load knowledge base!")
        sys.exit(1)

    # แสดงสถานะ
    info = vector_store.get_collection_info()
    print(f"\nVector Store Status:")
    print(f"  Collection: {info.get('name')}")
    print(f"  Documents: {info.get('count')}")
    print(f"  LLM Provider: {config.LLM_PROVIDER}")
    print()

    return vector_store


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='RAG Control System Exam Grader',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query mode
  python main.py --query

  # Grade exam
  python main.py --grade examples/sample_exam.json --output results.json

  # Reload knowledge base
  python main.py --reload
        """
    )

    parser.add_argument('--query', '-q', action='store_true',
                        help='Start query mode (Q&A)')
    parser.add_argument('--grade', '-g', type=str,
                        help='Grade exam from JSON file')
    parser.add_argument('--output', '-o', type=str,
                        help='Output file for grading results')
    parser.add_argument('--reload', '-r', action='store_true',
                        help='Reload knowledge base')

    args = parser.parse_args()

    # Initialize system
    vector_store = initialize_system()

    # Execute mode
    if args.query:
        query_mode(vector_store)
    elif args.grade:
        output_file = args.output or 'results.json'
        grade_exam_mode(vector_store, args.grade, output_file)
    else:
        # Default: show help and enter query mode
        parser.print_help()
        print("\nStarting query mode by default...")
        query_mode(vector_store)


if __name__ == "__main__":
    main()
