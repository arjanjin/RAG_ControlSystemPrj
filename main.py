"""
Main entry point for the RAG Control System.
"""

import argparse
import sys
from pathlib import Path

from rag_control_system import RAGEngine
from rag_control_system.config import get_settings
from rag_control_system.utils import setup_logger, load_documents


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Control System - Examination Management"
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="Query to process",
    )
    parser.add_argument(
        "--index",
        "-i",
        type=str,
        help="Path to documents file to index",
    )
    parser.add_argument(
        "--interactive",
        "-I",
        action="store_true",
        help="Start interactive mode",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger("rag_control_system", level=args.log_level)
    settings = get_settings()

    logger.info("Starting RAG Control System")
    logger.info(f"Configuration: {settings.app_name}")

    # Initialize RAG engine
    engine = RAGEngine()

    # Index documents if provided
    if args.index:
        index_path = Path(args.index)
        if not index_path.exists():
            logger.error(f"Index file not found: {args.index}")
            sys.exit(1)

        logger.info(f"Loading documents from {args.index}")
        try:
            documents = load_documents(args.index)
            engine.index_documents(documents)
            logger.info(f"Successfully indexed {len(documents)} documents")
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            sys.exit(1)

    # Process single query
    if args.query:
        logger.info(f"Processing query: {args.query}")
        result = engine.query(args.query)
        print("\n" + "=" * 80)
        print(f"Question: {result['question']}")
        print("-" * 80)
        print(f"Answer: {result['answer']}")
        print("=" * 80 + "\n")

        if result.get("documents"):
            print(f"Retrieved {len(result['documents'])} documents")

    # Interactive mode
    elif args.interactive:
        logger.info("Starting interactive mode")
        print("\nRAG Control System - Interactive Mode")
        print("Type 'exit' or 'quit' to end the session\n")

        while True:
            try:
                question = input("Your question: ").strip()

                if question.lower() in ["exit", "quit", "q"]:
                    print("Goodbye!")
                    break

                if not question:
                    continue

                result = engine.query(question)
                print("\n" + "-" * 80)
                print(f"Answer: {result['answer']}")
                print("-" * 80 + "\n")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"Error: {e}\n")

    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
