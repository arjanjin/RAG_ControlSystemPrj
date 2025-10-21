"""Main application for Control System RAG."""
import os
import sys
from dotenv import load_dotenv

from src.rag_system import ControlSystemRAG


def print_separator():
    """Print a separator line."""
    print("\n" + "="*80 + "\n")


def main():
    """Run the Control System RAG application."""
    # Load environment variables
    load_dotenv()
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it in a .env file or export it in your environment.")
        print("\nExample .env file:")
        print("OPENAI_API_KEY=your-api-key-here")
        sys.exit(1)
    
    print("Control System RAG - Examination Assistant")
    print_separator()
    
    # Initialize RAG system
    print("Initializing RAG system...")
    try:
        rag = ControlSystemRAG(
            documents_dir="documents",
            persist_directory="chroma_db"
        )
        rag.initialize(force_reload=False)
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        sys.exit(1)
    
    print_separator()
    print("RAG system initialized successfully!")
    print("You can now ask questions about control systems.")
    print("Type 'quit' or 'exit' to stop.")
    print_separator()
    
    # Interactive question-answering loop
    while True:
        try:
            question = input("\nYour question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            print("\nSearching for answer...")
            result = rag.query(question)
            
            print_separator()
            print("Answer:")
            print(result["answer"])
            
            if result["sources"]:
                print("\n\nSources:")
                for i, doc in enumerate(result["sources"], 1):
                    print(f"\n{i}. {doc.metadata.get('source', 'Unknown source')}")
                    print(f"   Content preview: {doc.page_content[:200]}...")
            
            print_separator()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError processing question: {e}")


if __name__ == "__main__":
    main()
