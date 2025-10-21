"""Example script demonstrating programmatic usage of the RAG system."""
import os
from dotenv import load_dotenv
from src.rag_system import ControlSystemRAG


def main():
    """Run example queries."""
    # Load environment variables
    load_dotenv()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set in environment")
        print("Please create a .env file with your OpenAI API key")
        return
    
    print("Initializing Control System RAG...")
    
    # Initialize the RAG system
    rag = ControlSystemRAG(
        documents_dir="documents",
        persist_directory="chroma_db"
    )
    
    # Initialize (will load existing vector store or create new one)
    rag.initialize(force_reload=False)
    
    print("\nRAG system initialized successfully!\n")
    print("="*80)
    
    # Example questions
    questions = [
        "What is the difference between open-loop and closed-loop control systems?",
        "How does the integral component of a PID controller work?",
        "What is the Routh-Hurwitz stability criterion?",
        "What are the advantages of PID controllers?",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 80)
        
        # Get answer
        result = rag.query(question)
        
        print(f"\nAnswer: {result['answer']}")
        
        # Show sources
        if result['sources']:
            print(f"\nSources used ({len(result['sources'])} documents):")
            for j, doc in enumerate(result['sources'], 1):
                source = doc.metadata.get('source', 'Unknown')
                preview = doc.page_content[:150].replace('\n', ' ')
                print(f"  {j}. {source}: {preview}...")
        
        print("\n" + "="*80)
    
    print("\n✅ Example completed successfully!")


if __name__ == "__main__":
    main()
