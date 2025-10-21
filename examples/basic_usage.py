"""
Basic usage example for RAG Control System.
"""

from rag_control_system import RAGEngine

# Sample documents for examination control
sample_documents = [
    {
        "id": "1",
        "content": "Students must arrive 15 minutes before the examination start time. "
        "Late arrivals may not be permitted to take the exam.",
        "title": "Exam Arrival Policy",
    },
    {
        "id": "2",
        "content": "All electronic devices including phones, smartwatches, and calculators "
        "must be turned off and stored in bags during examinations.",
        "title": "Electronic Device Policy",
    },
    {
        "id": "3",
        "content": "Students are allowed to bring water in clear bottles and may request "
        "additional answer sheets from the invigilator during the exam.",
        "title": "Permitted Items",
    },
    {
        "id": "4",
        "content": "The examination hall opens 30 minutes before the scheduled start time. "
        "Students should check the seating plan posted at the entrance.",
        "title": "Examination Hall Procedures",
    },
    {
        "id": "5",
        "content": "In case of illness or emergency, students must contact the examination "
        "office immediately and provide medical documentation within 48 hours.",
        "title": "Emergency Procedures",
    },
]


def main():
    """Run basic usage example."""
    print("=" * 80)
    print("RAG Control System - Basic Usage Example")
    print("=" * 80)

    # Initialize the RAG engine
    print("\n1. Initializing RAG Engine...")
    engine = RAGEngine()

    # Index the sample documents
    print("2. Indexing sample documents...")
    engine.index_documents(sample_documents)
    print(f"   Indexed {len(sample_documents)} documents")

    # Example queries
    queries = [
        "What time should students arrive for exams?",
        "Can I bring my phone to the examination?",
        "What should I do if I'm sick on exam day?",
    ]

    print("\n3. Processing example queries...\n")

    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 80)

        result = engine.query(query)

        print(f"Answer: {result['answer']}")
        print(f"Documents retrieved: {result['metadata']['documents_retrieved']}")

    # Show configuration
    print("\n" + "=" * 80)
    print("4. Current Configuration:")
    print("-" * 80)
    config = engine.get_config()
    print(f"Retriever Model: {config['retriever']['embedding_model']}")
    print(f"Generator Model: {config['generator']['model_name']}")
    print(f"Top-K Documents: {config['retriever']['top_k']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
