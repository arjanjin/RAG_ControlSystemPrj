#!/usr/bin/env python
"""Example script demonstrating GPU-accelerated RAG system usage."""

import logging
from src.rag_system import RAGSystem
from src.gpu_config import GPUConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Run example RAG system with GPU support."""
    
    print("=" * 60)
    print("RAG Control System - GPU Example")
    print("=" * 60)
    
    # Step 1: Check GPU configuration
    print("\n[1] Checking GPU Configuration...")
    gpu_config = GPUConfig()
    gpu_info = gpu_config.get_device_info()
    
    print("\nGPU Information:")
    for key, value in gpu_info.items():
        print(f"  {key}: {value}")
    
    # Step 2: Initialize RAG system
    print("\n[2] Initializing RAG System...")
    rag = RAGSystem(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        memory_fraction=0.8
    )
    
    # Step 3: Add sample documents
    print("\n[3] Adding Sample Documents...")
    sample_documents = [
        "The examination control system manages all aspects of student assessments including scheduling, grading, and reporting.",
        "Students must register for examinations at least two weeks before the exam date.",
        "The grading rubric includes criteria for content knowledge, critical thinking, and presentation skills.",
        "GPU acceleration significantly improves the performance of machine learning models by parallelizing computations.",
        "CUDA is NVIDIA's parallel computing platform that enables dramatic increases in computing performance.",
        "Examination results are typically released within 10 business days of the exam date.",
        "Students can appeal their grades within 14 days of result publication.",
        "The examination board meets quarterly to review assessment policies and procedures.",
        "All examination papers must be approved by at least two faculty members before administration.",
        "Special accommodations for students with disabilities must be requested in advance."
    ]
    
    rag.add_documents(sample_documents)
    print(f"Added {len(sample_documents)} documents to the system")
    
    # Step 4: Perform sample searches
    print("\n[4] Performing Sample Searches...")
    
    queries = [
        "How do I register for an examination?",
        "What is the role of GPU in machine learning?",
        "When are exam results published?",
        "What are the grading criteria?"
    ]
    
    for query in queries:
        print(f"\n{'─' * 60}")
        print(f"Query: {query}")
        print('─' * 60)
        
        results = rag.search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i} (Score: {result['score']:.4f}):")
            print(f"  {result['text']}")
    
    # Step 5: Display system information
    print("\n" + "=" * 60)
    print("[5] System Information")
    print("=" * 60)
    
    sys_info = rag.get_system_info()
    for key, value in sys_info.items():
        if key == 'gpu_info':
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    # Step 6: Cleanup
    print("\n[6] Cleaning up GPU cache...")
    rag.clear_cache()
    print("Done!")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
