"""Setup configuration for RAG Control System."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="rag-control-system",
    version="0.1.0",
    author="RAG Control System Team",
    description="GPU-accelerated RAG system for examination control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arjanjin/RAG_ControlSystemPrj",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.6.0",
        "transformers>=4.48.0",
        "sentence-transformers>=2.2.0",
        "langchain>=0.1.0",
        "chromadb>=0.4.0",
        "accelerate>=0.20.0",
        "faiss-gpu>=1.7.2",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.65.0",
        "fastapi>=0.109.1",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "cpu": [
            "faiss-cpu>=1.7.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "rag-system=src.rag_system:main",
            "rag-example=example:main",
        ],
    },
)
