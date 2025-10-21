from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rag-control-system",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="RAG for examination control system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arjanjin/RAG_ControlSystemPrj",
    packages=find_packages(exclude=["tests*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        "openai>=1.0.0",
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "fastapi>=0.109.0",
        "uvicorn>=0.27.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
)
