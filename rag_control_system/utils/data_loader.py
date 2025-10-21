"""
Data loading and saving utilities.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_documents(
    file_path: str,
    format: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load documents from a file.

    Args:
        file_path: Path to the file
        format: File format (json, csv, txt). If None, inferred from extension

    Returns:
        List of document dictionaries

    Raises:
        ValueError: If format is not supported
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Infer format from extension if not specified
    if format is None:
        format = path.suffix.lstrip(".")

    format = format.lower()

    if format == "json":
        return _load_json(path)
    elif format == "csv":
        return _load_csv(path)
    elif format == "txt":
        return _load_txt(path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_documents(
    documents: List[Dict[str, Any]],
    file_path: str,
    format: Optional[str] = None,
):
    """
    Save documents to a file.

    Args:
        documents: List of document dictionaries
        file_path: Path to save the file
        format: File format (json, csv). If None, inferred from extension

    Raises:
        ValueError: If format is not supported
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Infer format from extension if not specified
    if format is None:
        format = path.suffix.lstrip(".")

    format = format.lower()

    if format == "json":
        _save_json(documents, path)
    elif format == "csv":
        _save_csv(documents, path)
    else:
        raise ValueError(f"Unsupported format for saving: {format}")

    logger.info(f"Saved {len(documents)} documents to {file_path}")


def _load_json(path: Path) -> List[Dict[str, Any]]:
    """Load documents from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return [data]
    else:
        raise ValueError("JSON must contain a list or dict")


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    """Load documents from CSV file."""
    documents = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            documents.append(dict(row))
    return documents


def _load_txt(path: Path) -> List[Dict[str, Any]]:
    """Load documents from text file (each line is a document)."""
    documents = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                documents.append({"id": i, "content": line})
    return documents


def _save_json(documents: List[Dict[str, Any]], path: Path):
    """Save documents to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)


def _save_csv(documents: List[Dict[str, Any]], path: Path):
    """Save documents to CSV file."""
    if not documents:
        return

    fieldnames = list(documents[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(documents)
