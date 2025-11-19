from typing import List, Dict
import re
from pathlib import Path


def validate_pdf(file_path: Path) -> bool:
    """Validate that file is a PDF"""
    if not file_path.exists():
        return False
    if not file_path.suffix.lower() == '.pdf':
        return False
    # Could add more validation (check PDF header)
    return True


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for storage"""
    # Remove path components
    filename = Path(filename).name
    # Remove dangerous characters
    filename = re.sub(r'[^\w\-_\.]', '_', filename)
    return filename


def format_citation(doc_title: str, page_start: int, page_end: int) -> str:
    """Format citation string"""
    if page_start == page_end:
        return f"[{doc_title}, p{page_start}]"
    return f"[{doc_title}, p{page_start}-{page_end}]"


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to max length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

