from typing import List, Tuple
import tiktoken
from pathlib import Path


def get_tokenizer(model: str = "gpt-3.5-turbo"):
    """Get tiktoken tokenizer"""
    try:
        return tiktoken.encoding_for_model(model)
    except:
        return tiktoken.get_encoding("cl100k_base")


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 200,
    tokenizer_model: str = "gpt-3.5-turbo"
) -> List[str]:
    """
    Chunk text by tokens with overlap.

    Args:
        text: Text to chunk
        chunk_size: Target chunk size in tokens
        overlap: Overlap size in tokens
        tokenizer_model: Model name for tokenizer

    Returns:
        List of text chunks
    """
    tokenizer = get_tokenizer(tokenizer_model)
    tokens = tokenizer.encode(text)

    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)

        if i + chunk_size >= len(tokens):
            break
        i += chunk_size - overlap

    return chunks


def chunk_text_by_pages(
    pages: List[Tuple[int, str]],
    chunk_size: int = 800,
    overlap: int = 200
) -> List[dict]:
    """
    Chunk text with page information preserved.

    Args:
        pages: List of (page_num, text) tuples
        chunk_size: Target chunk size in tokens
        overlap: Overlap size in tokens

    Returns:
        List of dicts with 'text', 'page_start', 'page_end'
    """
    # Combine all pages
    full_text = "\n\n".join([text for _, text in pages])
    chunks = chunk_text(full_text, chunk_size, overlap)

    # Map chunks back to pages (simplified - could be more sophisticated)
    chunked_pages = []
    current_page = 1

    for chunk in chunks:
        # Estimate which pages this chunk spans
        # This is approximate - for exact mapping, track character positions
        chunk_start_page = current_page
        # Estimate end page based on text length
        estimated_pages = max(1, len(chunk) // 2000)  # Rough estimate
        chunk_end_page = min(current_page + estimated_pages - 1, len(pages))

        chunked_pages.append({
            "text": chunk,
            "page_start": chunk_start_page,
            "page_end": chunk_end_page
        })

        current_page = max(1, chunk_end_page - (overlap // 100))  # Rough estimate

    return chunked_pages

