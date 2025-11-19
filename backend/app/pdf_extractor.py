from pathlib import Path
from typing import Optional, Tuple
import pypdf
from pdfminer.high_level import extract_text as pdfminer_extract
import re


def extract_text_from_pdf(pdf_path: Path, method: str = "pypdf") -> Tuple[str, dict]:
    """
    Extract text from PDF and return text with metadata.

    Args:
        pdf_path: Path to PDF file
        method: "pypdf" or "pdfminer"

    Returns:
        Tuple of (text, metadata_dict)
    """
    metadata = {
        "title": None,
        "authors": None,
        "year": None,
        "doi": None,
        "num_pages": 0
    }

    try:
        if method == "pypdf":
            with open(pdf_path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)
                metadata["num_pages"] = len(pdf_reader.pages)

                # Extract metadata
                if pdf_reader.metadata:
                    metadata["title"] = pdf_reader.metadata.get("/Title", "")
                    metadata["authors"] = pdf_reader.metadata.get("/Author", "")
                    if "/CreationDate" in pdf_reader.metadata:
                        date_str = pdf_reader.metadata["/CreationDate"]
                        year_match = re.search(r"(\d{4})", date_str)
                        if year_match:
                            metadata["year"] = int(year_match.group(1))

                # Extract text page by page
                text_pages = []
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    text_pages.append((page_num, page_text))
                
                # If no title in metadata, try to extract from first page
                if not metadata["title"] or len(metadata["title"].strip()) < 3:
                    if text_pages and len(text_pages) > 0:
                        first_page_text = text_pages[0][1]
                        # Try to find title-like text (first line, or text before first newline)
                        lines = first_page_text.split("\n")
                        for line in lines[:5]:  # Check first 5 lines
                            line = line.strip()
                            if len(line) > 10 and len(line) < 200:  # Reasonable title length
                                # Skip common non-title patterns
                                if not line.lower().startswith(("abstract", "introduction", "keywords", "doi")):
                                    metadata["title"] = line
                                    break

                full_text = "\n\n".join([text for _, text in text_pages])

        else:  # pdfminer
            full_text = pdfminer_extract(str(pdf_path))
            # Try to get page count with pypdf
            try:
                with open(pdf_path, "rb") as file:
                    pdf_reader = pypdf.PdfReader(file)
                    metadata["num_pages"] = len(pdf_reader.pages)
            except:
                metadata["num_pages"] = len(full_text.split("\f"))

    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")

    # Clean text
    full_text = clean_text(full_text)

    # Try to extract DOI from text
    doi_match = re.search(r"doi[:\s]+([0-9.]+/[^\s]+)", full_text, re.IGNORECASE)
    if doi_match:
        metadata["doi"] = doi_match.group(1)

    return full_text, metadata


def clean_text(text: str) -> str:
    """Clean extracted text"""
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove page breaks
    text = text.replace("\f", "\n")
    # Remove excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove or clean template variables (optional - might want to keep them in some cases)
    # text = re.sub(r"\{[^}]+\}", "", text)  # Uncomment to remove template variables
    return text.strip()


def extract_text_by_page(pdf_path: Path) -> list[Tuple[int, str]]:
    """Extract text page by page for better chunking with page numbers"""
    pages = []
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = pypdf.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                text = clean_text(text)
                pages.append((page_num, text))
    except Exception as e:
        raise ValueError(f"Failed to extract pages: {str(e)}")
    return pages

