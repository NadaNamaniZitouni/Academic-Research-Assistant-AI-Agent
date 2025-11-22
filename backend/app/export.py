"""
Export functionality for answers and citations
"""
from typing import Dict, List
from datetime import datetime
import json


def generate_bibtex_entry(doc: Dict) -> str:
    """Generate a BibTeX entry for a document"""
    # Extract year if available
    year = doc.get('year') or datetime.now().year
    
    # Generate citation key from title or doc_id
    title = doc.get('title', 'Untitled')
    citation_key = title.replace(' ', '_').replace(':', '').replace(',', '')[:50]
    if not citation_key:
        citation_key = doc.get('doc_id', 'unknown')[:8]
    
    # Build BibTeX entry
    bibtex = f"@article{{{citation_key},\n"
    bibtex += f"  title = {{{title}}},\n"
    
    authors = doc.get('authors', '')
    if authors:
        bibtex += f"  author = {{{authors}}},\n"
    
    bibtex += f"  year = {{{year}}},\n"
    
    doi = doc.get('doi', '')
    if doi:
        bibtex += f"  doi = {{{doi}}},\n"
    
    bibtex += f"  note = {{Retrieved from Academic Research Assistant}}\n"
    bibtex += "}\n"
    
    return bibtex


def format_answer_for_export(answer: str, sources: List[Dict]) -> str:
    """Format answer with citations for export"""
    formatted = f"{answer}\n\n"
    formatted += "=" * 60 + "\n"
    formatted += "SOURCES\n"
    formatted += "=" * 60 + "\n\n"
    
    for i, source in enumerate(sources, 1):
        formatted += f"Source {i}:\n"
        formatted += f"  Document: {source.get('doc_title', 'Unknown')}\n"
        formatted += f"  Pages: {source.get('page_start', '?')}-{source.get('page_end', '?')}\n"
        formatted += f"  Text: {source.get('text', '')[:200]}...\n"
        formatted += f"  Similarity: {source.get('similarity_score', 0):.3f}\n\n"
    
    return formatted


def generate_markdown_export(answer: str, sources: List[Dict], question: str) -> str:
    """Generate Markdown formatted export"""
    md = f"# Research Query Result\n\n"
    md += f"**Question:** {question}\n\n"
    md += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md += "---\n\n"
    md += f"## Answer\n\n{answer}\n\n"
    md += "---\n\n"
    md += "## Sources\n\n"
    
    for i, source in enumerate(sources, 1):
        md += f"### Source {i}\n\n"
        md += f"- **Document:** {source.get('doc_title', 'Unknown')}\n"
        md += f"- **Pages:** {source.get('page_start', '?')}-{source.get('page_end', '?')}\n"
        md += f"- **Similarity Score:** {source.get('similarity_score', 0):.3f}\n\n"
        md += f"**Excerpt:**\n\n"
        md += f"> {source.get('text', '')}\n\n"
        md += "---\n\n"
    
    return md

