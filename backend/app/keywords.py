from typing import List
import re
from collections import Counter


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """Extract keywords from text using simple frequency analysis"""
    # Remove common stop words
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "can", "this",
        "that", "these", "those", "it", "its", "they", "them", "their"
    }

    # Extract words (alphanumeric, at least 3 chars)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

    # Filter stop words and count
    keywords = [w for w in words if w not in stop_words]
    counter = Counter(keywords)

    # Return top N
    return [word for word, _ in counter.most_common(top_n)]


def extract_keyphrases(text: str, n: int = 3) -> List[str]:
    """Extract n-gram keyphrases"""
    words = text.lower().split()
    keyphrases = []

    for i in range(len(words) - n + 1):
        phrase = " ".join(words[i:i+n])
        keyphrases.append(phrase)

    # Return most common
    counter = Counter(keyphrases)
    return [phrase for phrase, _ in counter.most_common(10)]

