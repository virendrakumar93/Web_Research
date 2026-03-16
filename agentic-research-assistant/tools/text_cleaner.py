"""
Text Cleaner Tool
=================

Purpose:
    Cleans and preprocesses raw scraped text for LLM consumption.
    Handles whitespace normalization, encoding issues, and text
    chunking for long documents.

Inputs:
    - text: Raw text string to clean.
    - chunk_size / chunk_overlap: Parameters for splitting long texts.

Outputs:
    - Cleaned text string, or list of text chunks.

Role in Architecture:
    Utility tool used between the scraper and extraction agent to
    ensure text quality before LLM processing.
"""

import re
import unicodedata


def clean_text(text: str) -> str:
    """Clean raw scraped text by normalizing whitespace and encoding.

    Args:
        text: Raw text string.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # Normalize unicode characters
    text = unicodedata.normalize("NFKD", text)

    # Remove null bytes and control characters (keep newlines and tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # Normalize whitespace: collapse multiple spaces/tabs into single space
    text = re.sub(r"[^\S\n]+", " ", text)

    # Collapse multiple consecutive newlines into at most two
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)

    # Remove lines that are just punctuation or very short noise
    lines = text.splitlines()
    cleaned_lines = [line for line in lines if len(line) > 2 or line == ""]
    text = "\n".join(cleaned_lines)

    return text.strip()


def chunk_text(text: str, chunk_size: int = 2000, chunk_overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks for processing.

    Args:
        text: The text to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of text chunks.
    """
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at a paragraph or sentence boundary
        if end < len(text):
            # Look for paragraph break
            para_break = text.rfind("\n\n", start + chunk_size // 2, end)
            if para_break != -1:
                end = para_break + 2
            else:
                # Look for sentence break
                sentence_break = text.rfind(". ", start + chunk_size // 2, end)
                if sentence_break != -1:
                    end = sentence_break + 2

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - chunk_overlap

    return chunks


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to a maximum length at a word boundary.

    Args:
        text: Text to truncate.
        max_length: Maximum character length.

    Returns:
        Truncated text.
    """
    if len(text) <= max_length:
        return text

    truncated = text[:max_length]
    last_space = truncated.rfind(" ")
    if last_space > max_length * 0.8:
        truncated = truncated[:last_space]

    return truncated + "..."
