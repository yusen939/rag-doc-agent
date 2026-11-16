from __future__ import annotations

import re

_SENTENCE_SEPARATORS = ("\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", "；", "; ")


def normalize_text(text: str) -> str:
    """Normalize line endings and excessive whitespace without dropping paragraphs."""
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _find_cut(text: str, start: int, chunk_size: int) -> int:
    hard_limit = min(len(text), start + chunk_size)
    if hard_limit >= len(text):
        return len(text)

    search_from = start + max(1, int(chunk_size * 0.55))
    best = -1
    for separator in _SENTENCE_SEPARATORS:
        position = text.rfind(separator, search_from, hard_limit)
        if position > best:
            best = position + len(separator)
    return best if best > start else hard_limit


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> list[str]:
    """Split text into overlapping chunks, preferring sentence boundaries.

    The implementation is character based so it works for both Chinese and English.
    """
    if chunk_size < 50:
        raise ValueError("chunk_size must be at least 50")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = _find_cut(text, start, chunk_size)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break

        next_start = max(start + 1, end - overlap)
        # Avoid starting a chunk in the middle of a Latin word when possible.
        if next_start > 0 and text[next_start - 1].isalnum() and text[next_start].isalnum():
            whitespace = text.find(" ", next_start, min(end, next_start + 80))
            if whitespace != -1:
                next_start = whitespace + 1
        start = next_start

    return chunks
