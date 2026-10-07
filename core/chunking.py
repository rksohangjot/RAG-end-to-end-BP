"""Chunking: split ingested document text into bounded, overlapping pieces.

Character-based, boundary-aware splitting (paragraph > sentence > word > raw
character fallback). No tokenizer/embedding model is chosen yet, so chunk
size is measured in characters rather than tokens.
"""

from dataclasses import dataclass

from core.ingestion import Document

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50

# Tried in order: split on paragraph breaks first, falling back to sentence
# breaks, then words, then (if a single word is still too long) raw characters.
_SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int


def _split_into_units(text: str, separators: list[str], chunk_size: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    if not separators:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, rest = separators[0], separators[1:]
    if sep not in text:
        return _split_into_units(text, rest, chunk_size)

    parts = text.split(sep)
    units = [part + sep for part in parts[:-1]] + ([parts[-1]] if parts[-1] else [])

    result = []
    for unit in units:
        if len(unit) > chunk_size:
            result.extend(_split_into_units(unit, rest, chunk_size))
        else:
            result.append(unit)
    return result


def _overlap_tail(chunk: str, overlap: int) -> str:
    """Last ~overlap chars of chunk, trimmed forward to start on a word boundary."""
    if not overlap or len(chunk) <= overlap:
        return chunk if overlap else ""
    tail = chunk[-overlap:]
    if chunk[-overlap - 1].isspace() or tail[0].isspace():
        return tail
    # The slice starts mid-word: drop the partial word (empty if no boundary).
    for i, ch in enumerate(tail):
        if ch.isspace():
            return tail[i:]
    return ""


def _merge_units(units: list[str], chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    current = ""

    for unit in units:
        if len(current) + len(unit) <= chunk_size:
            current += unit
            continue

        if current:
            chunks.append(current)

        tail = _overlap_tail(chunks[-1], overlap) if chunks else ""
        current = tail + unit

    if current:
        chunks.append(current)
    return chunks


def chunk_document(
    document: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    text = document.text.strip()
    if not text:
        return []

    units = _split_into_units(text, _SEPARATORS, chunk_size)
    texts = _merge_units(units, chunk_size, overlap)

    return [
        Chunk(text=t, source=document.name, chunk_index=i) for i, t in enumerate(texts)
    ]
