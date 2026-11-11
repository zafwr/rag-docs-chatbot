"""Split page text into overlapping chunks that fit comfortably in a prompt."""

import re
from dataclasses import dataclass

from .loaders import Page


@dataclass
class Chunk:
    text: str
    page: int


def _normalize(text: str) -> str:
    # PDFs often break lines mid-sentence; collapse whitespace but keep paragraph breaks.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_pages(pages: list[Page], chunk_size: int, overlap: int) -> list[Chunk]:
    """Split each page into windows of about `chunk_size` characters.

    Consecutive chunks share `overlap` characters so a sentence cut at a boundary
    still appears whole in at least one chunk. Cuts prefer a sentence or line end.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    chunks: list[Chunk] = []
    for page in pages:
        text = _normalize(page.text)
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                # Look back for a natural break in the last 30% of the window.
                window = text[start:end]
                cut = max(window.rfind(". "), window.rfind("\n"), window.rfind("? "), window.rfind("! "))
                if cut > chunk_size * 0.7:
                    end = start + cut + 1
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(text=piece, page=page.number))
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks
