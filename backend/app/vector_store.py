"""A small vector store: NumPy for search, JSON + .npy files for persistence.

For a few thousand chunks a brute-force cosine search is fast and has zero
infrastructure. For larger collections, swap this class for Chroma, pgvector
or Pinecone; it exposes the same few methods.
"""

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class StoredChunk:
    document_id: str
    document_name: str
    page: int
    text: str


@dataclass
class SearchResult:
    chunk: StoredChunk
    score: float


class VectorStore:
    def __init__(self, directory: Path):
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)
        self._chunks_path = directory / "chunks.json"
        self._vectors_path = directory / "vectors.npy"
        self._lock = threading.Lock()
        self._chunks: list[StoredChunk] = []
        self._vectors = np.zeros((0, 0), dtype=np.float32)
        self._load()

    # --- persistence ---

    def _load(self) -> None:
        if self._chunks_path.exists() and self._vectors_path.exists():
            raw = json.loads(self._chunks_path.read_text(encoding="utf-8"))
            self._chunks = [StoredChunk(**item) for item in raw]
            self._vectors = np.load(self._vectors_path)

    def _save(self) -> None:
        self._chunks_path.write_text(
            json.dumps([asdict(c) for c in self._chunks], ensure_ascii=False), encoding="utf-8"
        )
        np.save(self._vectors_path, self._vectors)

    # --- operations ---

    def add(self, chunks: list[StoredChunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return
        with self._lock:
            if self._vectors.size == 0:
                self._vectors = vectors.astype(np.float32)
            else:
                self._vectors = np.vstack([self._vectors, vectors.astype(np.float32)])
            self._chunks.extend(chunks)
            self._save()

    def search(self, query_vector: np.ndarray, top_k: int, min_score: float = 0.0) -> list[SearchResult]:
        with self._lock:
            if not self._chunks:
                return []
            # Vectors are L2-normalized, so the dot product is the cosine similarity.
            scores = self._vectors @ query_vector.astype(np.float32)
            order = np.argsort(-scores)[:top_k]
            return [
                SearchResult(chunk=self._chunks[i], score=float(scores[i]))
                for i in order
                if scores[i] >= min_score
            ]

    def delete_document(self, document_id: str) -> int:
        with self._lock:
            keep = [i for i, c in enumerate(self._chunks) if c.document_id != document_id]
            removed = len(self._chunks) - len(keep)
            if removed:
                self._chunks = [self._chunks[i] for i in keep]
                self._vectors = self._vectors[keep] if keep else np.zeros((0, 0), dtype=np.float32)
                self._save()
            return removed

    def documents(self) -> list[dict]:
        with self._lock:
            summary: dict[str, dict] = {}
            for chunk in self._chunks:
                entry = summary.setdefault(
                    chunk.document_id,
                    {"id": chunk.document_id, "name": chunk.document_name, "chunks": 0, "pages": set()},
                )
                entry["chunks"] += 1
                entry["pages"].add(chunk.page)
            return [{**d, "pages": len(d["pages"])} for d in summary.values()]
