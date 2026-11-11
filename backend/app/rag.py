"""The RAG pipeline: ingest documents, then answer questions from them."""

import uuid
from dataclasses import dataclass

from .chunking import chunk_pages
from .config import Settings
from .loaders import load_pages
from .providers import ChatModel, Embedder
from .vector_store import StoredChunk, VectorStore

NOT_FOUND_ANSWER = "I could not find anything about this in the uploaded documents."


class EmptyDocumentError(ValueError):
    pass


@dataclass
class Source:
    number: int
    document_name: str
    page: int
    text: str
    score: float


@dataclass
class Answer:
    answer: str
    sources: list[Source]


class RagService:
    def __init__(self, settings: Settings, embedder: Embedder, chat_model: ChatModel, store: VectorStore):
        self._settings = settings
        self._embedder = embedder
        self._chat_model = chat_model
        self._store = store

    def ingest(self, filename: str, data: bytes) -> dict:
        """Extract, chunk, embed and store a document. Returns a short summary."""
        pages = load_pages(filename, data)
        chunks = chunk_pages(pages, self._settings.chunk_size, self._settings.chunk_overlap)
        if not chunks:
            raise EmptyDocumentError(
                "No text could be extracted. Scanned PDFs need OCR before they can be searched."
            )

        document_id = uuid.uuid4().hex[:12]
        vectors = self._embedder.embed_documents([c.text for c in chunks])
        self._store.add(
            [StoredChunk(document_id, filename, c.page, c.text) for c in chunks],
            vectors,
        )
        return {"id": document_id, "name": filename, "pages": len(pages), "chunks": len(chunks)}

    def ask(self, question: str) -> Answer:
        """Retrieve the most relevant chunks and let the model answer from them only."""
        query_vector = self._embedder.embed_query(question)
        results = self._store.search(query_vector, self._settings.top_k, self._settings.min_score)
        if not results:
            return Answer(answer=NOT_FOUND_ANSWER, sources=[])

        sources = [
            Source(
                number=i,
                document_name=r.chunk.document_name,
                page=r.chunk.page,
                text=r.chunk.text,
                score=round(r.score, 3),
            )
            for i, r in enumerate(results, start=1)
        ]
        answer = self._chat_model.answer(question, [s.text for s in sources])
        return Answer(answer=answer or NOT_FOUND_ANSWER, sources=sources)

    def documents(self) -> list[dict]:
        return self._store.documents()

    def delete(self, document_id: str) -> bool:
        return self._store.delete_document(document_id) > 0
