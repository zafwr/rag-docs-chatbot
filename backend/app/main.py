"""FastAPI app exposing upload, document list, delete and chat endpoints."""

from dataclasses import asdict
from functools import lru_cache

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .loaders import UnsupportedFileError
from .providers import build_providers
from .rag import EmptyDocumentError, RagService
from .vector_store import VectorStore

app = FastAPI(title="RAG Docs Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache
def get_rag_service() -> RagService:
    settings = get_settings()
    embedder, chat_model = build_providers(settings)
    return RagService(settings, embedder, chat_model, VectorStore(settings.data_dir))


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.get("/api/health")
def health(settings: Settings = Depends(get_settings)) -> dict:
    return {"status": "ok", "provider": settings.llm_provider}


@app.post("/api/documents", status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    rag: RagService = Depends(get_rag_service),
    settings: Settings = Depends(get_settings),
) -> dict:
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {settings.max_upload_mb} MB.")
    try:
        return rag.ingest(file.filename or "document", data)
    except (UnsupportedFileError, EmptyDocumentError) as error:
        raise HTTPException(400, str(error)) from error


@app.get("/api/documents")
def list_documents(rag: RagService = Depends(get_rag_service)) -> list[dict]:
    return rag.documents()


@app.delete("/api/documents/{document_id}", status_code=204)
def delete_document(document_id: str, rag: RagService = Depends(get_rag_service)) -> None:
    if not rag.delete(document_id):
        raise HTTPException(404, "Document not found.")


@app.post("/api/chat")
def chat(request: ChatRequest, rag: RagService = Depends(get_rag_service)) -> dict:
    return asdict(rag.ask(request.question.strip()))
