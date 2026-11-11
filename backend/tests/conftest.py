from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app, get_rag_service
from app.providers import FakeChatModel, FakeEmbedder
from app.rag import RagService
from app.vector_store import VectorStore

SAMPLE_PDF = Path(__file__).resolve().parents[2] / "sample_docs" / "lumen-labs-handbook.pdf"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(llm_provider="fake", data_dir=tmp_path / "data", chunk_size=300, chunk_overlap=50, min_score=0.05)


@pytest.fixture
def rag(settings) -> RagService:
    return RagService(settings, FakeEmbedder(), FakeChatModel(), VectorStore(settings.data_dir))


@pytest.fixture
def client(settings, rag):
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_rag_service] = lambda: rag
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def sample_pdf() -> bytes:
    return SAMPLE_PDF.read_bytes()
