from types import SimpleNamespace

import numpy as np
import pytest

from app.config import Settings
from app.providers import GeminiChatModel, GeminiEmbedder, build_providers


class FakeModels:
    def __init__(self):
        self.embed_calls = []
        self.generate_calls = []

    def embed_content(self, model, contents, config):
        self.embed_calls.append((model, list(contents), config.task_type))
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[3.0, 4.0]) for _ in contents])

    def generate_content(self, model, contents, config):
        self.generate_calls.append((model, contents, config))
        return SimpleNamespace(text="  The answer [1]  ")


def test_gemini_embedder_batches_and_normalizes():
    models = FakeModels()
    embedder = GeminiEmbedder(SimpleNamespace(models=models), "embed-model")
    vectors = embedder.embed_documents([f"t{i}" for i in range(120)])

    assert vectors.shape == (120, 2)
    assert np.allclose(vectors[0], [0.6, 0.8])  # unit length
    assert [len(c[1]) for c in models.embed_calls] == [50, 50, 20]
    assert models.embed_calls[0][2] == "RETRIEVAL_DOCUMENT"

    embedder.embed_query("q")
    assert models.embed_calls[-1][2] == "RETRIEVAL_QUERY"


def test_gemini_chat_sends_system_prompt_and_sources():
    models = FakeModels()
    answer = GeminiChatModel(SimpleNamespace(models=models), "chat-model").answer("Q?", ["ctx"])

    assert answer == "The answer [1]"
    _, contents, config = models.generate_calls[0]
    assert "[1] ctx" in contents
    assert "ONLY the numbered sources" in config.system_instruction


def test_missing_api_key_gives_a_clear_error():
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        build_providers(Settings(llm_provider="gemini", gemini_api_key="", _env_file=None))
