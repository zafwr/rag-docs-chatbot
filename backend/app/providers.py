"""Embedding and chat model providers.

The rest of the app only talks to the two small interfaces below, so swapping
Gemini for OpenAI, Claude or a local model means adding one class here.
"""

import hashlib
import re
from typing import Protocol

import numpy as np

from .config import Settings
from .prompts import SYSTEM_PROMPT, build_user_prompt


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class ChatModel(Protocol):
    def answer(self, question: str, contexts: list[str]) -> str: ...


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


# --- Gemini ---------------------------------------------------------------


class GeminiEmbedder:
    BATCH_SIZE = 50

    def __init__(self, client, model: str):
        self._client = client
        self._model = model

    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        from google.genai import types

        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            response = self._client.models.embed_content(
                model=self._model,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            vectors.extend(embedding.values for embedding in response.embeddings)
        return _normalize_rows(np.array(vectors, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class GeminiChatModel:
    def __init__(self, client, model: str):
        self._client = client
        self._model = model

    def answer(self, question: str, contexts: list[str]) -> str:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self._model,
            contents=build_user_prompt(question, contexts),
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.2),
        )
        return (response.text or "").strip()


# --- Offline fake (tests and demos without an API key) --------------------

_WORD = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [token for token in _WORD.findall(text.lower()) if len(token) > 2]


class FakeEmbedder:
    """Hashed bag-of-words vectors: crude, but deterministic and offline."""

    DIMENSIONS = 512

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.DIMENSIONS, dtype=np.float32)
        for token in _tokens(text):
            index = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.DIMENSIONS
            vector[index] += 1.0
        return vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.DIMENSIONS), dtype=np.float32)
        return _normalize_rows(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class FakeChatModel:
    """Answers by quoting the sentence from the best source that shares the most words."""

    def answer(self, question: str, contexts: list[str]) -> str:
        if not contexts:
            return "I could not find this in the documents."
        query = set(_tokens(question))
        best_sentence, best_source, best_overlap = contexts[0].split(". ")[0], 1, -1
        for source, text in enumerate(contexts, start=1):
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
                overlap = len(query & set(_tokens(sentence)))
                if overlap > best_overlap:
                    best_sentence, best_source, best_overlap = sentence.strip(), source, overlap
        return f"{best_sentence} [{best_source}]"


def build_providers(settings: Settings) -> tuple[Embedder, ChatModel]:
    if settings.llm_provider == "fake":
        return FakeEmbedder(), FakeChatModel()
    if settings.llm_provider == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to backend/.env or set LLM_PROVIDER=fake.")
        from google import genai

        client = genai.Client(api_key=settings.gemini_api_key)
        return (
            GeminiEmbedder(client, settings.gemini_embedding_model),
            GeminiChatModel(client, settings.gemini_chat_model),
        )
    raise RuntimeError(f"Unknown LLM_PROVIDER '{settings.llm_provider}'. Use 'gemini' or 'fake'.")
