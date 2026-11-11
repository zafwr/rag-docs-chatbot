"""Prompt used to make the model answer only from the retrieved sources."""

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about the user's documents.

Rules:
- Use ONLY the numbered sources provided. Do not use outside knowledge.
- After every sentence that uses a source, cite it with its number in square brackets, e.g. [1] or [2][3].
- If the sources do not contain the answer, say you could not find it in the documents. Do not guess.
- Answer in the same language as the question. Be concise and clear."""


def build_user_prompt(question: str, contexts: list[str]) -> str:
    numbered = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(contexts, start=1))
    return f"Sources:\n{numbered}\n\nQuestion: {question}"
