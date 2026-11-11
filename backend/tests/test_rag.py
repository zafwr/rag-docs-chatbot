from app.prompts import build_user_prompt
from app.vector_store import VectorStore


def test_answer_cites_the_page_that_contains_it(rag, sample_pdf):
    rag.ingest("handbook.pdf", sample_pdf)
    result = rag.ask("How many days of paid annual leave do employees get?")

    assert "20 days" in result.answer
    assert "[" in result.answer  # cites a source number
    top = result.sources[0]
    assert top.document_name == "handbook.pdf"
    assert top.page == 2


def test_question_without_documents_returns_not_found(rag):
    result = rag.ask("What is the refund policy?")
    assert result.sources == []
    assert "could not find" in result.answer.lower()


def test_store_persists_to_disk_and_reloads(rag, settings, sample_pdf):
    rag.ingest("handbook.pdf", sample_pdf)
    reloaded = VectorStore(settings.data_dir)
    assert reloaded.documents()[0]["name"] == "handbook.pdf"


def test_delete_removes_document(rag, sample_pdf):
    doc = rag.ingest("handbook.pdf", sample_pdf)
    assert rag.delete(doc["id"]) is True
    assert rag.documents() == []
    assert rag.delete(doc["id"]) is False


def test_prompt_numbers_sources():
    prompt = build_user_prompt("Q?", ["first", "second"])
    assert "[1] first" in prompt and "[2] second" in prompt and prompt.endswith("Question: Q?")
