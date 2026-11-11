import pytest

from app.chunking import chunk_pages
from app.loaders import Page, UnsupportedFileError, load_pages


def test_short_page_becomes_one_chunk():
    chunks = chunk_pages([Page(1, "Hello world.")], chunk_size=100, overlap=10)
    assert [c.text for c in chunks] == ["Hello world."]
    assert chunks[0].page == 1


def test_long_text_is_split_with_overlap_and_keeps_page_numbers():
    text = " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = chunk_pages([Page(3, text)], chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    assert all(c.page == 3 for c in chunks)
    # Every sentence survives chunking.
    joined = " ".join(c.text for c in chunks)
    assert all(f"Sentence number {i} is here." in joined for i in range(60))


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        chunk_pages([Page(1, "text")], chunk_size=100, overlap=100)


def test_pdf_pages_are_extracted(sample_pdf):
    pages = load_pages("handbook.pdf", sample_pdf)
    assert [p.number for p in pages] == [1, 2, 3]
    assert "20 days of paid annual leave" in pages[1].text


def test_text_files_are_supported():
    pages = load_pages("notes.md", "# Title\nSome notes".encode())
    assert len(pages) == 1 and "Some notes" in pages[0].text


def test_unsupported_file_type_is_rejected():
    with pytest.raises(UnsupportedFileError):
        load_pages("image.png", b"\x89PNG")
