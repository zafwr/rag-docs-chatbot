"""Turn uploaded files into plain text, page by page."""

from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


class UnsupportedFileError(ValueError):
    pass


@dataclass
class Page:
    number: int  # 1-based page number (text files count as a single page)
    text: str


def load_pages(filename: str, data: bytes) -> list[Page]:
    """Extract text from a PDF, TXT or Markdown file.

    Returns one Page per PDF page so answers can cite the page they came from.
    Pages with no extractable text (for example scanned images) are skipped.
    """
    extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileError(
            f"Unsupported file type '{extension or filename}'. Use one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if extension == ".pdf":
        reader = PdfReader(BytesIO(data))
        pages = [Page(number=i + 1, text=(page.extract_text() or "")) for i, page in enumerate(reader.pages)]
    else:
        pages = [Page(number=1, text=data.decode("utf-8", errors="replace"))]

    return [page for page in pages if page.text.strip()]
