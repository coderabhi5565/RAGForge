from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document


def load_pdf(file_path: str) -> list[Document]:
    """
    Load a PDF page by page.

    Each page becomes one Document and preserves
    source and page metadata.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    reader = PdfReader(str(path))

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()

        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "file_type": "pdf",
                    "page": page_number,
                },
            )
        )

    return documents