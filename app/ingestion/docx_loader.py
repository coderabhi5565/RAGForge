from pathlib import Path

from docx import Document as DocxDocument
from langchain_core.documents import Document


def load_docx(file_path: str) -> list[Document]:
    """
    Load a DOCX document paragraph by paragraph
    and return a standardized Document object.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    doc = DocxDocument(str(path))

    paragraphs = []

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    if not paragraphs:
        return []

    full_text = "\n".join(paragraphs)

    return [
        Document(
            page_content=full_text,
            metadata={
                "source": path.name,
                "file_type": "docx",
            },
        )
    ]