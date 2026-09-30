from pathlib import Path

from langchain_core.documents import Document


def load_text(file_path: str) -> list[Document]:
    """
    Load a plain text file.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).strip()

    if not text:
        return []

    return [
        Document(
            page_content=text,
            metadata={
                "source": path.name,
                "file_type": "txt",
            },
        )
    ]