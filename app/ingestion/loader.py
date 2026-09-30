from pathlib import Path

from .pdf_loader import load_pdf
from .docx_loader import load_docx
from .text_loader import load_text
from .markdown_loader import load_markdown
from .csv_loader import load_csv


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".csv",
}


def load_document(file_path: str):
    """
    Load a document using the appropriate
    format-specific loader.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if extension == ".pdf":
        return load_pdf(file_path)

    if extension == ".docx":
        return load_docx(file_path)

    if extension == ".txt":
        return load_text(file_path)

    if extension == ".md":
        return load_markdown(file_path)

    if extension == ".csv":
        return load_csv(file_path)
    
    raise ValueError(f"Unsupported file type: {extension}")