from pathlib import Path

import pandas as pd
from langchain_core.documents import Document


def load_csv(file_path: str) -> list[Document]:
    """
    Load a CSV file and represent each row as a Document.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    dataframe = pd.read_csv(path)

    documents = []

    for row_number, row in dataframe.iterrows():

        row_text = "\n".join(
            f"{column}: {row[column]}"
            for column in dataframe.columns
        )

        documents.append(
            Document(
                page_content=row_text,
                metadata={
                    "source": path.name,
                    "file_type": "csv",
                    "row": int(row_number) + 1,
                },
            )
        )

    return documents