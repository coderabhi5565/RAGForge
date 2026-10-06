from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.ingestion.service import IngestionService
from app.retrieval.service import RAGService
from app.models.schemas import QueryRequest, QueryResponse


router = APIRouter()

ingestion_service = IngestionService()
rag_service = RAGService()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...)
):
    try:
        file_path = UPLOAD_DIR / file.filename

        contents = await file.read()

        with open(file_path, "wb") as f:
            f.write(contents)

        result = ingestion_service.ingest(
            str(file_path)
        )

        return {
            "message": "Document ingested successfully",
            "filename": file.filename,
            "documents": result["documents"],
            "chunks": result["chunks"],
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.post(
    "/query",
    response_model=QueryResponse
)
def query_documents(request: QueryRequest):

    try:
        result = rag_service.query(
            question=request.question,
            top_k=request.top_k,
        )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )