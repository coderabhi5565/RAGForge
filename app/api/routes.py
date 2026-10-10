
import logging
import uuid
from pathlib import Path
from app.routing.service import AdaptiveRAGService
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.ingestion.service import IngestionService
from app.retrieval.service import RAGService
from app.models.schemas import QueryRequest, QueryResponse


logger = logging.getLogger(__name__)

router = APIRouter()

ingestion_service = IngestionService()
adaptive_service = AdaptiveRAGService()

UPLOAD_DIR = Path("data/uploads")
MAX_FILE_SIZE = 20 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown", ".csv"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    original_name = (file.filename or "").replace("\\", "/")
    original_name = Path(original_name).name

    if not original_name or original_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    extension = Path(original_name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload PDF, DOCX, TXT, MD, or CSV.",
        )

    safe_name = f"{uuid.uuid4().hex}_{original_name}"
    file_path = UPLOAD_DIR / safe_name
    total_size = 0

    try:
        with file_path.open("wb") as output:
            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="File exceeds the 20 MB upload limit.",
                    )

                output.write(chunk)

        if total_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        result = ingestion_service.ingest(str(file_path))
        adaptive_service.rag_service.refresh_bm25()

        return {
            "message": "Document ingested successfully",
            "filename": original_name,
            "documents": result["documents"],
            "chunks": result["chunks"],
        }

    except HTTPException:
        file_path.unlink(missing_ok=True)
        raise

    except Exception as exc:
        file_path.unlink(missing_ok=True)
        logger.exception("Document ingestion failed")
        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed. Check server logs.",
        ) from exc

    finally:
        await file.close()


@router.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty or whitespace.",
        )

    try:
        return adaptive_service.query(
        question=request.question.strip(),
        top_k=request.top_k,
        )

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as exc:
        logger.exception("Query processing failed")
        raise HTTPException(
            status_code=500,
            detail="Query processing failed. Check server logs.",
        ) from exc
