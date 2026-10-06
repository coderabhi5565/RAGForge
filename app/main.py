from fastapi import FastAPI
from app.api.routes import router


app = FastAPI(title="RAGForge",description="Adaptive and Evaluated RAG System",version="0.1.0")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "RAGForge"
    }


app.include_router(router)