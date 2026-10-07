from fastapi import FastAPI, HTTPException

from .config import get_settings
from .models import AskRequest, AskResponse, SearchRequest, SearchResponse
from .service_factory import get_rag_service


app = FastAPI(
    title="Free Production-Style RAG API",
    version="1.0.0",
)


@app.get("/health")
def health():
    settings = get_settings()
    return {
        "status": "ok",
        "collection": settings.collection_name,
        "embedding_model": settings.embedding_model,
        "ollama_model": settings.ollama_model,
    }


@app.post("/v1/search", response_model=SearchResponse)
def search(request: SearchRequest):
    try:
        rag = get_rag_service()
        results = rag.retrieve(
            request.query,
            top_k=request.top_k,
            title=request.title,
        )
        return {"query": request.query, "results": results}
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="RAG index not found. Run scripts/ingest.py first.",
        )


@app.post("/v1/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        rag = get_rag_service()
        return rag.answer(request.query, title=request.title)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="RAG index not found. Run scripts/ingest.py first.",
        )
