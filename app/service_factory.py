from functools import lru_cache

from .bm25 import BM25Index
from .config import get_settings
from .embeddings import LocalEmbedder
from .llm import OllamaLLM
from .rag import RAGService
from .reranker import LocalReranker
from .vector_store import QdrantStore


@lru_cache
def get_rag_service() -> RAGService:
    settings = get_settings()

    bm25 = BM25Index.load(settings.bm25_path)
    embedder = LocalEmbedder(settings.embedding_model)

    vector_store = QdrantStore(
        collection_name=settings.collection_name,
        vector_size=embedder.dimension,
        path=settings.qdrant_path,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )

    reranker = LocalReranker(settings.reranker_model)

    llm = OllamaLLM(
        settings.ollama_base_url,
        settings.ollama_model,
    )

    return RAGService(
        bm25=bm25,
        embedder=embedder,
        vector_store=vector_store,
        reranker=reranker,
        llm=llm,
        top_k_bm25=settings.top_k_bm25,
        top_k_vector=settings.top_k_vector,
        top_k_rerank=settings.top_k_rerank,
        rrf_k=settings.rrf_k,
    )
