from __future__ import annotations

from typing import Any

from .bm25 import BM25Index
from .embeddings import LocalEmbedder
from .hybrid import reciprocal_rank_fusion
from .llm import OllamaLLM
from .reranker import LocalReranker
from .vector_store import QdrantStore


class RAGService:
    def __init__(
        self,
        bm25: BM25Index,
        embedder: LocalEmbedder,
        vector_store: QdrantStore,
        reranker: LocalReranker,
        llm: OllamaLLM,
        top_k_bm25: int = 20,
        top_k_vector: int = 20,
        top_k_rerank: int = 5,
        rrf_k: int = 60,
    ):
        self.bm25 = bm25
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker
        self.llm = llm
        self.top_k_bm25 = top_k_bm25
        self.top_k_vector = top_k_vector
        self.top_k_rerank = top_k_rerank
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        title: str | None = None,
    ) -> list[dict[str, Any]]:
        top_k = top_k or self.top_k_rerank

        bm25_results = self.bm25.search(
            query,
            top_k=self.top_k_bm25,
            filter_fn=(
                lambda item: item["metadata"].get("title") == title
                if title
                else True
            ),
        )

        query_vector = self.embedder.encode_query(query)

        vector_results = self.vector_store.search(
            query_vector,
            top_k=self.top_k_vector,
            title=title,
        )

        fused = reciprocal_rank_fusion(
            [bm25_results, vector_results],
            k=self.rrf_k,
        )

        return self.reranker.rerank(
            query,
            fused,
            top_k=top_k,
        )

    @staticmethod
    def build_context(results: list[dict[str, Any]]) -> str:
        blocks = []

        for i, result in enumerate(results, start=1):
            metadata = result["metadata"]
            blocks.append(
                f"""[SOURCE {i}]
Title: {metadata.get("title", "unknown")}
Chunk: {metadata.get("chunk_index", "unknown")}
Text:
{result["text"]}
"""
            )

        return "\n\n".join(blocks)

    def answer(
        self,
        query: str,
        title: str | None = None,
    ) -> dict[str, Any]:
        results = self.retrieve(
            query,
            top_k=self.top_k_rerank,
            title=title,
        )

        context = self.build_context(results)

        if not results:
            answer = "I don't have enough information in the retrieved documents."
        else:
            try:
                answer = self.llm.generate(query, context)
            except Exception as exc:
                answer = (
                    "Retrieval succeeded, but the local LLM is unavailable. "
                    "Run Ollama and pull the configured model. "
                    f"Technical detail: {exc}"
                )

        sources = [
            {
                "title": r["metadata"].get("title"),
                "chunk_index": r["metadata"].get("chunk_index"),
                "score": r.get("rerank_score", r.get("score")),
            }
            for r in results
        ]

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": results,
        }
