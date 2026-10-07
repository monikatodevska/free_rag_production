from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient, models


class QdrantStore:
    def __init__(
        self,
        collection_name: str,
        vector_size: int,
        path: str,
        url: str | None = None,
        api_key: str | None = None,
    ):
        self.collection_name = collection_name

        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(path=path)

        self._ensure_collection(vector_size)

    def _ensure_collection(self, vector_size: int) -> None:
        existing = {c.name for c in self.client.get_collections().collections}

        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        texts: list[str],
        metadata: list[dict[str, Any]],
        batch_size: int = 128,
    ) -> None:
        for start in range(0, len(ids), batch_size):
            end = start + batch_size

            points = [
                models.PointStruct(
                    id=ids[i],
                    vector=vectors[i],
                    payload={
                        "text": texts[i],
                        **metadata[i],
                    },
                )
                for i in range(start, min(end, len(ids)))
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

    def search(
        self,
        vector: list[float],
        top_k: int = 20,
        title: str | None = None,
    ) -> list[dict[str, Any]]:
        query_filter = None

        if title:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="title",
                        match=models.MatchValue(value=title),
                    )
                ]
            )

        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        ).points

        results = []
        for hit in hits:
            payload = dict(hit.payload or {})
            text = payload.pop("text", "")
            results.append(
                {
                    "chunk_id": str(hit.id),
                    "score": float(hit.score),
                    "text": text,
                    "metadata": payload,
                }
            )

        return results
