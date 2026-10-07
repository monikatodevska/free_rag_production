from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.bm25 import BM25Index
from app.config import get_settings
from app.embeddings import LocalEmbedder
from app.hybrid import reciprocal_rank_fusion
from app.reranker import LocalReranker
from app.vector_store import QdrantStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="squad_eval")
    parser.add_argument("--bm25-path", default="./data/bm25_eval.pkl")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    settings = get_settings()
    ds = load_dataset("rajpurkar/squad", split="validation")

    bm25 = BM25Index.load(args.bm25_path)
    embedder = LocalEmbedder(settings.embedding_model)

    store = QdrantStore(
        collection_name=args.collection,
        vector_size=embedder.dimension,
        path=settings.qdrant_path,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    reranker = LocalReranker(settings.reranker_model)

    recalls = {1: [], 5: [], 10: []}
    reciprocal_ranks = []

    rows = ds.select(range(min(args.limit, len(ds))))

    for i, row in enumerate(rows):
        query = row["question"]
        answer = row["answers"]["text"][0].strip()

        bm25_results = bm25.search(query, top_k=20)
        vector_results = store.search(
            embedder.encode_query(query),
            top_k=20,
        )

        fused = reciprocal_rank_fusion(
            [bm25_results, vector_results],
            k=settings.rrf_k,
        )
        reranked = reranker.rerank(query, fused, top_k=args.top_k)

        ranks = [
            rank
            for rank, item in enumerate(reranked, start=1)
            if answer.lower() in item["text"].lower()
        ]

        first_rank = min(ranks) if ranks else None

        for k in recalls:
            recalls[k].append(
                1.0 if first_rank is not None and first_rank <= k else 0.0
            )

        reciprocal_ranks.append(
            0.0 if first_rank is None else 1.0 / first_rank
        )

        if (i + 1) % 25 == 0:
            print(f"Evaluated {i + 1} queries...")

    print("\nRetrieval evaluation")
    print("====================")
    for k in [1, 5, 10]:
        print(f"Recall@{k}: {np.mean(recalls[k]):.4f}")
    print(f"MRR:       {np.mean(reciprocal_ranks):.4f}")
    print("\nThis measures retrieval of a chunk containing the gold answer span.")


if __name__ == "__main__":
    main()
