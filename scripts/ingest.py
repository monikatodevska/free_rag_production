from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.bm25 import BM25Index
from app.config import get_settings
from app.embeddings import LocalEmbedder
from app.ingestion import load_squad, make_chunks
from app.vector_store import QdrantStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--max-docs", type=int, default=1000)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--bm25-path", default=None)
    args = parser.parse_args()

    settings = get_settings()
    collection = args.collection or settings.collection_name
    bm25_path = args.bm25_path or settings.bm25_path

    print(f"Loading SQuAD split={args.split}, max_docs={args.max_docs}...")
    docs = load_squad(args.split, max_docs=args.max_docs)
    print(f"Unique documents: {len(docs)}")

    chunks = make_chunks(
        docs,
        size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    print(f"Chunks: {len(chunks)}")

    embedder = LocalEmbedder(settings.embedding_model)
    print(f"Embedding with {settings.embedding_model}...")
    vectors = embedder.encode([c.text for c in chunks])

    store = QdrantStore(
        collection_name=collection,
        vector_size=embedder.dimension,
        path=settings.qdrant_path,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )

    store.upsert(
        ids=[c.chunk_id for c in chunks],
        vectors=vectors,
        texts=[c.text for c in chunks],
        metadata=[c.metadata for c in chunks],
    )

    bm25_chunks = [
        {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "metadata": c.metadata,
        }
        for c in chunks
    ]
    BM25Index(bm25_chunks).save(bm25_path)

    print("Done.")
    print(f"Qdrant collection: {collection}")
    print(f"BM25 index: {bm25_path}")


if __name__ == "__main__":
    main()
