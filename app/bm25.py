from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Callable

from rank_bm25 import BM25Okapi


TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: list[dict[str, Any]]):
        self.chunks = chunks
        self.tokens = [tokenize(c["text"]) for c in chunks]
        self.index = BM25Okapi(self.tokens)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        with open(path, "rb") as f:
            chunks = pickle.load(f)
        return cls(chunks)

    def search(
        self,
        query: str,
        top_k: int = 20,
        filter_fn: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[dict[str, Any]]:
        scores = self.index.get_scores(tokenize(query))

        ranked = sorted(
            range(len(self.chunks)),
            key=lambda i: float(scores[i]),
            reverse=True,
        )

        results = []
        for idx in ranked:
            chunk = self.chunks[idx]

            if filter_fn and not filter_fn(chunk):
                continue

            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "score": float(scores[idx]),
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                }
            )

            if len(results) >= top_k:
                break

        return results
