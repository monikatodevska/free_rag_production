from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class LocalEmbedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError("Embedding model did not expose its dimension")
        return int(dimension)

    def encode(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32).tolist()

    def encode_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        return np.asarray(vector, dtype=np.float32).tolist()
