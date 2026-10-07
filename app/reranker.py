from __future__ import annotations

from sentence_transformers import CrossEncoder


class LocalReranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        if not candidates:
            return []

        pairs = [(query, item["text"]) for item in candidates]
        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        output = []
        for item, score in ranked[:top_k]:
            result = dict(item)
            result["rerank_score"] = float(score)
            result["score"] = float(score)
            output.append(result)

        return output
