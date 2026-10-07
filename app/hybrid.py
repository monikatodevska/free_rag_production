from __future__ import annotations

from collections import defaultdict
from typing import Any


def reciprocal_rank_fusion(
    result_lists: list[list[dict[str, Any]]],
    k: int = 60,
) -> list[dict[str, Any]]:
    scores = defaultdict(float)
    documents: dict[str, dict[str, Any]] = {}

    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            chunk_id = item["chunk_id"]
            scores[chunk_id] += 1.0 / (k + rank)
            documents[chunk_id] = item

    ranked_ids = sorted(
        scores,
        key=lambda chunk_id: scores[chunk_id],
        reverse=True,
    )

    output = []
    for chunk_id in ranked_ids:
        item = dict(documents[chunk_id])
        item["fusion_score"] = scores[chunk_id]
        output.append(item)

    return output
