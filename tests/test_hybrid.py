from app.hybrid import reciprocal_rank_fusion


def test_rrf_returns_fused_results():
    a = [
        {"chunk_id": "A", "text": "A", "metadata": {}},
        {"chunk_id": "B", "text": "B", "metadata": {}},
    ]
    b = [
        {"chunk_id": "B", "text": "B", "metadata": {}},
        {"chunk_id": "A", "text": "A", "metadata": {}},
    ]

    results = reciprocal_rank_fusion([a, b])
    assert len(results) == 2
    assert all("fusion_score" in x for x in results)
    assert results[0]["fusion_score"] > 0
