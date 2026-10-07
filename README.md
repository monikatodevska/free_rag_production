# Free Local-First Production-Style RAG

This project implements an end-to-end RAG service with no paid API required:

- SQuAD 1.1 documents + evaluation questions
- local sentence-transformers embeddings
- Qdrant vector storage
- BM25 lexical retrieval
- Reciprocal Rank Fusion
- cross-encoder reranking
- metadata filtering
- FastAPI
- optional local LLM through Ollama
- Recall@K and MRR retrieval evaluation

## Dataset

SQuAD 1.1 contains 100,000+ question/answer pairs over 500+ Wikipedia articles and is distributed under CC BY-SA 4.0.

Official dataset:
https://huggingface.co/datasets/rajpurkar/squad

## Install

Python 3.11+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
cp .env.example .env
```

The first run downloads the embedding/reranker models. No API key is needed.

## Ingest

Start small:

```bash
python scripts/ingest.py --split train --max-docs 1000
```

Then scale:

```bash
python scripts/ingest.py --split train --max-docs 10000
```

## Test retrieval

```bash
python -m app.cli search "When did Beyonce start becoming popular?"
```

## Optional free local generation

Install Ollama, then:

```bash
ollama pull llama3.2:3b
```

Run:

```bash
python -m app.cli ask "When did Beyonce start becoming popular?"
```

The system still works for retrieval if Ollama is unavailable.

## API

```bash
uvicorn app.main:app --reload
```

Open:

http://127.0.0.1:8000/docs

Example:

```bash
curl -X POST http://127.0.0.1:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query":"When did Beyonce start becoming popular?","top_k":5}'
```

Metadata filter:

```bash
curl -X POST http://127.0.0.1:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query":"When did Beyonce start becoming popular?","top_k":5,"title":"Beyoncé"}'
```

## Retrieval evaluation

Build an evaluation index from the validation split:

```bash
python scripts/ingest.py --split validation --collection squad_eval --bm25-path ./data/bm25_eval.pkl
python scripts/evaluate.py --collection squad_eval --bm25-path ./data/bm25_eval.pkl --limit 200
```

The evaluator checks whether a retrieved chunk contains the gold answer string.

Metrics:

- Recall@1
- Recall@5
- Recall@10
- MRR

This is retrieval evaluation, not a claim about end-to-end answer quality.

## Production upgrades

The code is production-structured, but the default setup is intentionally free and local.

For a real deployment:

1. Use secured Qdrant server/cloud or another vector store.
2. Replace the in-memory BM25 index with OpenSearch/Elasticsearch so lexical retrieval and authorization filters happen inside the search engine.
3. Add authentication and tenant-aware authorization.
4. Store source documents in PostgreSQL/object storage.
5. Add background ingestion workers.
6. Version documents and embeddings.
7. Add tracing, metrics, retries, rate limiting and circuit breakers.
8. Maintain a real evaluation set from production queries.
9. Calibrate reranker/no-answer thresholds using validation data.
10. Treat retrieved text as untrusted data to defend against indirect prompt injection.
