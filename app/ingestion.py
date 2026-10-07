from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    if size <= overlap:
        raise ValueError("chunk size must be greater than overlap")

    text = clean_text(text)
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))

        if end < len(text):
            boundary = max(
                text.rfind(". ", start, end),
                text.rfind("\n", start, end),
                text.rfind(" ", start, end),
            )
            if boundary > start + int(size * 0.55):
                end = boundary + 1

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= len(text):
            break

        start = max(0, end - overlap)

    return chunks


def stable_uuid(value: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def load_squad(split: str, max_docs: int | None = None) -> list[dict[str, Any]]:
    ds = load_dataset("rajpurkar/squad", split=split)

    seen: set[str] = set()
    docs: list[dict[str, Any]] = []

    for row in ds:
        context = clean_text(row["context"])
        digest = hashlib.sha256(context.encode("utf-8")).hexdigest()

        if digest in seen:
            continue

        seen.add(digest)
        docs.append(
            {
                "document_id": stable_uuid(f"squad:{split}:{digest}"),
                "title": row["title"],
                "text": context,
                "split": split,
                "source": "SQuAD 1.1 / Wikipedia",
                "content_hash": digest,
            }
        )

        if max_docs and len(docs) >= max_docs:
            break

    return docs


def make_chunks(
    docs: list[dict[str, Any]],
    size: int = 900,
    overlap: int = 150,
) -> list[Chunk]:
    output: list[Chunk] = []

    for doc in docs:
        pieces = chunk_text(doc["text"], size=size, overlap=overlap)

        for index, piece in enumerate(pieces):
            chunk_id = stable_uuid(
                f'{doc["document_id"]}:{index}:{piece}'
            )

            output.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=piece,
                    metadata={
                        "document_id": doc["document_id"],
                        "title": doc["title"],
                        "source": doc["source"],
                        "split": doc["split"],
                        "chunk_index": index,
                        "content_hash": doc["content_hash"],
                    },
                )
            )

    return output
