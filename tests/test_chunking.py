from app.ingestion import chunk_text


def test_chunking_preserves_content():
    text = "A " * 2000
    chunks = chunk_text(text, size=100, overlap=20)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_small_text_is_one_chunk():
    text = "hello world"
    assert chunk_text(text, size=100, overlap=20) == [text]
