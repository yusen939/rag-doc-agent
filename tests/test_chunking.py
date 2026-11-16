from app.services.chunking import chunk_text, normalize_text


def test_short_text_returns_single_chunk():
    assert chunk_text("这是一段简短文档。", chunk_size=100, overlap=20) == ["这是一段简短文档。"]


def test_chunks_respect_size_and_overlap():
    text = "第一段介绍产品功能。\n\n" + "这是用于测试切块逻辑的句子。" * 80
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 2
    assert all(len(chunk) <= 200 for chunk in chunks)
    assert chunks[0] != chunks[1]


def test_chunking_rejects_invalid_parameters():
    for chunk_size, overlap in [(20, 5), (100, 100), (100, -1)]:
        try:
            chunk_text("content", chunk_size=chunk_size, overlap=overlap)
        except ValueError:
            continue
        raise AssertionError("invalid chunk parameters should fail")


def test_normalize_text_removes_excess_blank_lines_and_control_chars():
    assert normalize_text("a\x00b\r\n\r\n\r\nc") == "a b\n\nc"
