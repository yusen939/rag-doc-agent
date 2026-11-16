from app.services.reranker import LexicalReranker, RankedChunk, tokenize


def test_tokenize_handles_chinese_bigrams_and_english_words():
    tokens = tokenize("退款 Refund 政策")
    assert "退" in tokens
    assert "退款" in tokens
    assert "refund" in tokens


def test_reranker_prefers_lexically_relevant_candidate():
    reranker = LexicalReranker(lexical_weight=0.7)
    candidates = [
        RankedChunk(
            id="a",
            text="企业版支持私有化部署和审计日志。",
            metadata={},
            dense_score=0.63,
            rerank_score=0.63,
        ),
        RankedChunk(
            id="b",
            text="标准版价格为 29 元每人每月，支持退款。",
            metadata={},
            dense_score=0.60,
            rerank_score=0.60,
        ),
    ]
    ranked = reranker.rerank("标准版多少钱，可以退款吗", candidates, top_n=2)
    assert ranked[0].id == "b"


def test_reranker_rejects_invalid_weight():
    try:
        LexicalReranker(lexical_weight=1.2)
    except ValueError:
        return
    raise AssertionError("lexical_weight must be between 0 and 1")
