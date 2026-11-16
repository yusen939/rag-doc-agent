from __future__ import annotations

import re
from dataclasses import dataclass

_ASCII_WORD = re.compile(r"[a-z0-9_]+")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")


@dataclass(slots=True)
class RankedChunk:
    id: str
    text: str
    metadata: dict
    dense_score: float
    rerank_score: float


def tokenize(text: str) -> set[str]:
    """Create lightweight lexical tokens for Chinese and English text."""
    lowered = text.lower()
    tokens = set(_ASCII_WORD.findall(lowered))
    for run in _CJK_RUN.findall(lowered):
        tokens.update(run)
        tokens.update(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


class LexicalReranker:
    """A dependency-free reranker combining dense similarity and lexical overlap.

    It is intentionally lightweight: the goal is to demonstrate a two-stage
    retrieval pipeline without downloading a cross-encoder at first run. The
    interface can be replaced by a model-based reranker in production.
    """

    def __init__(self, lexical_weight: float = 0.45) -> None:
        if not 0 <= lexical_weight <= 1:
            raise ValueError("lexical_weight must be between 0 and 1")
        self.lexical_weight = lexical_weight

    def score(self, query: str, text: str, dense_score: float) -> float:
        query_tokens = tokenize(query)
        if not query_tokens:
            return dense_score
        document_tokens = tokenize(text)
        if not document_tokens:
            return dense_score

        overlap = len(query_tokens & document_tokens) / len(query_tokens)
        union = len(query_tokens | document_tokens)
        jaccard = len(query_tokens & document_tokens) / union if union else 0.0
        phrase_bonus = 1.0 if query.strip().lower() in text.lower() else 0.0
        lexical = 0.65 * overlap + 0.25 * jaccard + 0.10 * phrase_bonus

        dense_normalized = max(0.0, min(1.0, (dense_score + 1.0) / 2.0))
        return (1.0 - self.lexical_weight) * dense_normalized + self.lexical_weight * lexical

    def rerank(
        self,
        query: str,
        candidates: list[RankedChunk],
        top_n: int,
    ) -> list[RankedChunk]:
        if top_n <= 0:
            return []
        for candidate in candidates:
            candidate.rerank_score = self.score(query, candidate.text, candidate.dense_score)
        return sorted(candidates, key=lambda item: item.rerank_score, reverse=True)[:top_n]
