"""Corpus-wide IDF importance scorer for QuickBERT.

Computes IDF(t) = log(N / df(t)) once offline over a reference passage collection.
Also identifies pure punctuation tokens while preserving WordPiece subwords (e.g., ##bet).
"""
import math
from typing import Dict, Iterable, List


class IDFScorer:
    """Computes and caches corpus-level IDF weights over tokenizer vocabulary."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.df: Dict[int, int] = {}
        self.num_docs: int = 0
        self._idf_cache: Dict[int, float] = {}
        self._content_cache: Dict[int, bool] = {}

    def fit(self, passages: Iterable[str]) -> "IDFScorer":
        """Compute document frequencies df(t) over a passage collection."""
        self.df = {}
        self.num_docs = 0
        for text in passages:
            self.num_docs += 1
            token_ids = self.tokenizer.encode(
                text, add_special_tokens=False, truncation=True, max_length=512
            )
            for tid in set(token_ids):
                self.df[tid] = self.df.get(tid, 0) + 1
        self._idf_cache = {
            tid: math.log(self.num_docs / freq)
            for tid, freq in self.df.items()
        }
        return self

    def idf(self, token_id: int) -> float:
        """Return IDF(t) = log(N / df(t)); unseen rare tokens receive log(N + 1)."""
        if token_id in self._idf_cache:
            return self._idf_cache[token_id]
        return math.log(self.num_docs + 1)

    def is_content_token(self, token_id: int) -> bool:
        """Return True for alphanumeric words and WordPiece subwords; False for punctuation."""
        if token_id not in self._content_cache:
            tok = self.tokenizer.convert_ids_to_tokens(int(token_id))
            stripped = tok[2:] if tok.startswith("##") else tok
            self._content_cache[token_id] = any(ch.isalnum() for ch in stripped)
        return self._content_cache[token_id]

    def score_ids(self, token_ids: List[int]) -> List[float]:
        """Return corpus IDF scores for a list of token IDs."""
        return [self.idf(int(tid)) for tid in token_ids]

    def inspect_query(self, query: str, k: int = 8) -> Dict[str, object]:
        """Return token-level IDF scores and kept/pruned breakdown for qualitative analysis."""
        ids = self.tokenizer.encode(query, add_special_tokens=False)
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        scores = self.score_ids(ids)
        ranked = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)
        keep_set = set(ranked[: min(k, len(ids))])
        kept = [(tokens[i], round(scores[i], 3)) for i in range(len(ids)) if i in keep_set]
        pruned = [(tokens[i], round(scores[i], 3)) for i in range(len(ids)) if i not in keep_set]
        return {"query": query, "tokens": tokens, "scores": scores, "kept": kept, "pruned": pruned}
