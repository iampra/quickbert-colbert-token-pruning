"""Training-free token selection for QuickBERT.

Supports:
  - idf:     pure corpus IDF top-k pruning (original QuickBERT baseline)
  - tfidf:   deduplicated, punctuation-filtered (1 + log tf) * IDF top-k pruning
  - dynamic: TF-IDF pruning with per-query dynamic k_q = max(4, |Q| // 2)
  - random:  uniformly random k tokens (ablation baseline)
"""
import math
from typing import List, Tuple
import torch
from src.idf_scorer import IDFScorer


class TokenSelector:
    """Prunes query and document token embeddings prior to late interaction."""

    def __init__(
        self,
        idf_scorer: IDFScorer,
        k: int = 8,
        mode: str = "tfidf",
        seed: int = 42,
    ):
        assert mode in {"idf", "tfidf", "dynamic", "random"}, f"Unknown mode: {mode}"
        self.idf_scorer = idf_scorer
        self.k = k
        self.mode = mode
        self.seed = seed

    def _score_sequence(self, id_list: List[int]) -> List[float]:
        if self.mode == "idf":
            return [self.idf_scorer.idf(tid) for tid in id_list]

        # TF-IDF with punctuation downweighting and first-occurrence deduplication
        counts = {}
        for tid in id_list:
            counts[tid] = counts.get(tid, 0) + 1

        scores = []
        seen = set()
        for tid in id_list:
            base = self.idf_scorer.idf(tid)
            val = (1.0 + math.log(counts[tid])) * base
            if not self.idf_scorer.is_content_token(tid):
                val *= 0.1
            if tid in seen:
                val = -1.0  # avoid wasting slots on duplicate token occurrences
            seen.add(tid)
            scores.append(val)
        return scores

    def select_batch(
        self,
        emb_list: List[torch.Tensor],
        ids_list: List[torch.Tensor],
        is_query: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Select pruned token tensors and validity masks for a list of encoded texts."""
        n = len(emb_list)
        dim = emb_list[0].shape[-1]
        device = emb_list[0].device
        dtype = emb_list[0].dtype

        budgets = []
        for ids in ids_list:
            seq_len = int(ids.numel())
            if self.mode == "dynamic" and is_query:
                budgets.append(min(max(4, seq_len // 2), seq_len))
            else:
                budgets.append(min(self.k, seq_len))

        max_kept = max(budgets) if budgets else 1
        out = torch.zeros((n, max_kept, dim), dtype=dtype, device=device)
        mask = torch.zeros((n, max_kept), dtype=torch.bool, device=device)

        gen = torch.Generator(device="cpu").manual_seed(self.seed) if self.mode == "random" else None
        total_kept = 0

        for i in range(n):
            take = budgets[i]
            id_list = ids_list[i].tolist()
            seq_len = len(id_list)
            total_kept += take

            if take == seq_len:
                idx = torch.arange(seq_len, device=device)
            elif self.mode == "random":
                idx = torch.randperm(seq_len, generator=gen)[:take].to(device)
            else:
                seq_scores = torch.tensor(
                    self._score_sequence(id_list), dtype=torch.float32, device=device
                )
                idx = torch.topk(seq_scores, take, largest=True).indices

            out[i, :take] = emb_list[i][idx]
            mask[i, :take] = True

        return out, mask, total_kept / max(n, 1)
