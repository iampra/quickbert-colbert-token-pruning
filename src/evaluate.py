"""Retrieval quality metrics (Recall@10, MRR@10, NDCG@10) and GPU timing utilities."""
import math
import time
from typing import Callable, Dict, Iterable, Union
import torch


def compute_metrics(
    scores: torch.Tensor,
    gold: Union[set, Dict[int, int]],
    k: int = 10,
) -> Dict[str, float]:
    """Compute Recall@k (Hit@k), MRR@k, and NDCG@k for one query.

    Args:
        scores: (N,) relevance scores over N passages
        gold:   set of relevant indices (binary) or dict {idx: grade} (graded)
        k:      cutoff (default 10)
    """
    topk = torch.topk(scores, min(k, scores.numel())).indices.tolist()
    if isinstance(gold, dict):
        rel_map = gold
    else:
        rel_map = {idx: 1 for idx in gold}

    hit = 1.0 if any(idx in rel_map for idx in topk) else 0.0

    mrr = 0.0
    for rank, idx in enumerate(topk, start=1):
        if idx in rel_map:
            mrr = 1.0 / rank
            break

    dcg = 0.0
    for rank, idx in enumerate(topk, start=1):
        rel = rel_map.get(idx, 0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(rank + 1)

    ideal_rels = sorted(rel_map.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(rank + 1) for rank, rel in enumerate(ideal_rels, start=1))
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    return {"recall@10": hit, "mrr@10": mrr, "ndcg@10": ndcg}


@torch.no_grad()
def benchmark_fn(fn: Callable, repeats: int = 5, warmup: int = 2) -> float:
    """Measure average wall-clock execution time (ms) with CUDA synchronization."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return sum(times_ms) / len(times_ms)
