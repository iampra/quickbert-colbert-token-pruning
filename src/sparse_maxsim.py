"""Vectorized GPU full ColBERT MaxSim and QuickBERT sparse MaxSim."""
from typing import List, Tuple
import torch


def pad_variable_embeddings(
    emb_list: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Pack variable-length token embeddings into padded (N, L_max, D) tensor + mask."""
    n = len(emb_list)
    max_len = max(e.shape[0] for e in emb_list)
    dim = emb_list[0].shape[-1]
    device = emb_list[0].device
    dtype = emb_list[0].dtype

    padded = torch.zeros((n, max_len, dim), dtype=dtype, device=device)
    mask = torch.zeros((n, max_len), dtype=torch.bool, device=device)
    total_len = 0
    for i, e in enumerate(emb_list):
        length = e.shape[0]
        total_len += length
        padded[i, :length] = e
        mask[i, :length] = True
    return padded, mask, total_len / max(n, 1)


@torch.no_grad()
def full_maxsim(
    query_emb: torch.Tensor,
    doc_emb: torch.Tensor,
    doc_mask: torch.Tensor,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Compute exact full ColBERT MaxSim between one query and N candidate documents.

    Args:
        query_emb: (Lq, D) L2-normalized query token embeddings
        doc_emb:   (N, Ld, D) padded document token embeddings
        doc_mask:  (N, Ld) boolean mask indicating valid document tokens
        chunk_size: number of documents scored per GPU matmul chunk
    Returns:
        scores: (N,) relevance scores
    """
    n_docs = doc_emb.shape[0]
    q_t = query_emb.to(doc_emb.dtype).transpose(0, 1)  # (D, Lq)
    scores = torch.empty(n_docs, dtype=torch.float32, device=doc_emb.device)

    for start in range(0, n_docs, chunk_size):
        end = min(start + chunk_size, n_docs)
        # (C, Ld, D) @ (D, Lq) -> (C, Ld, Lq)
        sim = torch.matmul(doc_emb[start:end], q_t)
        sim.masked_fill_(~doc_mask[start:end].unsqueeze(-1), -1e4)
        scores[start:end] = sim.max(dim=1).values.float().sum(dim=-1)
    return scores


@torch.no_grad()
def sparse_maxsim(
    query_emb: torch.Tensor,
    query_mask: torch.Tensor,
    doc_emb: torch.Tensor,
    doc_mask: torch.Tensor,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Compute QuickBERT sparse MaxSim over top-k query and document tokens.

    Args:
        query_emb:  (Kq, D) pruned query token embeddings
        query_mask: (Kq,) boolean validity mask for query tokens
        doc_emb:    (N, Kd, D) pruned document token embeddings
        doc_mask:   (N, Kd) boolean validity mask for document tokens
        chunk_size: number of documents scored per GPU matmul chunk
    Returns:
        scores: (N,) relevance scores
    """
    n_docs = doc_emb.shape[0]
    q_t = query_emb.to(doc_emb.dtype).transpose(0, 1)  # (D, Kq)
    qm = query_mask.float().unsqueeze(0)               # (1, Kq)
    scores = torch.empty(n_docs, dtype=torch.float32, device=doc_emb.device)

    for start in range(0, n_docs, chunk_size):
        end = min(start + chunk_size, n_docs)
        # (C, Kd, D) @ (D, Kq) -> (C, Kd, Kq)
        sim = torch.matmul(doc_emb[start:end], q_t)
        sim.masked_fill_(~doc_mask[start:end].unsqueeze(-1), -1e4)
        max_sims = sim.max(dim=1).values.float()
        scores[start:end] = (max_sims * qm).sum(dim=-1)
    return scores
