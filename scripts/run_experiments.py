"""Unified experiment runner for QuickBERT (MS MARCO, TREC-COVID, and H100 benchmarks).

Usage:
  python scripts/run_experiments.py --mode baseline
  python scripts/run_experiments.py --mode quickbert --k 16 --pruner tfidf
  python scripts/run_experiments.py --mode all
"""
import argparse
import json
import os
import sys
import time
import yaml
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import build_msmarco_subset, build_trec_covid_subset, load_msmarco_background
from src.encoder import ColBERTEncoder
from src.evaluate import benchmark_fn, compute_metrics
from src.idf_scorer import IDFScorer
from src.sparse_maxsim import full_maxsim, pad_variable_embeddings, sparse_maxsim
from src.token_selector import TokenSelector
from src.utils import get_hardware_info, get_logger, save_json, set_seed


def evaluate_config(
    q_embs, q_ids, d_embs, d_ids, golds, idf_scorer,
    mode: str, k: int = 8, repeats: int = 5, seed: int = 42,
):
    """Evaluate full ColBERTv2 baseline or QuickBERT pruning configuration."""
    n_queries = len(q_embs)
    if mode == "baseline":
        D_mat, D_mask, avg_d = pad_variable_embeddings(d_embs)
        avg_q = sum(e.shape[0] for e in q_embs) / n_queries

        def score_all():
            return [full_maxsim(q_embs[i], D_mat, D_mask) for i in range(n_queries)]

        all_scores = score_all()
        total_ms = benchmark_fn(score_all, repeats=repeats, warmup=2)
    else:
        selector = TokenSelector(idf_scorer, k=k, mode=mode, seed=seed)
        Q_sel, Q_mask, avg_q = selector.select_batch(q_embs, q_ids, is_query=True)
        D_sel, D_mask, avg_d = selector.select_batch(d_embs, d_ids, is_query=False)

        def score_all():
            return [sparse_maxsim(Q_sel[i], Q_mask[i], D_sel, D_mask) for i in range(n_queries)]

        all_scores = score_all()
        total_ms = benchmark_fn(score_all, repeats=repeats, warmup=2)

    recalls, mrrs, ndcgs = [], [], []
    for i, s in enumerate(all_scores):
        m = compute_metrics(s, golds[i], k=10)
        recalls.append(m["recall@10"])
        mrrs.append(m["mrr@10"])
        ndcgs.append(m["ndcg@10"])

    return {
        "recall@10": sum(recalls) / n_queries,
        "mrr@10": sum(mrrs) / n_queries,
        "ndcg@10": sum(ndcgs) / n_queries,
        "ms_per_query": total_ms / n_queries,
        "avg_q_tokens": avg_q,
        "avg_d_tokens": avg_d,
        "avg_interactions": avg_q * avg_d,
    }


def run_msmarco_5sub(encoder: ColBERTEncoder, corpus_idf: IDFScorer, cfg: dict, log):
    """Experiment 1: 5 random MS MARCO subsets (100 queries x 1,000 passages)."""
    log.info("=== Experiment 1: MS MARCO 5-Subset Sweep (50k Offline Corpus IDF) ===")
    seeds = cfg["subset_seeds"]
    ks = cfg["ks"]
    repeats = cfg["timing_repeats"]

    rows = []
    qualitative_saved = False

    for seed in seeds:
        queries, corpus, golds = build_msmarco_subset(
            seed=seed,
            num_eval_queries=cfg["num_eval_queries"],
            num_passages=cfg["num_passages"],
        )
        q_embs, q_ids = encoder.encode_queries(queries)
        d_embs, d_ids = encoder.encode_docs(corpus)

        if not qualitative_saved:
            samples = [
                corpus_idf.inspect_query("what is the population of raleigh", k=2),
                corpus_idf.inspect_query("what is the weather in raleigh today", k=4),
                corpus_idf.inspect_query(queries[0], k=4),
                corpus_idf.inspect_query(queries[1], k=4),
            ]
            save_json({"samples": samples}, "results/qualitative_idf.json")
            qualitative_saved = True

        base = evaluate_config(q_embs, q_ids, d_embs, d_ids, golds, corpus_idf, mode="baseline", repeats=repeats)
        rows.append({
            "seed": seed, "pruner": "baseline", "k": 0,
            "recall@10": base["recall@10"], "mrr@10": base["mrr@10"], "ndcg@10": base["ndcg@10"],
            "recall_drop": 0.0, "ms_per_query": base["ms_per_query"], "speedup": 1.0,
            "avg_interactions": base["avg_interactions"],
        })

        for pruner in ["idf", "tfidf"]:
            for k in ks:
                res = evaluate_config(q_embs, q_ids, d_embs, d_ids, golds, corpus_idf, mode=pruner, k=k, repeats=repeats)
                rows.append({
                    "seed": seed, "pruner": pruner, "k": k,
                    "recall@10": res["recall@10"], "mrr@10": res["mrr@10"], "ndcg@10": res["ndcg@10"],
                    "recall_drop": base["recall@10"] - res["recall@10"],
                    "ms_per_query": res["ms_per_query"], "speedup": base["ms_per_query"] / res["ms_per_query"],
                    "avg_interactions": res["avg_interactions"],
                })

        # Dynamic-k and Random-k ablations at k=8 and k=16
        for mode_name in ["dynamic", "random"]:
            for k in [8, 16]:
                res = evaluate_config(q_embs, q_ids, d_embs, d_ids, golds, corpus_idf, mode=mode_name, k=k, repeats=repeats, seed=seed)
                rows.append({
                    "seed": seed, "pruner": mode_name, "k": k,
                    "recall@10": res["recall@10"], "mrr@10": res["mrr@10"], "ndcg@10": res["ndcg@10"],
                    "recall_drop": base["recall@10"] - res["recall@10"],
                    "ms_per_query": res["ms_per_query"], "speedup": base["ms_per_query"] / res["ms_per_query"],
                    "avg_interactions": res["avg_interactions"],
                })

    df = pd.DataFrame(rows)
    df.to_csv("results/msmarco_5sub_raw.csv", index=False)

    summary = (
        df.groupby(["pruner", "k"], sort=False)
        .agg(
            recall_mean=("recall@10", "mean"),
            recall_std=("recall@10", "std"),
            recall_drop_mean=("recall_drop", "mean"),
            recall_drop_std=("recall_drop", "std"),
            mrr_mean=("mrr@10", "mean"),
            ndcg_mean=("ndcg@10", "mean"),
            ms_mean=("ms_per_query", "mean"),
            speedup_mean=("speedup", "mean"),
            speedup_std=("speedup", "std"),
            avg_interactions=("avg_interactions", "mean"),
        )
        .reset_index()
    )
    summary.to_csv("results/msmarco_5sub_summary.csv", index=False)
    log.info("\n" + summary.to_string(index=False))
    return summary


def run_large_scale(encoder: ColBERTEncoder, corpus_idf: IDFScorer, cfg: dict, log):
    """Experiment 2A: Large-scale MS MARCO evaluation (500 queries x 10,000 passages)."""
    n_q = cfg["large_scale_queries"]
    n_p = cfg["large_scale_passages"]
    log.info(f"=== Experiment 2A: Large-Scale MS MARCO ({n_q} queries x {n_p} passages) ===")

    queries, corpus, golds = build_msmarco_subset(seed=42, num_eval_queries=n_q, num_passages=n_p)
    q_embs, q_ids = encoder.encode_queries(queries)
    d_embs, d_ids = encoder.encode_docs(corpus)

    base = evaluate_config(q_embs, q_ids, d_embs, d_ids, golds, corpus_idf, mode="baseline", repeats=3)
    rows = [{"pruner": "baseline", "k": 0, **base, "recall_drop": 0.0, "speedup": 1.0}]

    for pruner in ["idf", "tfidf"]:
        for k in cfg["ks"]:
            res = evaluate_config(q_embs, q_ids, d_embs, d_ids, golds, corpus_idf, mode=pruner, k=k, repeats=3)
            rows.append({
                "pruner": pruner, "k": k, **res,
                "recall_drop": base["recall@10"] - res["recall@10"],
                "speedup": base["ms_per_query"] / res["ms_per_query"],
            })

    for mode_name in ["dynamic", "random"]:
        for k in [8, 16]:
            res = evaluate_config(q_embs, q_ids, d_embs, d_ids, golds, corpus_idf, mode=mode_name, k=k, repeats=3)
            rows.append({
                "pruner": mode_name, "k": k, **res,
                "recall_drop": base["recall@10"] - res["recall@10"],
                "speedup": base["ms_per_query"] / res["ms_per_query"],
            })

    df = pd.DataFrame(rows)
    df.to_csv("results/msmarco_large_scale.csv", index=False)
    log.info("\n" + df[["pruner", "k", "recall@10", "mrr@10", "ndcg@10", "ms_per_query", "speedup", "avg_interactions"]].to_string(index=False))
    return df


def run_trec_covid(encoder: ColBERTEncoder, msmarco_idf: IDFScorer, cfg: dict, log):
    """Experiment 2B: Out-of-domain BEIR TREC-COVID evaluation (in-domain vs cross-domain IDF)."""
    n_p = cfg["trec_passages"]
    log.info(f"=== Experiment 2B: BEIR TREC-COVID (50 queries x {n_p} passages) ===")

    queries, corpus, qrels = build_trec_covid_subset(num_passages=n_p, seed=42)
    in_domain_idf = IDFScorer(encoder.tokenizer).fit(corpus)
    q_embs, q_ids = encoder.encode_queries(queries)
    d_embs, d_ids = encoder.encode_docs(corpus)

    base = evaluate_config(q_embs, q_ids, d_embs, d_ids, qrels, in_domain_idf, mode="baseline", repeats=5)
    rows = [{"idf_source": "none", "pruner": "baseline", "k": 0, **base, "ndcg_drop": 0.0, "speedup": 1.0}]

    for pruner in ["idf", "tfidf"]:
        for k in [4, 8, 16, 32]:
            res = evaluate_config(q_embs, q_ids, d_embs, d_ids, qrels, in_domain_idf, mode=pruner, k=k, repeats=5)
            rows.append({
                "idf_source": "in_domain", "pruner": pruner, "k": k, **res,
                "ndcg_drop": base["ndcg@10"] - res["ndcg@10"],
                "speedup": base["ms_per_query"] / res["ms_per_query"],
            })

    # Cross-domain transfer using MS MARCO IDF on biomedical TREC-COVID passages
    for k in [8, 16, 32]:
        res = evaluate_config(q_embs, q_ids, d_embs, d_ids, qrels, msmarco_idf, mode="tfidf", k=k, repeats=5)
        rows.append({
            "idf_source": "cross_domain_msmarco", "pruner": "tfidf", "k": k, **res,
            "ndcg_drop": base["ndcg@10"] - res["ndcg@10"],
            "speedup": base["ms_per_query"] / res["ms_per_query"],
        })

    for k in [8, 16]:
        rnd = evaluate_config(q_embs, q_ids, d_embs, d_ids, qrels, in_domain_idf, mode="random", k=k, repeats=5)
        rows.append({
            "idf_source": "random", "pruner": "random", "k": k, **rnd,
            "ndcg_drop": base["ndcg@10"] - rnd["ndcg@10"],
            "speedup": base["ms_per_query"] / rnd["ms_per_query"],
        })

    df = pd.DataFrame(rows)
    df.to_csv("results/trec_covid_results.csv", index=False)
    log.info("\n" + df[["idf_source", "pruner", "k", "recall@10", "ndcg@10", "mrr@10", "speedup", "avg_interactions"]].to_string(index=False))
    return df


def run_hardware_bench(encoder: ColBERTEncoder, corpus_idf: IDFScorer, cfg: dict, log):
    """Experiment 3: Fixed-hardware H100 MIG scaling & end-to-end latency breakdown."""
    log.info("=== Experiment 3: Fixed-Hardware H100 MIG Latency Benchmark ===")
    hw = get_hardware_info()
    save_json(hw, "results/hardware_env.json")

    queries, corpus_10k, _ = build_msmarco_subset(seed=99, num_eval_queries=100, num_passages=10000)
    q_embs, q_ids = encoder.encode_queries(queries)
    d_embs_all, d_ids_all = encoder.encode_docs(corpus_10k)

    enc_ms_per_q = benchmark_fn(lambda: encoder.encode_queries(queries), repeats=5, warmup=2) / len(queries)
    prune_ms_per_q = benchmark_fn(
        lambda: TokenSelector(corpus_idf, k=16, mode="tfidf").select_batch(q_embs, q_ids, is_query=True),
        repeats=5, warmup=2,
    ) / len(queries)

    rows = []
    for corpus_n in [1000, 5000, 10000]:
        d_embs = d_embs_all[:corpus_n]
        d_ids = d_ids_all[:corpus_n]
        D_full, M_full, _ = pad_variable_embeddings(d_embs)

        def base_score():
            for i in range(len(q_embs)):
                full_maxsim(q_embs[i], D_full, M_full)

        base_ms = benchmark_fn(base_score, repeats=5, warmup=2) / len(q_embs)
        rows.append({
            "corpus_size": corpus_n, "k": 0, "method": "baseline",
            "score_ms_per_q": base_ms, "encode_ms_per_q": enc_ms_per_q,
            "prune_ms_per_q": 0.0, "e2e_ms_per_q": base_ms + enc_ms_per_q,
            "score_speedup": 1.0, "e2e_speedup": 1.0,
        })

        for k in cfg["ks"]:
            sel = TokenSelector(corpus_idf, k=k, mode="tfidf")
            Q_sel, Q_mask, _ = sel.select_batch(q_embs, q_ids, is_query=True)
            D_sel, D_mask, _ = sel.select_batch(d_embs, d_ids, is_query=False)

            def qb_score():
                for i in range(len(q_embs)):
                    sparse_maxsim(Q_sel[i], Q_mask[i], D_sel, D_mask)

            qb_ms = benchmark_fn(qb_score, repeats=5, warmup=2) / len(q_embs)
            e2e_ms = qb_ms + enc_ms_per_q + prune_ms_per_q
            rows.append({
                "corpus_size": corpus_n, "k": k, "method": f"quickbert_k={k}",
                "score_ms_per_q": qb_ms, "encode_ms_per_q": enc_ms_per_q,
                "prune_ms_per_q": prune_ms_per_q, "e2e_ms_per_q": e2e_ms,
                "score_speedup": base_ms / qb_ms,
                "e2e_speedup": (base_ms + enc_ms_per_q) / e2e_ms,
            })

    df = pd.DataFrame(rows)
    df.to_csv("results/hardware_benchmark.csv", index=False)
    log.info(f"Hardware: {hw['gpu_name']} | Query encoding: {enc_ms_per_q:.3f} ms/q | Query pruning: {prune_ms_per_q:.3f} ms/q")
    log.info("\n" + df.to_string(index=False))
    return df


def main():
    parser = argparse.ArgumentParser(description="Run QuickBERT retrieval & latency experiments.")
    parser.add_argument(
        "--mode",
        choices=["baseline", "quickbert", "msmarco_5sub", "large_scale", "trec_covid", "hardware_bench", "all"],
        default="all",
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--pruner", choices=["idf", "tfidf", "dynamic", "random"], default="tfidf")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(42)
    log = get_logger()
    hw = get_hardware_info()
    log.info(f"Environment: {json.dumps(hw)}")

    encoder = ColBERTEncoder(
        model_name=cfg["encoder"],
        q_maxlen=cfg["q_maxlen"],
        d_maxlen=cfg["d_maxlen"],
        batch_size=cfg["batch_size"],
    )

    t0 = time.perf_counter()
    bg_passages = load_msmarco_background(num_passages=50000)
    corpus_idf = IDFScorer(encoder.tokenizer).fit(bg_passages)
    log.info(f"Offline corpus IDF fitted on {corpus_idf.num_docs:,} passages ({len(corpus_idf.df):,} vocab tokens) in {time.perf_counter()-t0:.2f}s")

    if args.mode in {"baseline", "quickbert"}:
        queries, corpus, golds = build_msmarco_subset(
            seed=0, num_eval_queries=cfg["num_eval_queries"], num_passages=cfg["num_passages"]
        )
        q_embs, q_ids = encoder.encode_queries(queries)
        d_embs, d_ids = encoder.encode_docs(corpus)
        k_val = args.k if args.k is not None else cfg["k"]
        run_mode = "baseline" if args.mode == "baseline" else args.pruner
        res = evaluate_config(q_embs, q_ids, d_embs, d_ids, golds, corpus_idf, mode=run_mode, k=k_val, repeats=cfg["timing_repeats"])
        log.info(f"[{args.mode.upper()} mode={run_mode} k={k_val if run_mode!='baseline' else 'full'}] {res}")
        return

    if args.mode in {"msmarco_5sub", "all"}:
        run_msmarco_5sub(encoder, corpus_idf, cfg, log)
    if args.mode in {"large_scale", "all"}:
        run_large_scale(encoder, corpus_idf, cfg, log)
    if args.mode in {"trec_covid", "all"}:
        run_trec_covid(encoder, corpus_idf, cfg, log)
    if args.mode in {"hardware_bench", "all"}:
        run_hardware_bench(encoder, corpus_idf, cfg, log)


if __name__ == "__main__":
    main()
