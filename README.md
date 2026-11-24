# QuickBERT: Training-Free IDF & TF-IDF Token Pruning for Efficient ColBERT Retrieval

**Author:** Pranav Varenya Venkatesh (`Pvareny@ncsu.edu`)  
**Affiliation:** Department of Computer Science, North Carolina State University  
**Course:** ECE 592 / CSC 591 — Advanced Topics in Efficient Deep Learning (Fall 2026)  
**Final Report:** NeurIPS 2024 PDF submitted alongside this repository.

---

## Overview

Multi-vector late-interaction retrieval models such as **ColBERTv2** achieve high passage retrieval quality by computing MaxSim over every query–document token pair at $O(\vert{}Q\vert{} \times \vert{}D\vert{})$ cost. **QuickBERT** introduces a **training-free, zero-parameter** token-pruning stage prior to late interaction:

1. **Offline Corpus IDF:** Computes $\text{IDF}(t) = \log(N / \text{df}(t))$ once offline over 50,000 MS MARCO passages (25,688 vocabulary tokens in 8.38s), enabling $O(1)$ lookup at query time.
2. **Token Selection (`idf`, `tfidf`, `dynamic`):** Retains only the top-$k$ most informative tokens per query and document using either pure corpus IDF or deduplicated, punctuation-filtered $\text{TF-IDF} = (1 + \log \text{tf}(t)) \cdot \text{IDF}(t)$, plus a dynamic query-length variant $k_q = \max(4, \lfloor \vert{}Q\vert{}/2 \rfloor)$.
3. **Sparse GPU MaxSim:** Computes batched MaxSim strictly over retained token pairs, reducing interaction complexity from $O(\vert{}Q\vert{} \times \vert{}D\vert{})$ to $O(k^2)$ with **0 added parameters** and **no retraining**.

This repository contains the complete, reproducible Fall 2026 implementation using authentic 128-d `colbert-ir/colbertv2.0` embeddings, fulfilling all post-presentation additions: large-scale MS MARCO evaluation, out-of-domain BEIR TREC-COVID evaluation, and fixed-hardware NVIDIA H100 benchmarking.

---

## Repository Organization

```text
QuickBERT/
├── README.md                     # Full documentation, module/class/function map, and CLI commands
├── requirements.txt              # Reproducible Python package dependencies
├── .gitignore                    # Ignores caches, large binary dumps, and temporary logs
├── configs/
│   └── default.yaml              # Central hyperparameter config (k sweep, seeds, corpus sizes)
├── src/
│   ├── __init__.py               # Package initializer
│   ├── idf_scorer.py             # Offline corpus IDF computation & qualitative token inspection
│   ├── encoder.py                # Pretrained ColBERTv2 wrapper with authentic 128-d projection
│   ├── token_selector.py         # Fixed-k IDF, deduplicated TF-IDF, dynamic-k, and random-k pruners
│   ├── sparse_maxsim.py          # Vectorized GPU full MaxSim and QuickBERT sparse MaxSim kernels
│   ├── data.py                   # Reproducible MS MARCO v2.1 and BEIR TREC-COVID dataset loaders
│   ├── evaluate.py               # Recall@10, MRR@10, NDCG@10 metrics & CUDA-synchronized timing
│   └── utils.py                  # Deterministic seeding, hardware reporting, and logging helpers
├── scripts/
│   ├── run_experiments.py        # Main CLI orchestrating all 4 evaluation & benchmark suites
│   ├── make_figures.py           # Generates all 6 300-DPI report figures directly from CSVs
│   ├── run_baseline.sh           # Runs Full ColBERTv2 baseline
│   ├── run_quickbert.sh          # Runs QuickBERT with configurable --k and --pruner flags
│   └── run_ablation.sh           # Runs all MS MARCO, TREC-COVID, and H100 benchmark suites
├── notebooks/
│   └── quickbert_demo.ipynb      # Clean interactive demo of QuickBERT pruning and scoring
└── results/
    ├── msmarco_5sub_raw.csv      # Per-seed raw metrics across 5 MS MARCO subsets
    ├── msmarco_5sub_summary.csv  # Aggregated mean ± std metrics for MS MARCO 5-subset sweep
    ├── msmarco_large_scale.csv   # 500-query x 10,000-passage MS MARCO results
    ├── trec_covid_results.csv    # BEIR TREC-COVID in-domain vs. cross-domain transfer results
    ├── hardware_benchmark.csv    # Fixed-hardware NVIDIA H100 MIG scaling & latency breakdown
    ├── hardware_env.json         # Exact GPU/CUDA/PyTorch environment metadata
    ├── qualitative_idf.json      # Token-level corpus IDF scores for qualitative inspection
    └── figures/                  # 300-DPI figures (fig1_pipeline.png ... fig6_qualitative_idf.png)
```

---

## Detailed File, Class, and Function Reference

### `src/idf_scorer.py`
* **`class IDFScorer`**: Computes and caches corpus-level IDF weights over BERT WordPiece token IDs.
  * `__init__(tokenizer)`: Initializes document frequency maps and lookup caches.
  * `fit(passages)`: Scans a reference passage collection (default: 50,000 MS MARCO passages) to compute $\text{df}(t)$ and cache $\text{IDF}(t) = \log(N / \text{df}(t))$.
  * `idf(token_id)`: $O(1)$ lookup returning $\text{IDF}(t)$, or $\log(N + 1)$ for unseen rare tokens.
  * `is_content_token(token_id)`: Returns `True` for alphanumeric words and WordPiece subwords (`##...`), and `False` for punctuation.
  * `score_ids(token_ids)`: Returns corpus IDF scores for a list of token IDs.
  * `inspect_query(query, k=8)`: Outputs kept vs. pruned tokens and their IDF scores for qualitative analysis.

### `src/encoder.py`
* **`class ColBERTEncoder`**: Loads the frozen pretrained `colbert-ir/colbertv2.0` checkpoint and produces authentic L2-normalized 128-d token embeddings without fine-tuning.
  * `__init__(model_name, device, q_maxlen, d_maxlen, batch_size)`: Loads the HuggingFace BERT backbone and extracts ColBERTv2's 768→128 `linear.weight` projection directly from `model.safetensors`.
  * `_encode_batch(texts, maxlen, marker_id)`: Prepends ColBERT `[Q]` (`[unused0]`) or `[D]` (`[unused1]`) markers, runs FP16 GPU inference, L2-normalizes embeddings, and strips special/padding tokens.
  * `encode_queries(queries)`: Encodes query strings into variable-length `(Lq, 128)` token embedding tensors.
  * `encode_docs(docs)`: Encodes passage strings into variable-length `(Ld, 128)` token embedding tensors.

### `src/token_selector.py`
* **`class TokenSelector`**: Performs training-free token pruning prior to MaxSim scoring.
  * `__init__(idf_scorer, k=8, mode="tfidf", seed=42)`: Configures pruning budget $k$ and strategy (`idf`, `tfidf`, `dynamic`, or `random`).
  * `_score_sequence(id_list)`: Computes pure corpus IDF scores (`idf`) or deduplicated, punctuation-downweighted $(1 + \log \text{tf}) \cdot \text{IDF}$ scores (`tfidf` / `dynamic`).
  * `select_batch(emb_list, ids_list, is_query=False)`: Selects top-$k$ tokens (or dynamic $k_q = \max(4, \lfloor |Q|/2 \rfloor)$ on queries) and returns packed GPU tensors `(N, max_kept, 128)`, boolean validity masks, and average retained token count.

### `src/sparse_maxsim.py`
* **`pad_variable_embeddings(emb_list)`**: Packs variable-length token tensors into contiguous padded `(N, L_max, 128)` GPU tensors with boolean masks and computes mean token length.
* **`full_maxsim(query_emb, doc_emb, doc_mask, chunk_size=2048)`**: Exact chunked GPU ColBERT MaxSim baseline comparing all query tokens against all valid document tokens.
* **`sparse_maxsim(query_emb, query_mask, doc_emb, doc_mask, chunk_size=4096)`**: Masked GPU sparse MaxSim restricted to retained top-$k$ query and document tokens.

### `src/data.py`
* **`_get_msmarco_validation()`**: Caches the HuggingFace `microsoft/ms_marco` (v2.1) validation split.
* **`load_msmarco_background(num_passages=50000)`**: Extracts 50,000 MS MARCO passages for offline corpus-wide IDF fitting.
* **`build_msmarco_subset(seed, num_eval_queries, num_passages)`**: Constructs deterministic MS MARCO retrieval subsets preserving all official positive passages alongside hard negatives and corpus distractors.
* **`build_trec_covid_subset(num_passages=5000, seed=42)`**: Loads all 50 official BEIR TREC-COVID biomedical queries and graded relevance judgments (`score` 1–2) over 5,000 passages.

### `src/evaluate.py`
* **`compute_metrics(scores, gold, k=10)`**: Computes Recall@10 (Hit@10), MRR@10, and NDCG@10 for binary or graded relevance labels.
* **`benchmark_fn(fn, repeats=5, warmup=2)`**: Measures mean wall-clock execution time (ms) using GPU warmup iterations and `torch.cuda.synchronize()` barriers.

### `src/utils.py`
* **`set_seed(seed=42)`**: Sets deterministic seeds across Python, NumPy, and PyTorch/CUDA.
* **`get_hardware_info()`**: Captures GPU model, CUDA version, PyTorch version, Python version, and CPU count.
* **`get_logger(name, log_file)`**: Configures dual stdout + file logging into `logs/quickbert.log`.
* **`save_json(obj, path)`**: Writes structured JSON outputs to disk.

### `scripts/run_experiments.py`
* **`evaluate_config(...)`**: Evaluates a single baseline or pruning setting and returns Recall@10, MRR@10, NDCG@10, latency (ms/query), and average interaction pairs.
* **`run_msmarco_5sub(encoder, corpus_idf, cfg, log)`**: Runs Experiment 1 (5 MS MARCO seeds × 100 queries × 1,000 passages across baseline, `idf`, `tfidf`, `dynamic`, and `random` pruners) and saves `results/msmarco_5sub_raw.csv`, `results/msmarco_5sub_summary.csv`, and `results/qualitative_idf.json`.
* **`run_large_scale(encoder, corpus_idf, cfg, log)`**: Runs Experiment 2A (500 queries × 10,000 MS MARCO passages) and saves `results/msmarco_large_scale.csv`.
* **`run_trec_covid(encoder, msmarco_idf, cfg, log)`**: Runs Experiment 2B (50 BEIR TREC-COVID queries × 5,000 passages comparing in-domain IDF vs. cross-domain MS MARCO IDF vs. random pruning) and saves `results/trec_covid_results.csv`.
* **`run_hardware_bench(encoder, corpus_idf, cfg, log)`**: Runs Experiment 3 (fixed-hardware H100 latency scaling across $N \in \{1000, 5000, 10000\}$ separating BERT encoding, IDF pruning, and MaxSim scoring) and saves `results/hardware_benchmark.csv` and `results/hardware_env.json`.
* **`main()`**: CLI entry point supporting `--mode`, `--config`, `--k`, and `--pruner`.

### `scripts/make_figures.py`
* **`make_pipeline_figure()`**: Renders the QuickBERT architecture diagram (`fig1_pipeline.png`).
* **`make_tradeoff_figure()`**: Renders 5-subset MS MARCO error-bar curves and 10k speed–accuracy trade-off (`fig2_tradeoff.png`).
* **`make_interaction_and_dynamic_figure()`**: Renders measured token-pair collapse and fixed vs. dynamic vs. random $k$ comparison (`fig3_interactions_and_dynamic.png`).
* **`make_trec_covid_figure()`**: Renders TREC-COVID in-domain vs. cross-domain transfer results (`fig4_trec_covid_transfer.png`).
* **`make_hardware_benchmark_figure()`**: Renders H100 MIG scaling curves and end-to-end latency breakdown (`fig5_h100_scaling.png`).
* **`make_qualitative_idf_figure()`**: Renders real corpus IDF scores for query tokens (`fig6_qualitative_idf.png`).

---

## Author Contributions vs. Reused Components

* **Developed from scratch by the author (Pranav Varenya Venkatesh):**
  * Offline corpus-wide `IDFScorer` and punctuation-aware deduplicated `TokenSelector` supporting pure IDF, TF-IDF, dynamic-$k$, and random-$k$ ablations (`src/idf_scorer.py`, `src/token_selector.py`).
  * Standalone ColBERTv2 128-d projection loader and `[Q]`/`[D]` marker encoding wrapper (`src/encoder.py`), avoiding `colbert-ai` build dependencies.
  * Vectorized chunked GPU `full_maxsim` and `sparse_maxsim` scoring kernels (`src/sparse_maxsim.py`).
  * Full evaluation suite (Recall@10, MRR@10, NDCG@10), CUDA-synchronized latency benchmarking, MS MARCO + BEIR TREC-COVID data pipelines, CLI experiment orchestrator, and plotting pipeline (`src/data.py`, `src/evaluate.py`, `src/utils.py`, `scripts/run_experiments.py`, `scripts/make_figures.py`).
* **Reused external artifacts:**
  * Frozen pretrained weights and tokenizer from `colbert-ir/colbertv2.0` (HuggingFace Hub) — used strictly for token embedding extraction with **no retraining and 0 added parameters**.
  * Public benchmark datasets `microsoft/ms_marco` (v2.1) and `BeIR/trec-covid` loaded via HuggingFace `datasets`.

---

## Key Experimental Results (NVIDIA H100 80GB HBM3 MIG `1g.10gb`)

Hardware: `NVIDIA H100 80GB HBM3 MIG 1g.10gb`, CUDA 12.8, PyTorch 2.10.0+cu128. Offline corpus IDF fitted once on 50,000 MS MARCO passages (25,688 vocab tokens) in 8.38s.

### 1. MS MARCO 5-Subset Sweep (100 queries × 1,000 passages, 5 seeds)
| Method | $k$ | Recall@10 (mean ± std) | MRR@10 | Avg Pairs | Scoring Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Full ColBERTv2 (Baseline) | Full | **99.4% ± 0.55%** | 65.71% | 475.9 | 1.00× |
| QuickBERT-IDF (Pure) | 8 | 87.8% ± 4.09% | 44.53% | 50.1 | 2.79× |
| Random-$k$ Ablation | 8 | 93.4% ± 1.67% | 50.75% | 50.1 | 2.79× |
| **QuickBERT-TFIDF (Ours)** | **8** | **94.2% ± 0.84%** | **53.62%** | **50.1** | **2.79×** |
| **QuickBERT-TFIDF (Ours)** | **16** | **97.0% ± 1.87%** | **58.46%** | **110.9** | **2.78×** |
| **QuickBERT-Dynamic (Ours)** | **16** | **97.8% ± 1.30%** | **56.05%** | **67.2** | **2.84×** |
| **QuickBERT-TFIDF (Ours)** | **32** | **98.6% ± 0.55%** | **62.25%** | **221.5** | **2.85×** |

### 2. Large-Scale MS MARCO (500 queries × 10,000 passages) & End-to-End H100 Scaling
| Method | Recall@10 | MRR@10 | Scoring Latency | Scoring Speedup | End-to-End Speedup (incl. BERT) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Full ColBERTv2 (Baseline) | 98.8% | 67.78% | 3.018 ms/q | 1.00× | 1.00× |
| QuickBERT-TFIDF ($k=8$) | 87.2% | 50.57% | 0.286 ms/q | **10.57×** | **4.48×** |
| QuickBERT-TFIDF ($k=16$) | 93.6% | 58.36% | 0.329 ms/q | **9.17×** | **4.22×** |
| QuickBERT-Dynamic ($k=16$) | 95.2% | 57.85% | 0.414 ms/q | **7.30×** | — |
| QuickBERT-TFIDF ($k=32$) | **97.0%** | **64.22%** | 1.124 ms/q | **2.69×** | **2.84×** |

*(Online query BERT encoding = 0.420 ms/q; online query IDF pruning = 0.046 ms/q.)*

### 3. Out-of-Domain Generalization: BEIR TREC-COVID (50 queries × 5,000 passages)
| Method | IDF Source | Recall@10 | NDCG@10 | MRR@10 | Avg Pairs | Scoring Speedup |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Full ColBERTv2 (Baseline) | None | 100.0% | 69.59% | 95.00% | 2,374.3 | 1.00× |
| **QuickBERT-TFIDF ($k=8$)** | **Cross-Domain (MS MARCO)** | **100.0%** | **63.75%** | **90.80%** | **63.7 (-97.3%)** | **9.11×** |
| QuickBERT-TFIDF ($k=8$) | In-Domain (TREC-COVID) | 96.0% | 47.98% | 79.72% | 63.7 | 9.05× |
| QuickBERT-IDF ($k=8$) | In-Domain (TREC-COVID) | 92.0% | 41.40% | 68.31% | 63.7 | 9.11× |
| Random-$k$ ($k=8$) | None | 88.0% | 38.62% | 65.49% | 63.7 | 9.05× |

---

## Commands to Run Experiments

### 1. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Baseline and Single QuickBERT Configurations
```bash
# Full ColBERTv2 baseline (Seed 0, 100 queries x 1,000 passages)
bash scripts/run_baseline.sh

# QuickBERT with TF-IDF pruning (k=8, k=16, or k=32)
bash scripts/run_quickbert.sh --k 8 --pruner tfidf
bash scripts/run_quickbert.sh --k 16 --pruner tfidf
bash scripts/run_quickbert.sh --k 32 --pruner tfidf

# QuickBERT with pure IDF pruning (k=8)
bash scripts/run_quickbert.sh --k 8 --pruner idf
```

### 3. Run Full Benchmark & Ablation Suites
```bash
# Run all 4 experiment suites end-to-end (MS MARCO 5-sub, 10k MS MARCO, TREC-COVID, H100 benchmark)
bash scripts/run_ablation.sh

# Or run individual suites selectively:
python scripts/run_experiments.py --mode msmarco_5sub
python scripts/run_experiments.py --mode large_scale
python scripts/run_experiments.py --mode trec_covid
python scripts/run_experiments.py --mode hardware_bench
```

### 4. Generate All Report Figures
```bash
# Reads CSV/JSON outputs in results/ and writes fig1..fig6 at 300 DPI to results/figures/
python scripts/make_figures.py
```