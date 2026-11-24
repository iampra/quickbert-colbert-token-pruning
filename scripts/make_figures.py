"""Generates all 6 publication-grade figures for the QuickBERT NeurIPS 2024 report.

Reads directly from CSV/JSON outputs produced by scripts/run_experiments.py (no hardcoded results).
Outputs saved to results/figures/ at 300 DPI.
"""
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

os.makedirs("results/figures", exist_ok=True)
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "legend.fontsize": 9,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "figure.dpi": 300,
})


def make_pipeline_figure():
    """Figure 1: Clean vector-quality QuickBERT architecture pipeline diagram."""
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    def draw_box(x, y, w, h, text, fc, ec, bold=True, fontsize=9.5):
        box = patches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.12,rounding_size=0.18",
            facecolor=fc, edgecolor=ec, linewidth=1.6
        )
        ax.add_patch(box)
        ax.text(
            x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold" if bold else "normal", color="#1a202c"
        )

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.3", color="#2d3748", lw=1.5)
        )

    # Query branch (left) & Document branch (right)
    draw_box(2.3, 7.4, 3.3, 0.95, "Query Tokens Q\n(ColBERTv2 128-d Embeddings)", "#dbeafe", "#2563eb")
    draw_box(7.7, 7.4, 3.3, 0.95, "Document Tokens D\n(ColBERTv2 128-d Embeddings)", "#dbeafe", "#2563eb")

    draw_box(2.3, 5.7, 3.3, 0.95, "Offline Corpus IDF Lookup\n+ TF-IDF Deduplication", "#dcfce7", "#16a34a")
    draw_box(7.7, 5.7, 3.3, 0.95, "Offline Corpus IDF Lookup\n+ TF-IDF Deduplication", "#dcfce7", "#16a34a")

    draw_box(2.3, 4.0, 3.3, 0.95, "Select Top-k Query Tokens\n(Fixed k or Dynamic k)", "#fef3c7", "#d97706")
    draw_box(7.7, 4.0, 3.3, 0.95, "Select Top-k Doc Tokens\n(No Retraining / 0 Params)", "#fef3c7", "#d97706")

    draw_box(5.0, 2.25, 5.4, 1.05, "Sparse ColBERT MaxSim Interaction\nO(|Q| x |D|)  -->  O(k^2)  (Up to 97% Fewer Pairs)", "#ede9fe", "#7c3aed")
    draw_box(5.0, 0.75, 3.4, 0.85, "Relevance Score & Ranked Passages", "#fee2e2", "#dc2626")

    # Arrows
    arrow(2.3, 6.85, 2.3, 6.25)
    arrow(7.7, 6.85, 7.7, 6.25)
    arrow(2.3, 5.15, 2.3, 4.55)
    arrow(7.7, 5.15, 7.7, 4.55)

    # Merge lines into Sparse MaxSim
    ax.plot([2.3, 2.3, 3.6], [3.45, 2.95, 2.95], color="#2d3748", lw=1.5)
    arrow(3.6, 2.95, 3.6, 2.84)
    ax.plot([7.7, 7.7, 6.4], [3.45, 2.95, 2.95], color="#2d3748", lw=1.5)
    arrow(6.4, 2.95, 6.4, 2.84)

    arrow(5.0, 1.65, 5.0, 1.24)

    plt.tight_layout()
    plt.savefig("results/figures/fig1_pipeline.png", dpi=300, bbox_inches="tight")
    plt.close()


def make_tradeoff_figure():
    """Figure 2: MS MARCO 5-Subset & 10k Large-Scale Speed vs Accuracy Trade-off."""
    df5 = pd.read_csv("results/msmarco_5sub_summary.csv")
    df10k = pd.read_csv("results/msmarco_large_scale.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    # Panel A: 5-Subset MS MARCO (Recall@10 with error bars: IDF vs TF-IDF vs Random)
    base5 = df5[df5["pruner"] == "baseline"].iloc[0]
    idf5 = df5[df5["pruner"] == "idf"].sort_values("k")
    tfidf5 = df5[df5["pruner"] == "tfidf"].sort_values("k")
    rnd5 = df5[df5["pruner"] == "random"].sort_values("k")

    ax1.axhline(base5["recall_mean"] * 100, color="#4a5568", ls="--", lw=1.4, label=f"Full ColBERTv2 ({base5['recall_mean']*100:.1f}%)")
    ax1.errorbar(tfidf5["k"], tfidf5["recall_mean"] * 100, yerr=tfidf5["recall_std"] * 100,
                 marker="o", color="#2563eb", lw=2, capsize=3.5, label="QuickBERT-TFIDF (Ours)")
    ax1.errorbar(idf5["k"], idf5["recall_mean"] * 100, yerr=idf5["recall_std"] * 100,
                 marker="^", color="#dc2626", lw=1.8, ls="-.", capsize=3.5, label="QuickBERT-IDF (Pure)")
    ax1.errorbar(rnd5["k"], rnd5["recall_mean"] * 100, yerr=rnd5["recall_std"] * 100,
                 marker="s", color="#d97706", lw=1.6, ls=":", capsize=3.5, label="Random-k Ablation")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks([2, 4, 8, 16, 32])
    ax1.set_xticklabels(["2", "4", "8", "16", "32"])
    ax1.set_xlabel("Retained Tokens per Side (k)")
    ax1.set_ylabel("Recall@10 (%)")
    ax1.set_title("(a) MS MARCO 5-Subset Sweep (mean ± std)")
    ax1.set_ylim(40, 102)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="lower right", frameon=True)

    # Panel B: Large-Scale 10k MS MARCO (Speedup vs Recall@10 dual-axis)
    base10 = df10k[df10k["pruner"] == "baseline"].iloc[0]
    tfidf10 = df10k[df10k["pruner"] == "tfidf"].sort_values("k")
    idf10 = df10k[df10k["pruner"] == "idf"].sort_values("k")

    l1 = ax2.plot(tfidf10["k"], tfidf10["speedup"], marker="o", color="#2563eb", lw=2, label="H100 Scoring Speedup (x)")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks([2, 4, 8, 16, 32])
    ax2.set_xticklabels(["2", "4", "8", "16", "32"])
    ax2.set_xlabel("Retained Tokens per Side (k)")
    ax2.set_ylabel("Scoring Speedup (x)", color="#2563eb")
    ax2.tick_params(axis="y", labelcolor="#2563eb")
    ax2.set_ylim(0, 13.0)
    ax2.grid(True, alpha=0.25)

    ax2_r = ax2.twinx()
    l2 = ax2_r.plot(tfidf10["k"], tfidf10["recall@10"] * 100, marker="s", color="#16a34a", ls="--", lw=2, label="QuickBERT-TFIDF Recall@10")
    l3 = ax2_r.plot(idf10["k"], idf10["recall@10"] * 100, marker="^", color="#dc2626", ls=":", lw=1.8, label="QuickBERT-IDF Recall@10")
    l4 = ax2_r.axhline(base10["recall@10"] * 100, color="#4a5568", ls="-.", lw=1.2, label=f"Baseline ({base10['recall@10']*100:.1f}%)")
    ax2_r.set_ylabel("Recall@10 (%)", color="#16a34a")
    ax2_r.tick_params(axis="y", labelcolor="#16a34a")
    ax2_r.set_ylim(30, 103)
    ax2.set_title("(b) Large-Scale MS MARCO (500q x 10k Passages)")

    lines = l1 + l2 + l3 + [l4]
    labels = [ln.get_label() for ln in lines]
    ax2.legend(lines, labels, loc="lower left", frameon=True, fontsize=8.2)

    plt.tight_layout()
    plt.savefig("results/figures/fig2_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close()


def make_interaction_and_dynamic_figure():
    """Figure 3: Token Interaction Collapse (left) & Fixed vs Dynamic k (right)."""
    df5 = pd.read_csv("results/msmarco_5sub_summary.csv")
    df_trec = pd.read_csv("results/trec_covid_results.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.2))

    # Left: Real measured interaction collapse across MS MARCO and TREC-COVID
    ms_base = df5[df5["pruner"] == "baseline"]["avg_interactions"].iloc[0]
    ms_k8 = df5[(df5["pruner"] == "tfidf") & (df5["k"] == 8)]["avg_interactions"].iloc[0]
    ms_k16 = df5[(df5["pruner"] == "tfidf") & (df5["k"] == 16)]["avg_interactions"].iloc[0]
    tr_base = df_trec[df_trec["pruner"] == "baseline"]["avg_interactions"].iloc[0]
    tr_k8 = df_trec[(df_trec["idf_source"] == "cross_domain_msmarco") & (df_trec["k"] == 8)]["avg_interactions"].iloc[0]

    labels = ["MS MARCO\nFull", "MS MARCO\nk=16", "MS MARCO\nk=8", "TREC-COVID\nFull", "TREC-COVID\nk=8"]
    vals = [ms_base, ms_k16, ms_k8, tr_base, tr_k8]
    colors = ["#cbd5e1", "#60a5fa", "#2563eb", "#cbd5e1", "#7c3aed"]
    bars = ax1.bar(labels, vals, color=colors, edgecolor="#334155", linewidth=0.8)
    ax1.set_yscale("log")
    ax1.set_ylabel("Avg Token Pairs per (Q, D) [Log Scale]")
    ax1.set_title("(a) Measured Token Interaction Collapse")
    ax1.set_ylim(20, 6000)
    ax1.grid(True, axis="y", alpha=0.25)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, v * 1.15, f"{v:.0f}", ha="center", va="bottom", fontweight="bold", fontsize=9)

    # Right: Fixed vs Dynamic k vs Random k on MS MARCO 5-Subset (Recall@10 & MRR@10)
    f8 = df5[(df5["pruner"] == "tfidf") & (df5["k"] == 8)].iloc[0]
    d8 = df5[(df5["pruner"] == "dynamic") & (df5["k"] == 8)].iloc[0]
    f16 = df5[(df5["pruner"] == "tfidf") & (df5["k"] == 16)].iloc[0]
    d16 = df5[(df5["pruner"] == "dynamic") & (df5["k"] == 16)].iloc[0]
    r8 = df5[(df5["pruner"] == "random") & (df5["k"] == 8)].iloc[0]

    x = np.arange(5)
    names = ["Random\nk=8", "Fixed\nk=8", "Dynamic\nk=8", "Fixed\nk=16", "Dynamic\nk=16"]
    r10s = [r8["recall_mean"] * 100, f8["recall_mean"] * 100, d8["recall_mean"] * 100, f16["recall_mean"] * 100, d16["recall_mean"] * 100]
    r10_err = [r8["recall_std"] * 100, f8["recall_std"] * 100, d8["recall_std"] * 100, f16["recall_std"] * 100, d16["recall_std"] * 100]
    c2 = ["#94a3b8", "#2563eb", "#f97316", "#1d4ed8", "#ea580c"]

    b2 = ax2.bar(x, r10s, yerr=r10_err, capsize=4, color=c2, edgecolor="#334155", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylabel("Recall@10 (%)")
    ax2.set_ylim(88, 101)
    ax2.set_title("(b) Fixed vs. Dynamic k vs. Random Pruning")
    ax2.grid(True, axis="y", alpha=0.25)
    for bar, v in zip(b2, r10s):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.55, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=8.8)

    plt.tight_layout()
    plt.savefig("results/figures/fig3_interactions_and_dynamic.png", dpi=300, bbox_inches="tight")
    plt.close()


def make_trec_covid_figure():
    """Figure 4: Out-of-Domain TREC-COVID Transfer (In-Domain vs Cross-Domain IDF)."""
    df = pd.read_csv("results/trec_covid_results.csv")
    base = df[df["pruner"] == "baseline"].iloc[0]
    in_idf8 = df[(df["idf_source"] == "in_domain") & (df["pruner"] == "idf") & (df["k"] == 8)].iloc[0]
    in_tf8 = df[(df["idf_source"] == "in_domain") & (df["pruner"] == "tfidf") & (df["k"] == 8)].iloc[0]
    cr_tf8 = df[(df["idf_source"] == "cross_domain_msmarco") & (df["k"] == 8)].iloc[0]
    cr_tf32 = df[(df["idf_source"] == "cross_domain_msmarco") & (df["k"] == 32)].iloc[0]
    rnd8 = df[(df["pruner"] == "random") & (df["k"] == 8)].iloc[0]

    labels = ["Full\nColBERTv2", "Cross-Dom\nTF-IDF (k=8)", "Cross-Dom\nTF-IDF (k=32)", "In-Domain\nTF-IDF (k=8)", "In-Domain\nIDF (k=8)", "Random\n(k=8)"]
    ndcgs = [x["ndcg@10"] * 100 for x in [base, cr_tf8, cr_tf32, in_tf8, in_idf8, rnd8]]
    mrrs = [x["mrr@10"] * 100 for x in [base, cr_tf8, cr_tf32, in_tf8, in_idf8, rnd8]]

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    b1 = ax.bar(x - w / 2, ndcgs, w, label="NDCG@10 (%)", color="#2563eb", edgecolor="#1e3a8a", linewidth=0.7)
    b2 = ax.bar(x + w / 2, mrrs, w, label="MRR@10 (%)", color="#10b981", edgecolor="#065f46", linewidth=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 108)
    ax.set_title("BEIR TREC-COVID (50 Queries x 5,000 Biomedical Passages): Domain-Shift Study")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right")

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold")

    plt.tight_layout()
    plt.savefig("results/figures/fig4_trec_covid_transfer.png", dpi=300, bbox_inches="tight")
    plt.close()


def make_hardware_benchmark_figure():
    """Figure 5: Fixed-Hardware NVIDIA H100 MIG Latency Scaling & Breakdown."""
    df = pd.read_csv("results/hardware_benchmark.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.2))

    # Left: Scoring & End-to-End Speedup across corpus sizes (N=1k, 5k, 10k)
    sizes = [1000, 5000, 10000]
    for k, col, mk in [(8, "#2563eb", "o"), (16, "#16a34a", "s"), (32, "#7c3aed", "^")]:
        sub = df[df["k"] == k].sort_values("corpus_size")
        ax1.plot(sub["corpus_size"], sub["score_speedup"], marker=mk, color=col, lw=2, label=f"k={k} (Scoring)")
        ax1.plot(sub["corpus_size"], sub["e2e_speedup"], marker=mk, color=col, lw=1.6, ls="--", alpha=0.75, label=f"k={k} (End-to-End)")
    ax1.set_xticks(sizes)
    ax1.set_xticklabels(["1,000", "5,000", "10,000"])
    ax1.set_xlabel("Candidate Corpus Size (Passages)")
    ax1.set_ylabel("Speedup over Full ColBERTv2 (x)")
    ax1.set_title("(a) H100 MIG Speedup Scaling vs. Corpus Size")
    ax1.grid(True, alpha=0.25)
    ax1.legend(ncol=2, fontsize=8, loc="upper left")

    # Right: Stacked End-to-End Latency Breakdown at N=10,000 passages
    d10 = df[df["corpus_size"] == 10000].copy()
    order = ["baseline", "quickbert_k=32", "quickbert_k=16", "quickbert_k=8", "quickbert_k=4"]
    d10 = d10.set_index("method").loc[order].reset_index()
    x_lbls = ["Baseline\n(Full)", "QuickBERT\nk=32", "QuickBERT\nk=16", "QuickBERT\nk=8", "QuickBERT\nk=4"]
    x = np.arange(len(x_lbls))

    enc = d10["encode_ms_per_q"].values
    prn = d10["prune_ms_per_q"].values
    scr = d10["score_ms_per_q"].values

    ax2.bar(x, enc, label="Query BERT Encoding", color="#94a3b8")
    ax2.bar(x, prn, bottom=enc, label="IDF Token Pruning", color="#f59e0b")
    ax2.bar(x, scr, bottom=enc + prn, label="MaxSim Scoring", color="#2563eb")
    totals = enc + prn + scr
    for xi, tot in zip(x, totals):
        ax2.text(xi, tot + 0.08, f"{tot:.2f} ms", ha="center", va="bottom", fontweight="bold", fontsize=8.8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(x_lbls)
    ax2.set_ylabel("End-to-End Latency (ms / query)")
    ax2.set_ylim(0, max(totals) * 1.18)
    ax2.set_title("(b) Latency Breakdown at N=10,000 (NVIDIA H100 MIG)")
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("results/figures/fig5_h100_scaling.png", dpi=300, bbox_inches="tight")
    plt.close()


def make_qualitative_idf_figure():
    """Figure 6: Real Offline Corpus IDF Scores for Query Tokens."""
    with open("results/qualitative_idf.json") as f:
        data = json.load(f)
    sample = data["samples"][1]  # "what is the weather in raleigh today"
    tokens = sample["tokens"]
    scores = sample["scores"]
    kept_set = {t for t, _ in sample["kept"]}
    colors = ["#2563eb" if t in kept_set else "#ef4444" for t in tokens]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bars = ax.bar(tokens, scores, color=colors, edgecolor="#1e293b", linewidth=0.7)
    thresh = min(s for _, s in sample["kept"]) - 0.15
    ax.axhline(thresh, color="#475569", ls="--", lw=1.3, label=f"Top-4 Selection Cutoff")
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, s + 0.18, f"{s:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Legend patches
    kept_patch = patches.Patch(color="#2563eb", label="Retained (High IDF)")
    pruned_patch = patches.Patch(color="#ef4444", label="Pruned (Filler / Low IDF)")
    ax.legend(handles=[kept_patch, pruned_patch], loc="upper left")
    ax.set_ylabel("Corpus IDF Score: log(N / df(t))")
    ax.set_title(f'Real Corpus-Wide IDF Scores: "{sample["query"]}"')
    ax.set_ylim(0, max(scores) * 1.18)
    ax.grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig("results/figures/fig6_qualitative_idf.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    make_pipeline_figure()
    make_tradeoff_figure()
    make_interaction_and_dynamic_figure()
    make_trec_covid_figure()
    make_hardware_benchmark_figure()
    make_qualitative_idf_figure()
    print("Generated all 6 publication-grade figures in results/figures/:")
    for fn in sorted(os.listdir("results/figures")):
        if fn.startswith("fig"):
            print("  -", fn)
