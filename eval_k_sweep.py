#!/usr/bin/env python3
"""测试 weighted_avg 策略下 k=1~320，找出最优 k 并分析显著 token 分布"""
import sys
sys.path.insert(0, "/home/moxu/MMRAG/otherExp/colpali/mteb")

import json
import os
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from accelerate import Accelerator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mteb.models.model_implementations.colqwen_models import ColQwen3VLEmbeddingWrapper
from mteb.similarity_functions import _convert_to_tensor
from mteb.get_tasks import get_tasks
from datasets import load_dataset


# ========== Weighted Avg MaxSim with Token Tracking ==========

def weighted_avg_with_tracking(q_embeds, d_embeds, k, doc_positions=None):
    """Weighted average of top-k doc tokens.

    Returns:
        scores: (n_q, n_d)
        selected_positions: list of positions selected per (q, d) pair, for analysis
    """
    scores = torch.einsum("ash,bth->abst", q_embeds, d_embeds)  # (n_q, n_d, seq_q, seq_d)
    seq_q = scores.size(2)
    seq_d = scores.size(3)

    # Doc position weighting
    if doc_positions is None:
        doc_positions = torch.arange(seq_d, device=d_embeds.device, dtype=torch.float32)
    doc_weights = torch.exp(doc_positions / seq_d)
    doc_weights = doc_weights / doc_weights.sum()

    # Top-k doc tokens per (query, doc, query_token)
    topk_scores, topk_idxs = scores.topk(k=k, axis=-1)  # (n_q, n_d, seq_q, k)

    # Apply doc weights to top-k
    topk_weights = doc_weights[-k:].unsqueeze(0).unsqueeze(0).unsqueeze(0)
    weighted_topk = topk_scores * topk_weights
    avg_scores = weighted_topk.sum(axis=-1) / k
    return avg_scores.sum(axis=-1), topk_idxs


def compute_similarity_weighted_avg_tracking(q_embeds, d_embeds, k, corpus_chunk_size=200, query_step=4):
    """Compute weighted_avg similarity with token position tracking."""
    num_docs = d_embeds.size(0)
    final_scores = None
    all_topk_idxs = []

    for chunk_start in range(0, num_docs, corpus_chunk_size):
        chunk_end = min(chunk_start + corpus_chunk_size, num_docs)
        d_chunk = d_embeds[chunk_start:chunk_end]
        chunk_scores_list = []
        chunk_topk_list = []

        for q_start in range(0, q_embeds.size(0), query_step):
            q_end = min(q_start + query_step, q_embeds.size(0))
            q_batch = q_embeds[q_start:q_end]
            scores, topk_idxs = weighted_avg_with_tracking(q_batch, d_chunk, k)
            chunk_scores_list.append(scores)
            chunk_topk_list.append(topk_idxs)

        chunk_scores = torch.cat(chunk_scores_list, dim=0)
        if final_scores is None:
            final_scores = chunk_scores
        else:
            final_scores = torch.cat([final_scores, chunk_scores], dim=1)

    return final_scores


def compute_similarity_baseline(q_embeds, d_embeds, corpus_chunk_size=200, query_step=4):
    """Original max_sim."""
    def max_sim_base(a, b):
        a = _convert_to_tensor(a)
        b = _convert_to_tensor(b)
        if len(a.shape) == 2:
            a = a.reshape(1, *a.shape)
        if len(b.shape) == 2:
            b = b.reshape(1, *b.shape)
        scores = torch.einsum("ash,bth->abst", a, b)
        max_scores = scores.max(axis=-1).values
        return max_scores.sum(axis=-1)

    num_docs = d_embeds.size(0)
    final_scores = None
    for chunk_start in range(0, num_docs, corpus_chunk_size):
        chunk_end = min(chunk_start + corpus_chunk_size, num_docs)
        d_chunk = d_embeds[chunk_start:chunk_end]
        chunk_scores_list = []
        for q_start in range(0, q_embeds.size(0), query_step):
            q_end = min(q_start + query_step, q_embeds.size(0))
            q_batch = q_embeds[q_start:q_end]
            chunk_scores_list.append(max_sim_base(q_batch, d_chunk))
        chunk_scores = torch.cat(chunk_scores_list, dim=0)
        if final_scores is None:
            final_scores = chunk_scores
        else:
            final_scores = torch.cat([final_scores, chunk_scores], dim=1)
    return final_scores


# ========== Retrieval Metrics ==========

def compute_retrieval_metrics(
    scores: torch.Tensor,
    query_ids: list[str],
    doc_ids: list[str],
    relevant_docs: dict[str, dict[str, int]],
    k_values: list[int] = [5, 10, 20, 100],
) -> dict[str, float]:
    metrics = {}
    mrr = 0.0
    recall_at_k = {k: 0.0 for k in k_values}
    ndcg_at_k = {k: 0.0 for k in k_values}
    num_evaluated = 0

    for i, qid in enumerate(query_ids):
        if qid not in relevant_docs:
            continue
        relevant = relevant_docs[qid]
        if not relevant:
            continue
        num_evaluated += 1
        ranking = torch.argsort(scores[i], descending=True)
        best_rank = float('inf')

        dcg = {k: 0.0 for k in k_values}
        idcg = {k: 0.0 for k in k_values}
        sorted_rel_scores = sorted(relevant.values(), reverse=True)

        for rank, doc_idx in enumerate(ranking.tolist(), 1):
            did = doc_ids[doc_idx]
            if did in relevant:
                if best_rank == float('inf'):
                    best_rank = rank
                for k in k_values:
                    if rank <= k:
                        rel = relevant[did]
                        dcg[k] += rel / np.log2(rank + 1) if rank > 0 else rel

        for k in k_values:
            idcg_rel = sorted_rel_scores[:k]
            for rank, rel in enumerate(idcg_rel, 1):
                idcg[k] += rel / np.log2(rank + 1) if rank > 0 else rel

        if best_rank < float('inf'):
            mrr += 1.0 / best_rank
            for k in k_values:
                top_k_docs = set(ranking[:k].tolist())
                top_k_dids = {doc_ids[idx] for idx in top_k_docs}
                if set(relevant.keys()) & top_k_dids:
                    recall_at_k[k] += 1

        for k in k_values:
            if idcg[k] > 0:
                ndcg_at_k[k] += dcg[k] / idcg[k]

    if num_evaluated > 0:
        mrr /= num_evaluated
        for k in k_values:
            recall_at_k[k] /= num_evaluated
            ndcg_at_k[k] /= num_evaluated

    metrics["mrr"] = mrr
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k[k]
        metrics[f"ndcg@{k}"] = ndcg_at_k[k]

    return metrics


# ========== Token Distribution Tracking ==========

def track_selected_token_positions(q_embeds, d_embeds, k, corpus_chunk_size=200, query_step=4, sample_queries=50):
    """Track which doc token positions are selected for top-k analysis.

    Returns:
        selected_counts: (seq_d,) - how many times each doc token position was selected
        selected_per_query: list of dicts, each {doc_idx: [positions]} for sampled queries
    """
    num_docs = d_embeds.size(0)
    seq_d = d_embeds.size(1)
    total_selected = np.zeros(seq_d, dtype=np.int64)
    sampled_per_query = []

    sample_step = max(1, q_embeds.size(0) // sample_queries)

    for chunk_start in range(0, num_docs, corpus_chunk_size):
        chunk_end = min(chunk_start + corpus_chunk_size, num_docs)
        d_chunk = d_embeds[chunk_start:chunk_end]

        for q_start in range(0, q_embeds.size(0), query_step):
            q_end = min(q_start + query_step, q_embeds.size(0))
            q_batch = q_embeds[q_start:q_end]
            scores = torch.einsum("ash,bth->abst", q_batch, d_chunk)
            topk_scores, topk_idxs = scores.topk(k=k, axis=-1)

            # Count positions across all (q, d) pairs
            topk_idxs_cpu = topk_idxs.cpu().numpy()
            for qi in range(topk_idxs.size(0)):
                for di in range(topk_idxs.size(1)):
                    for qi2 in range(topk_idxs.size(2)):
                        for pos in topk_idxs_cpu[qi, di, qi2]:
                            total_selected[pos] += 1

            # Sample queries
            if q_start % sample_step == 0 and len(sampled_per_query) < sample_queries:
                for qi in range(min(q_end - q_start, sample_queries - len(sampled_per_query))):
                    q_sampled = {}
                    for di in range(topk_idxs.size(1)):
                        q_sampled[di] = topk_idxs_cpu[qi, di, :, :].tolist()
                    sampled_per_query.append(q_sampled)

    return total_selected, sampled_per_query


# ========== Data Loading ==========

def load_task_data(task_name: str):
    tasks = get_tasks(tasks=[task_name])
    if not tasks:
        raise ValueError(f"Task {task_name} not found")
    task = tasks[0]
    dataset_path = task.metadata.dataset["path"]
    revision = task.metadata.dataset["revision"]

    corpus_ds = load_dataset(dataset_path, name="english-corpus", revision=revision, split="test")
    queries_ds = load_dataset(dataset_path, name="english-queries", revision=revision, split="test")
    qrels_ds = load_dataset(dataset_path, name="english-qrels", revision=revision, split="test")

    corpus = {row["id"]: row for row in corpus_ds}
    queries = {row["id"]: row for row in queries_ds}

    relevant_docs = defaultdict(dict)
    for row in qrels_ds:
        relevant_docs[row["query-id"]][row["corpus-id"]] = row["score"]

    return queries, corpus, dict(relevant_docs)


# ========== Main ==========

def main():
    import argparse
    p = argparse.ArgumentParser(description="K sweep for weighted_avg strategy")
    p.add_argument("--embeddings-cache-dir", type=str, default="./embeddings_cache")
    p.add_argument("--task", type=str, default="Vidore3PhysicsRetrieval.v2")
    p.add_argument("--output-dir", type=str, default="./k_sweep_results")
    p.add_argument("--k-min", type=int, default=1)
    p.add_argument("--k-max", type=int, default=320)
    p.add_argument("--k-step", type=int, default=1)
    args = p.parse_args()

    accelerator = Accelerator()
    if accelerator.num_processes != 1:
        raise RuntimeError(f"仅支持单进程，当前 num_processes={accelerator.num_processes}")

    cache_dir = Path(args.embeddings_cache_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task
    print(f"Loading task: {task_name}")

    # Load cached embeddings
    q_cache = cache_dir / f"{task_name}_queries.pt"
    d_cache = cache_dir / f"{task_name}_docs.pt"

    print("Loading cached embeddings...")
    q_loaded = torch.load(q_cache)
    d_loaded = torch.load(d_cache)
    query_embeds = q_loaded["data"].to(accelerator.device).float() * q_loaded["scale"].to(accelerator.device)
    doc_embeds = d_loaded["data"].to(accelerator.device).float() * d_loaded["scale"].to(accelerator.device)
    q_order_ids = q_loaded["ids"]
    d_order_ids = d_loaded["ids"]

    queries_dict, corpus_dict, relevant_docs = load_task_data(task_name)

    print(f"Query embeds: {query_embeds.shape}, Doc embeds: {doc_embeds.shape}")
    seq_d = doc_embeds.size(1)
    print(f"Doc seq length: {seq_d}")

    # Baseline
    print("\n=== Baseline (original max_sim) ===")
    scores_base = compute_similarity_baseline(query_embeds, doc_embeds)
    metrics_base = compute_retrieval_metrics(scores_base, q_order_ids, d_order_ids, relevant_docs, k_values=[5, 10, 20, 100])
    print(f"MRR: {metrics_base['mrr']:.4f}, NDCG@10: {metrics_base['ndcg@10']:.4f}")

    # K sweep
    k_values = list(range(args.k_min, args.k_max + 1, args.k_step))
    print(f"\n=== K Sweep: k={args.k_min}~{args.k_max} (step={args.k_step}) ===")

    results = []
    best_mrr = metrics_base['mrr']
    best_k = None

    for k in k_values:
        if k > seq_d:
            k_effective = seq_d
        else:
            k_effective = k

        scores = compute_similarity_weighted_avg_tracking(query_embeds, doc_embeds, k_effective)
        metrics = compute_retrieval_metrics(scores, q_order_ids, d_order_ids, relevant_docs, k_values=[5, 10, 20, 100])

        delta_mrr = metrics['mrr'] - metrics_base['mrr']
        delta_ndcg10 = metrics['ndcg@10'] - metrics_base['ndcg@10']

        results.append({
            "k": k,
            "k_effective": k_effective,
            "mrr": metrics['mrr'],
            "ndcg@10": metrics['ndcg@10'],
            "recall@5": metrics['recall@5'],
            "recall@10": metrics['recall@10'],
            "recall@20": metrics['recall@20'],
            "recall@100": metrics['recall@100'],
            "delta_mrr": delta_mrr,
            "delta_ndcg10": delta_ndcg10,
        })

        if k % 20 == 0 or k <= 10:
            print(f"k={k:3d}: MRR={metrics['mrr']:.4f} (Δ{delta_mrr:+.4f}), "
                  f"NDCG@10={metrics['ndcg@10']:.4f} (Δ{delta_ndcg10:+.4f})")

        if metrics['mrr'] > best_mrr:
            best_mrr = metrics['mrr']
            best_k = k

    # Save all results
    results_path = os.path.join(args.output_dir, f"k_sweep_k{k_values[0]}_to_k{k_values[-1]}.json")
    with open(results_path, "w") as f:
        json.dump({
            "baseline": {k: float(v) for k, v in metrics_base.items()},
            "best_k": best_k,
            "best_mrr": float(best_mrr),
            "results": results,
        }, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Top 10 k values by MRR
    top10_mrr = sorted(results, key=lambda x: x['mrr'], reverse=True)[:10]
    print(f"\n=== Top 10 k by MRR ===")
    print(f"{'Rank':>4} {'k':>4} {'MRR':>8} {'ΔMRR':>8} {'NDCG@10':>8} {'ΔNDCG':>8}")
    for rank, r in enumerate(top10_mrr, 1):
        print(f"{rank:>4} {r['k']:>4} {r['mrr']:>8.4f} {r['delta_mrr']:>+8.4f} "
              f"{r['ndcg@10']:>8.4f} {r['delta_ndcg10']:>+8.4f}")

    # Top 10 k values by NDCG@10
    top10_ndcg = sorted(results, key=lambda x: x['ndcg@10'], reverse=True)[:10]
    print(f"\n=== Top 10 k by NDCG@10 ===")
    print(f"{'Rank':>4} {'k':>4} {'MRR':>8} {'ΔMRR':>8} {'NDCG@10':>8} {'ΔNDCG':>8}")
    for rank, r in enumerate(top10_ndcg, 1):
        print(f"{rank:>4} {r['k']:>4} {r['mrr']:>8.4f} {r['delta_mrr']:>+8.4f} "
              f"{r['ndcg@10']:>8.4f} {r['delta_ndcg10']:>+8.4f}")

    # ========== Token Distribution Analysis ==========
    print(f"\n=== Analyzing token distribution for best k={best_k} ===")

    # Track token positions for best k
    doc_positions = torch.arange(seq_d, device=doc_embeds.device, dtype=torch.float32)
    selected_counts, sampled_per_query = track_selected_token_positions(
        query_embeds, doc_embeds, best_k, sample_queries=100
    )

    # Normalize to distribution
    total_selections = selected_counts.sum()
    selection_prob = selected_counts / total_selections

    # Compute statistics
    mean_pos = np.sum(np.arange(seq_d) * selection_prob)
    std_pos = np.sqrt(np.sum((np.arange(seq_d) - mean_pos)**2 * selection_prob))
    median_pos = np.searchsorted(np.cumsum(selection_prob), 0.5)

    print(f"Doc token seq length: {seq_d}")
    print(f"Total selections: {total_selections:,}")
    print(f"Mean selected position: {mean_pos:.1f} (relative: {mean_pos/seq_d:.1%})")
    print(f"Std dev: {std_pos:.1f}")
    print(f"Median position: {median_pos} (relative: {median_pos/seq_d:.1%})")

    # Percentiles
    cumsum = np.cumsum(selection_prob)
    percentiles = [25, 50, 75, 90, 95, 99]
    for p in percentiles:
        idx = np.searchsorted(cumsum, p/100)
        print(f"  P{p}: position {idx} (relative: {idx/seq_d:.1%})")

    # First vs last comparison
    n_edge = seq_d // 4
    first_quarter = selection_prob[:n_edge].sum()
    last_quarter = selection_prob[-n_edge:].sum()
    middle_half = selection_prob[n_edge:-n_edge].sum()
    print(f"\nSelection probability by doc region:")
    print(f"  First 25%: {first_quarter:.2%}")
    print(f"  Middle 50%: {middle_half:.2%}")
    print(f"  Last 25%: {last_quarter:.2%}")

    # Plot selection distribution
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Selection count per position
    ax = axes[0, 0]
    ax.plot(np.arange(seq_d), selected_counts, color="steelblue", linewidth=1)
    ax.set_xlabel("Doc Token Position")
    ax.set_ylabel("Selection Count")
    ax.set_title(f"Which doc tokens are selected? (k={best_k})\nTotal: {total_selections:,} selections")
    ax.axvline(x=mean_pos, color="red", linestyle="--", label=f"mean={mean_pos:.0f}")
    ax.legend()

    # Plot 2: Selection probability density
    ax = axes[0, 1]
    ax.plot(np.arange(seq_d), selection_prob, color="coral", linewidth=1.5)
    ax.fill_between(np.arange(seq_d), selection_prob, alpha=0.3)
    ax.set_xlabel("Doc Token Position")
    ax.set_ylabel("Selection Probability")
    ax.set_title(f"Selection probability distribution\nmean={mean_pos:.0f}, std={std_pos:.0f}, median={median_pos}")
    ax.axvline(x=median_pos, color="green", linestyle="--", alpha=0.7, label=f"median={median_pos}")
    ax.legend()

    # Plot 3: Cumulative distribution
    ax = axes[1, 0]
    cumprob = np.cumsum(selection_prob)
    ax.plot(np.arange(seq_d), cumprob, color="purple", linewidth=2)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.7)
    ax.axvline(x=median_pos, color="green", linestyle="--", alpha=0.7)
    ax.set_xlabel("Doc Token Position")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(f"Cumulative selection distribution\nMedian at position {median_pos} ({median_pos/seq_d:.1%})")
    ax.grid(True, alpha=0.3)

    # Plot 4: MRR/NDCG@10 vs k
    ax = axes[1, 1]
    ks = [r['k'] for r in results]
    mrrs = [r['mrr'] for r in results]
    ndcgs = [r['ndcg@10'] for r in results]
    ax.plot(ks, mrrs, label="MRR", color="blue", linewidth=1.5)
    ax.plot(ks, ndcgs, label="NDCG@10", color="orange", linewidth=1.5)
    ax.axhline(y=metrics_base['mrr'], color="blue", linestyle="--", alpha=0.5, label=f"baseline MRR={metrics_base['mrr']:.4f}")
    ax.axhline(y=metrics_base['ndcg@10'], color="orange", linestyle="--", alpha=0.5, label=f"baseline NDCG@10={metrics_base['ndcg@10']:.4f}")
    ax.axvline(x=best_k, color="red", linestyle=":", alpha=0.7, label=f"best k={best_k}")
    ax.set_xlabel("k")
    ax.set_ylabel("Score")
    ax.set_title("MRR & NDCG@10 vs k")
    ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, f"k_sweep_analysis_k{best_k}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved plot: {plot_path}")

    # Save token distribution data
    dist_path = os.path.join(args.output_dir, f"token_distribution_k{best_k}.json")
    with open(dist_path, "w") as f:
        json.dump({
            "best_k": best_k,
            "seq_d": seq_d,
            "total_selections": int(total_selections),
            "mean_pos": float(mean_pos),
            "mean_pos_relative": float(mean_pos / seq_d),
            "std_pos": float(std_pos),
            "median_pos": int(median_pos),
            "median_pos_relative": float(median_pos / seq_d),
            "selection_prob": selection_prob.tolist(),
            "percentiles": {p: int(np.searchsorted(np.cumsum(selection_prob), p/100)) for p in percentiles},
            "region_probs": {
                "first_25pct": float(first_quarter),
                "middle_50pct": float(middle_half),
                "last_25pct": float(last_quarter),
            }
        }, f, indent=2)
    print(f"Saved distribution: {dist_path}")

    print(f"\n=== Summary ===")
    print(f"Best k: {best_k} (MRR={best_mrr:.4f}, vs baseline {metrics_base['mrr']:.4f})")
    print(f"Best k relative position: {best_k/seq_d:.1%} of doc")
    print(f"Mean selected token position: {mean_pos:.1f} ({mean_pos/seq_d:.1%})")
    print(f"Median selected token position: {median_pos} ({median_pos/seq_d:.1%})")


if __name__ == "__main__":
    main()
