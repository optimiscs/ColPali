#!/usr/bin/env python3
"""结合分层思想 + token压缩的最优策略"""
import sys
sys.path.insert(0, "/home/moxu/MMRAG/otherExp/colpali/mteb")

import json
import numpy as np
import torch
from pathlib import Path
from accelerate import Accelerator
from collections import defaultdict
from mteb.get_tasks import get_tasks
from mteb.similarity_functions import _convert_to_tensor
from datasets import load_dataset


def compute_ndcg(scores_matrix, q_order_ids, d_order_ids, relevant_docs, ndcg_k=10):
    ndcg_sum = 0.0
    n_evaluated = 0
    for i, qid in enumerate(q_order_ids):
        if qid not in relevant_docs:
            continue
        relevant = relevant_docs[qid]
        if not relevant:
            continue
        n_evaluated += 1

        ranking = np.argsort(-scores_matrix[i])
        sorted_rel = sorted(relevant.values(), reverse=True)

        dcg = 0.0
        for rank_idx, doc_idx in enumerate(ranking[:ndcg_k], 1):
            did = d_order_ids[doc_idx]
            if did in relevant:
                rel = relevant[did]
                dcg += rel / np.log2(rank_idx + 1)

        idcg = 0.0
        for rank_idx, rel in enumerate(sorted_rel[:ndcg_k], 1):
            idcg += rel / np.log2(rank_idx + 1)

        if idcg > 0:
            ndcg_sum += dcg / idcg

    return ndcg_sum / n_evaluated if n_evaluated > 0 else 0.0


def stratified_topk_max_sim(q_embeds, d_embeds, n_regions, k_per_region, region_weights=None, corpus_chunk_size=200, query_step=4):
    """从每个region取top-k，然后加权组合。

    比单纯top-k更公平地覆盖各region。
    """
    num_docs = d_embeds.size(0)
    seq_d = d_embeds.size(1)

    if region_weights is None:
        region_weights = [1.0 / n_regions] * n_regions

    # Calculate regions
    region_size = seq_d // n_regions
    regions = []
    for i in range(n_regions):
        start = i * region_size
        end = seq_d if i == n_regions - 1 else (i + 1) * region_size
        regions.append((start, end))

    final_scores = None
    for chunk_start in range(0, num_docs, corpus_chunk_size):
        chunk_end = min(chunk_start + corpus_chunk_size, num_docs)
        d_chunk = d_embeds[chunk_start:chunk_end]
        chunk_scores_list = []

        for q_start in range(0, q_embeds.size(0), query_step):
            q_end = min(q_start + query_step, q_embeds.size(0))
            q_batch = q_embeds[q_start:q_end]

            scores = torch.einsum("ash,bth->abst", q_batch, d_chunk)
            region_max_scores = []

            for ri, (start, end) in enumerate(regions):
                region_scores = scores[:, :, :, start:end]
                region_len = end - start
                k_effective = min(k_per_region, region_len)
                region_topk = region_scores.topk(k=k_effective, axis=-1).values
                # Apply exponential weighting within region
                region_pos = torch.arange(region_len, device=d_embeds.device, dtype=torch.float32)[:k_effective]
                region_w = torch.exp(region_pos / region_len)
                region_w = region_w / region_w.sum()
                weighted = (region_topk * region_w.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(axis=-1)
                # Weighted sum over query tokens
                region_max = weighted.sum(axis=-1) * region_weights[ri]
                region_max_scores.append(region_max)

            combined = sum(region_max_scores)
            chunk_scores_list.append(combined)

        chunk_scores = torch.cat(chunk_scores_list, dim=0)
        if final_scores is None:
            final_scores = chunk_scores
        else:
            final_scores = torch.cat([final_scores, chunk_scores], dim=1)
    return final_scores


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


def compute_similarity_base(q_embeds, d_embeds, corpus_chunk_size=200, query_step=4):
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


def compute_similarity_subset(q_embeds, d_embeds, token_range, corpus_chunk_size=200, query_step=4):
    num_docs = d_embeds.size(0)
    d_subset = d_embeds[:, token_range[0]:token_range[1], :]
    return compute_similarity_base(q_embeds, d_subset, corpus_chunk_size, query_step)


def main():
    accelerator = Accelerator()
    if accelerator.num_processes != 1:
        raise RuntimeError("Only single process")

    cache_dir = Path("./embeddings_cache")
    task_name = "Vidore3PhysicsRetrieval.v2"

    print("Loading embeddings...")
    q_loaded = torch.load(cache_dir / f"{task_name}_queries.pt")
    d_loaded = torch.load(cache_dir / f"{task_name}_docs.pt")
    query_embeds = q_loaded["data"].to(accelerator.device).float() * q_loaded["scale"].to(accelerator.device)
    doc_embeds = d_loaded["data"].to(accelerator.device).float() * d_loaded["scale"].to(accelerator.device)
    q_order_ids = q_loaded["ids"]
    d_order_ids = d_loaded["ids"]

    tasks = get_tasks(tasks=[task_name])
    task = tasks[0]
    dataset_path = task.metadata.dataset["path"]
    revision = task.metadata.dataset["revision"]
    qrels_ds = load_dataset(dataset_path, name="english-qrels", revision=revision, split="test")
    relevant_docs = defaultdict(dict)
    for row in qrels_ds:
        relevant_docs[row["query-id"]][row["corpus-id"]] = row["score"]

    seq_d = doc_embeds.size(1)
    print(f"Query: {query_embeds.shape}, Doc: {doc_embeds.shape}, seq_d={seq_d}")

    # Baseline
    scores_full = compute_similarity_base(query_embeds, doc_embeds)
    ndcg_full = compute_ndcg(scores_full.cpu().numpy(), q_order_ids, d_order_ids, relevant_docs)
    print(f"\nBaseline (full): NDCG@10 = {ndcg_full:.4f}")

    results = []

    # Test: Stratified top-k from each region
    print("\n=== Stratified Top-K (k_per_region from each region) ===")
    print(f"{'n_reg':>6} {'k/reg':>6} {'Total':>6} {'NDCG@10':>10} {'Δ':>8}")
    print("-" * 45)

    configs = [
        # (n_regions, k_per_region, region_weights)
        (2, 2, [0.0, 1.0]),      # 2 regions, k=2 per region, only use last region
        (2, 4, [0.0, 1.0]),
        (2, 8, [0.0, 1.0]),
        (2, 16, [0.0, 1.0]),
        (2, 32, [0.0, 1.0]),
        (3, 2, [0.0, 0.0, 1.0]),  # 3 regions, only last
        (3, 4, [0.0, 0.0, 1.0]),
        (3, 8, [0.0, 0.0, 1.0]),
        (3, 16, [0.0, 0.0, 1.0]),
        (4, 1, [0.0, 0.0, 0.0, 1.0]),  # 4 regions, only last
        (4, 2, [0.0, 0.0, 0.0, 1.0]),
        (4, 4, [0.0, 0.0, 0.0, 1.0]),
        (4, 8, [0.0, 0.0, 0.0, 1.0]),
        (4, 16, [0.0, 0.0, 0.0, 1.0]),
        (4, 32, [0.0, 0.0, 0.0, 1.0]),
        # With small weight on earlier regions
        (4, 2, [0.05, 0.05, 0.1, 0.8]),
        (4, 4, [0.05, 0.05, 0.1, 0.8]),
        (4, 8, [0.05, 0.05, 0.1, 0.8]),
        (3, 4, [0.05, 0.1, 0.85]),
        (3, 8, [0.05, 0.1, 0.85]),
        (2, 8, [0.1, 0.9]),
        (2, 16, [0.1, 0.9]),
    ]

    for n_reg, k_pr, r_weights in configs:
        scores = stratified_topk_max_sim(query_embeds, doc_embeds, n_reg, k_pr, r_weights)
        ndcg = compute_ndcg(scores.cpu().numpy(), q_order_ids, d_order_ids, relevant_docs)
        total_tokens = n_reg * k_pr
        print(f"{n_reg:>6} {k_pr:>6} {total_tokens:>6} {ndcg:>10.4f} {ndcg-ndcg_full:>+8.4f}")
        results.append({
            "strategy": f"strat_topk_{n_reg}x{k_pr}",
            "n_regions": n_reg, "k_per_region": k_pr,
            "region_weights": r_weights, "total_tokens": total_tokens,
            "ndcg10": ndcg, "delta": ndcg - ndcg_full
        })

    # Compare with simple "last-k" approach
    print("\n=== Comparison: Last-K vs Stratified ===")
    print(f"{'K':>4} {'Last-K':>10} {'Strat-4xK':>12} {'Diff':>8}")
    print("-" * 40)

    for k in [1, 2, 4, 8, 16, 32, 64]:
        # Last-k
        scores_lastk = compute_similarity_subset(query_embeds, doc_embeds, (seq_d - k, seq_d))
        ndcg_lastk = compute_ndcg(scores_lastk.cpu().numpy(), q_order_ids, d_order_ids, relevant_docs)

        # Stratified: 4 regions, k tokens each = 4k total
        scores_strat = stratified_topk_max_sim(query_embeds, doc_embeds, 4, k, [0.0, 0.0, 0.0, 1.0])
        ndcg_strat = compute_ndcg(scores_strat.cpu().numpy(), q_order_ids, d_order_ids, relevant_docs)

        diff = ndcg_strat - ndcg_lastk
        print(f"{k:>4} {ndcg_lastk:>10.4f} {ndcg_strat:>12.4f} {diff:>+8.4f}")

    # Find best
    best = max(results, key=lambda x: x["ndcg10"])
    print(f"\n=== Best ===")
    print(f"Strategy: {best['strategy']}")
    print(f"NDCG@10: {best['ndcg10']:.4f} (Δ={best['delta']:+.4f})")
    print(f"Total tokens: {best['total_tokens']} ({best['total_tokens']/seq_d:.1%} of {seq_d})")

    # Save
    with open("./k_sweep_results/compression_strategies_v2.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
