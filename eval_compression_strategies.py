#!/usr/bin/env python3
"""测试各种token压缩策略，找出最优方法"""
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    """Use only token_range [start, end) of doc tokens."""
    num_docs = d_embeds.size(0)
    d_subset = d_embeds[:, token_range[0]:token_range[1], :]
    return compute_similarity_base(q_embeds, d_subset, corpus_chunk_size, query_step)


def weighted_max_sim(q_embeds, d_embeds, k, doc_weights, corpus_chunk_size=200, query_step=4):
    """Top-k weighted max_sim with custom doc_weights."""
    num_docs = d_embeds.size(0)
    final_scores = None
    for chunk_start in range(0, num_docs, corpus_chunk_size):
        chunk_end = min(chunk_start + corpus_chunk_size, num_docs)
        d_chunk = d_embeds[chunk_start:chunk_end]
        chunk_scores_list = []
        for q_start in range(0, q_embeds.size(0), query_step):
            q_end = min(q_start + query_step, q_embeds.size(0))
            q_batch = q_embeds[q_start:q_end]
            scores = torch.einsum("ash,bth->abst", q_batch, d_chunk)
            topk_scores, _ = scores.topk(k=k, axis=-1)
            # Apply weights to top-k positions
            topk_w = doc_weights[-k:].unsqueeze(0).unsqueeze(0).unsqueeze(0)
            weighted = topk_scores * topk_w
            avg_scores = weighted.sum(axis=-1) / k
            final_q = avg_scores.sum(axis=-1)
            chunk_scores_list.append(final_q)
        chunk_scores = torch.cat(chunk_scores_list, dim=0)
        if final_scores is None:
            final_scores = chunk_scores
        else:
            final_scores = torch.cat([final_scores, chunk_scores], dim=1)
    return final_scores


def hybrid_max_sim(q_embeds, d_embeds, first_k, last_k, first_weight, last_weight, corpus_chunk_size=200, query_step=4):
    """Combine first_k and last_k doc tokens with separate weights."""
    num_docs = d_embeds.size(0)
    seq_d = d_embeds.size(1)
    final_scores = None

    for chunk_start in range(0, num_docs, corpus_chunk_size):
        chunk_end = min(chunk_start + corpus_chunk_size, num_docs)
        d_chunk = d_embeds[chunk_start:chunk_end]
        chunk_scores_list = []
        for q_start in range(0, q_embeds.size(0), query_step):
            q_end = min(q_start + query_step, q_embeds.size(0))
            q_batch = q_embeds[q_start:q_end]

            # All scores
            scores = torch.einsum("ash,bth->abst", q_batch, d_chunk)  # (n_q, n_d, seq_q, seq_d)

            # First k tokens (from start)
            first_scores = scores[:, :, :, :first_k]
            first_max = first_scores.max(axis=-1).values.sum(axis=-1)

            # Last k tokens (from end)
            last_scores = scores[:, :, :, seq_d-last_k:]
            last_max = last_scores.max(axis=-1).values.sum(axis=-1)

            # Weighted combination
            combined = first_weight * first_max + last_weight * last_max
            chunk_scores_list.append(combined)

        chunk_scores = torch.cat(chunk_scores_list, dim=0)
        if final_scores is None:
            final_scores = chunk_scores
        else:
            final_scores = torch.cat([final_scores, chunk_scores], dim=1)
    return final_scores


def stratified_max_sim(q_embeds, d_embeds, n_regions, tokens_per_region, region_weights=None, corpus_chunk_size=200, query_step=4):
    """Sample tokens from n regions evenly, with custom region weights."""
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
                region_max = region_scores.max(axis=-1).values.sum(axis=-1)
                region_max_scores.append(region_max * region_weights[ri])

            combined = sum(region_max_scores)
            chunk_scores_list.append(combined)

        chunk_scores = torch.cat(chunk_scores_list, dim=0)
        if final_scores is None:
            final_scores = chunk_scores
        else:
            final_scores = torch.cat([final_scores, chunk_scores], dim=1)
    return final_scores


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

    results = []

    # Baseline
    print("\n=== Strategy 1: Last-K tokens only ===")
    print(f"{'K':>4} {'NDCG@10':>10} {'vs Full':>10} {'Tokens':>8}")
    print("-" * 40)
    scores_full = compute_similarity_base(query_embeds, doc_embeds)
    ndcg_full = compute_ndcg(scores_full.cpu().numpy(), q_order_ids, d_order_ids, relevant_docs)
    print(f"Full: NDCG@10 = {ndcg_full:.4f}")

    for k in [4, 8, 16, 32, 64, 128, 256]:
        scores = compute_similarity_subset(query_embeds, doc_embeds, (seq_d - k, seq_d))
        ndcg = compute_ndcg(scores.cpu().numpy(), q_order_ids, d_order_ids, relevant_docs)
        print(f"last {k:>3}: NDCG@10 = {ndcg:.4f} (Δ={ndcg-ndcg_full:+.4f})")
        results.append({"strategy": f"last_{k}", "k": k, "ndcg10": ndcg, "delta": ndcg - ndcg_full})

    # Strategy 2: Hybrid first + last
    print("\n=== Strategy 2: Hybrid (first + last with weight) ===")
    print(f"{'first':>6}+{'last':>4} {'w_first':>8} {'w_last':>8} {'NDCG@10':>10} {'Δ':>8}")
    print("-" * 50)

    hybrid_configs = [
        (4, 4, 0.0, 1.0),
        (4, 8, 0.0, 1.0),
        (4, 16, 0.0, 1.0),
        (4, 32, 0.0, 1.0),
        (0, 32, 0.0, 1.0),
        (0, 64, 0.0, 1.0),
        (0, 128, 0.0, 1.0),
        (8, 24, 0.1, 0.9),
        (8, 24, 0.2, 0.8),
        (16, 16, 0.1, 0.9),
        (16, 16, 0.2, 0.8),
        (8, 32, 0.05, 0.95),
        (4, 28, 0.05, 0.95),
    ]

    for first_k, last_k, w_first, w_last in hybrid_configs:
        if first_k == 0:
            scores = compute_similarity_subset(query_embeds, doc_embeds, (seq_d - last_k, seq_d))
            ndcg = compute_ndcg(scores.cpu().numpy(), q_order_ids, d_order_ids, relevant_docs)
        else:
            scores = hybrid_max_sim(query_embeds, doc_embeds, first_k, last_k, w_first, w_last)
            ndcg = compute_ndcg(scores.cpu().numpy(), q_order_ids, d_order_ids, relevant_docs)
        print(f"{first_k:>5}+{last_k:>4} {w_first:>8.2f} {w_last:>8.2f} {ndcg:>10.4f} {ndcg-ndcg_full:>+8.4f}")
        results.append({"strategy": f"hybrid_{first_k}+{last_k}", "first_k": first_k, "last_k": last_k,
                       "w_first": w_first, "w_last": w_last, "ndcg10": ndcg, "delta": ndcg - ndcg_full})

    # Strategy 3: Stratified (n regions evenly)
    print("\n=== Strategy 3: Stratified (equal regions) ===")
    print(f"{'n_regions':>10} {'per_region':>10} {'total':>8} {'NDCG@10':>10} {'Δ':>8}")
    print("-" * 55)

    for n_regions in [2, 3, 4, 6, 8]:
        per_region = seq_d // n_regions
        total = per_region * n_regions
        # Weight later regions more
        region_weights = [np.exp(i * 0.5) for i in range(n_regions)]
        region_weights = [w / sum(region_weights) for w in region_weights]
        scores = stratified_max_sim(query_embeds, doc_embeds, n_regions, per_region, region_weights)
        ndcg = compute_ndcg(scores.cpu().numpy(), q_order_ids, d_order_ids, relevant_docs)
        print(f"{n_regions:>10} {per_region:>10} {total:>8} {ndcg:>10.4f} {ndcg-ndcg_full:>+8.4f}")
        results.append({"strategy": f"stratified_{n_regions}", "n_regions": n_regions,
                       "per_region": per_region, "ndcg10": ndcg, "delta": ndcg - ndcg_full})

    # Strategy 4: Weighted top-k with varying alpha
    print("\n=== Strategy 4: Weighted top-k (exp position weighting) ===")
    doc_weights_full = torch.exp(torch.arange(seq_d, device=doc_embeds.device, dtype=torch.float32) / seq_d)
    doc_weights_full = doc_weights_full / doc_weights_full.sum()

    print(f"{'k':>4} {'NDCG@10':>10} {'Δ':>8}")
    print("-" * 25)
    for k in [4, 8, 16, 32, 64, 128, 256]:
        scores = weighted_max_sim(query_embeds, doc_embeds, k, doc_weights_full)
        ndcg = compute_ndcg(scores.cpu().numpy(), q_order_ids, d_order_ids, relevant_docs)
        print(f"{k:>4} {ndcg:>10.4f} {ndcg-ndcg_full:>+8.4f}")
        results.append({"strategy": f"weighted_k{k}", "k": k, "ndcg10": ndcg, "delta": ndcg - ndcg_full})

    # Save results
    with open("./k_sweep_results/compression_strategies.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to ./k_sweep_results/compression_strategies.json")

    # Find best
    best = max(results, key=lambda x: x.get("ndcg10", 0))
    print(f"\n=== Best ===")
    print(f"Strategy: {best}")


if __name__ == "__main__":
    main()
