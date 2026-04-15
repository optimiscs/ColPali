#!/usr/bin/env python3
"""位置指数加权 + 按位置比例采样（不用top-k筛选）"""
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


def position_weighted_max_sim(q_embeds, d_embeds, alpha, corpus_chunk_size=200, query_step=4):
    """对所有doc token做指数位置加权，然后求max后求和。

    不做top-k筛选，直接对所有token加权。
    alpha越大，后期token权重越高。
    """
    num_docs = d_embeds.size(0)
    seq_d = d_embeds.size(1)

    # 指数位置权重
    doc_positions = torch.arange(seq_d, device=d_embeds.device, dtype=torch.float32)
    doc_weights = torch.exp(alpha * doc_positions / seq_d)
    doc_weights = doc_weights / doc_weights.sum()

    final_scores = None
    for chunk_start in range(0, num_docs, corpus_chunk_size):
        chunk_end = min(chunk_start + corpus_chunk_size, num_docs)
        d_chunk = d_embeds[chunk_start:chunk_end]
        chunk_scores_list = []

        for q_start in range(0, q_embeds.size(0), query_step):
            q_end = min(q_start + query_step, q_embeds.size(0))
            q_batch = q_embeds[q_start:q_end]

            # scores: (n_q, n_d, seq_q, seq_d)
            scores = torch.einsum("ash,bth->abst", q_batch, d_chunk)

            # 乘以位置权重
            weighted_scores = scores * doc_weights.unsqueeze(0).unsqueeze(0).unsqueeze(0)

            # Max over doc tokens
            max_scores = weighted_scores.max(axis=-1).values

            # Sum over query tokens
            final_q = max_scores.sum(axis=-1)
            chunk_scores_list.append(final_q)

        chunk_scores = torch.cat(chunk_scores_list, dim=0)
        if final_scores is None:
            final_scores = chunk_scores
        else:
            final_scores = torch.cat([final_scores, chunk_scores], dim=1)
    return final_scores


def sampled_position_weighted_max_sim(q_embeds, d_embeds, alpha, n_samples, corpus_chunk_size=200, query_step=4):
    """随机采样n_samples个token，按位置概率加权。"""
    num_docs = d_embeds.size(0)
    seq_d = d_embeds.size(1)

    # 采样概率（按指数位置加权）
    doc_positions = torch.arange(seq_d, device=d_embeds.device, dtype=torch.float32)
    sample_probs = torch.exp(alpha * doc_positions / doc_positions.max())
    sample_probs = sample_probs / sample_probs.sum()

    # 采样 indices: (n_samples,)
    sample_indices = torch.multinomial(sample_probs, n_samples, replacement=True)

    final_scores = None
    for chunk_start in range(0, num_docs, corpus_chunk_size):
        chunk_end = min(chunk_start + corpus_chunk_size, num_docs)
        d_chunk = d_embeds[chunk_start:chunk_end]
        chunk_scores_list = []

        for q_start in range(0, q_embeds.size(0), query_step):
            q_end = min(q_start + query_step, q_embeds.size(0))
            q_batch = q_embeds[q_start:q_end]

            scores = torch.einsum("ash,bth->abst", q_batch, d_chunk)

            # 取采样的token
            sampled_scores = scores[:, :, :, sample_indices]  # (n_q, n_d, seq_q, n_samples)

            # 采样权重
            sample_weights = sample_probs[sample_indices]
            sample_weights = sample_weights / sample_weights.sum()

            # 加权max
            weighted_scores = sampled_scores * sample_weights.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            max_scores = weighted_scores.max(axis=-1).values
            final_q = max_scores.sum(axis=-1)
            chunk_scores_list.append(final_q)

        chunk_scores = torch.cat(chunk_scores_list, dim=0)
        if final_scores is None:
            final_scores = chunk_scores
        else:
            final_scores = torch.cat([final_scores, chunk_scores], dim=1)
    return final_scores


def topk_position_weighted_max_sim(q_embeds, d_embeds, alpha, k, corpus_chunk_size=200, query_step=4):
    """Top-k选择 + 位置指数加权。"""
    num_docs = d_embeds.size(0)
    seq_d = d_embeds.size(1)

    doc_positions = torch.arange(seq_d, device=d_embeds.device, dtype=torch.float32)
    doc_weights = torch.exp(alpha * doc_positions / seq_d)
    doc_weights = doc_weights / doc_weights.sum()

    final_scores = None
    for chunk_start in range(0, num_docs, corpus_chunk_size):
        chunk_end = min(chunk_start + corpus_chunk_size, num_docs)
        d_chunk = d_embeds[chunk_start:chunk_end]
        chunk_scores_list = []

        for q_start in range(0, q_embeds.size(0), query_step):
            q_end = min(q_start + query_step, q_embeds.size(0))
            q_batch = q_embeds[q_start:q_end]

            scores = torch.einsum("ash,bth->abst", q_batch, d_chunk)

            # Top-k
            topk_scores, _ = scores.topk(k=k, axis=-1)

            # 位置权重（只取最后k个位置的权重，因为top-k选的是得分最高的）
            topk_weights = doc_weights[-k:].unsqueeze(0).unsqueeze(0).unsqueeze(0)
            weighted_topk = topk_scores * topk_weights

            avg_scores = weighted_topk.sum(axis=-1) / k
            final_q = avg_scores.sum(axis=-1)
            chunk_scores_list.append(final_q)

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

    # Strategy 1: Position-weighted (no top-k, just multiply by position weights)
    print("\n=== Strategy 1: Position-weighted only (no top-k) ===")
    print(f"{'alpha':>6} {'NDCG@10':>10} {'Δ':>8}")
    print("-" * 30)

    for alpha in [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
        scores = position_weighted_max_sim(query_embeds, doc_embeds, alpha)
        ndcg = compute_ndcg(scores.cpu().numpy(), q_order_ids, d_order_ids, relevant_docs)
        print(f"{alpha:>6.1f} {ndcg:>10.4f} {ndcg-ndcg_full:>+8.4f}")
        results.append({"strategy": f"pos_weighted_alpha{alpha}", "alpha": alpha, "ndcg10": ndcg, "delta": ndcg - ndcg_full})

    # Strategy 2: Top-k + position-weighted (our original weighted_avg)
    print("\n=== Strategy 2: Top-k + Position-weighted (original) ===")
    print(f"{'k':>4} {'alpha':>6} {'NDCG@10':>10} {'Δ':>8}")
    print("-" * 35)

    for k in [4, 8, 16]:
        for alpha in [0.5, 1.0, 1.5, 2.0, 3.0]:
            scores = topk_position_weighted_max_sim(query_embeds, doc_embeds, alpha, k)
            ndcg = compute_ndcg(scores.cpu().numpy(), q_order_ids, d_order_ids, relevant_docs)
            print(f"{k:>4} {alpha:>6.1f} {ndcg:>10.4f} {ndcg-ndcg_full:>+8.4f}")
            results.append({"strategy": f"topk{k}_alpha{alpha}", "k": k, "alpha": alpha, "ndcg10": ndcg, "delta": ndcg - ndcg_full})

    # Strategy 3: Sampled position-weighted
    print("\n=== Strategy 3: Sampled (by position probability) ===")
    print(f"{'n_samples':>10} {'alpha':>6} {'NDCG@10':>10} {'Δ':>8}")
    print("-" * 40)

    for n_samples in [4, 8, 16, 32, 64]:
        for alpha in [1.0, 2.0, 3.0, 5.0]:
            scores = sampled_position_weighted_max_sim(query_embeds, doc_embeds, alpha, n_samples)
            ndcg = compute_ndcg(scores.cpu().numpy(), q_order_ids, d_order_ids, relevant_docs)
            print(f"{n_samples:>10} {alpha:>6.1f} {ndcg:>10.4f} {ndcg-ndcg_full:>+8.4f}")
            results.append({"strategy": f"sampled{n_samples}_alpha{alpha}", "n_samples": n_samples, "alpha": alpha, "ndcg10": ndcg, "delta": ndcg - ndcg_full})

    # Find best
    best = max(results, key=lambda x: x["ndcg10"])
    print(f"\n=== BEST ===")
    print(f"Strategy: {best['strategy']}")
    print(f"NDCG@10: {best['ndcg10']:.4f} (Δ={best['delta']:+.4f})")

    # Save
    with open("./k_sweep_results/position_weighted_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
