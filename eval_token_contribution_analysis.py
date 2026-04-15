#!/usr/bin/env python3
"""分析前面 token 和后面 token 对最终 score 的贡献 - 分块处理避免OOM"""
import sys
sys.path.insert(0, "/home/moxu/MMRAG/otherExp/colpali/mteb")

import json
import numpy as np
import torch
from pathlib import Path
from accelerate import Accelerator
from collections import defaultdict
from mteb.get_tasks import get_tasks
from datasets import load_dataset


def weighted_avg_chunked_with_tracking(q_embeds, d_embeds, k, corpus_chunk_size=200, query_step=4):
    """Weighted average with contribution tracking, processed in chunks."""
    num_docs = d_embeds.size(0)
    seq_d = d_embeds.size(1)
    doc_positions = torch.arange(seq_d, device=d_embeds.device, dtype=torch.float32)
    doc_weights = torch.exp(doc_positions / doc_positions.max())
    doc_weights = doc_weights / doc_weights.sum()

    mid_point = seq_d // 2

    # Use same chunking as compute_similarity_exp for consistency
    all_first_scores = []
    all_second_scores = []
    all_first_raw = []
    all_second_raw = []
    all_first_cnt = []
    all_second_cnt = []
    all_total = []

    for chunk_start in range(0, num_docs, corpus_chunk_size):
        chunk_end = min(chunk_start + corpus_chunk_size, num_docs)
        d_chunk = d_embeds[chunk_start:chunk_end]
        chunk_first_scores = []
        chunk_second_scores = []
        chunk_first_raw = []
        chunk_second_raw = []
        chunk_first_cnt = []
        chunk_second_cnt = []
        chunk_total = []

        for q_start in range(0, q_embeds.size(0), query_step):
            q_end = min(q_start + query_step, q_embeds.size(0))
            q_batch = q_embeds[q_start:q_end]

            scores = torch.einsum("ash,bth->abst", q_batch, d_chunk)
            topk_scores, topk_idxs = scores.topk(k=k, axis=-1)

            topk_weights = doc_weights[-k:].unsqueeze(0).unsqueeze(0).unsqueeze(0)
            weighted_topk = topk_scores * topk_weights

            # Split by position
            first_mask = (topk_idxs < mid_point).float()
            second_mask = 1.0 - first_mask

            # Weighted contributions
            first_contrib = (weighted_topk * first_mask).sum(axis=-1)
            second_contrib = (weighted_topk * second_mask).sum(axis=-1)

            # Raw scores
            first_raw = (topk_scores * first_mask).sum(axis=-1)
            second_raw = (topk_scores * second_mask).sum(axis=-1)

            # Count
            first_cnt = first_mask.sum(axis=-1)
            second_cnt = second_mask.sum(axis=-1)

            # Total
            total_contrib = first_contrib + second_contrib

            # Append (query_batch_size, chunk_size)
            chunk_first_scores.append(first_contrib.sum(axis=2).cpu())
            chunk_second_scores.append(second_contrib.sum(axis=2).cpu())
            chunk_first_raw.append(first_raw.sum(axis=2).cpu())
            chunk_second_raw.append(second_raw.sum(axis=2).cpu())
            chunk_first_cnt.append(first_cnt.cpu())
            chunk_second_cnt.append(second_cnt.cpu())
            chunk_total.append(total_contrib.sum(axis=2).cpu())

        # Concatenate within chunk (query batches -> query_total, doc_chunk)
        all_first_scores.append(torch.cat(chunk_first_scores, dim=0))
        all_second_scores.append(torch.cat(chunk_second_scores, dim=0))
        all_first_raw.append(torch.cat(chunk_first_raw, dim=0))
        all_second_raw.append(torch.cat(chunk_second_raw, dim=0))
        all_first_cnt.append(torch.cat(chunk_first_cnt, dim=0))
        all_second_cnt.append(torch.cat(chunk_second_cnt, dim=0))
        all_total.append(torch.cat(chunk_total, dim=0))

    # Concatenate across chunks (doc dimension)
    first_half_scores = torch.cat(all_first_scores, dim=1)
    second_half_scores = torch.cat(all_second_scores, dim=1)
    first_half_raw = torch.cat(all_first_raw, dim=1)
    second_half_raw = torch.cat(all_second_raw, dim=1)
    first_half_count = torch.cat(all_first_cnt, dim=1)
    second_half_count = torch.cat(all_second_cnt, dim=1)
    total = torch.cat(all_total, dim=1)

    first_ratio = first_half_scores / (total + 1e-10)

    return {
        "first_half_scores": first_half_scores.numpy(),
        "second_half_scores": second_half_scores.numpy(),
        "first_half_ratio": first_ratio.numpy(),
        "first_half_raw": first_half_raw.numpy(),
        "second_half_raw": second_half_raw.numpy(),
        "first_half_count": first_half_count.numpy(),
        "second_half_count": second_half_count.numpy(),
        "total": total.numpy(),
    }


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


def analyze_region_contributions(q_embeds, d_embeds, relevant_docs, q_order_ids, d_order_ids, k_values):
    print(f"\n{'k':>4} {'NDCG@10':>10} {'NDCG(1st)':>10} {'NDCG(2nd)':>10} {'1st ratio':>10} {'2nd ratio':>10} {'1st cnt':>8} {'2nd cnt':>8}")
    print("-" * 80)

    results = {}
    for k in k_values:
        contrib = weighted_avg_chunked_with_tracking(q_embeds, d_embeds, k)

        final_scores = contrib["first_half_scores"] + contrib["second_half_scores"]
        ndcg_full = compute_ndcg(final_scores, q_order_ids, d_order_ids, relevant_docs)
        ndcg_first = compute_ndcg(contrib["first_half_scores"], q_order_ids, d_order_ids, relevant_docs)
        ndcg_second = compute_ndcg(contrib["second_half_scores"], q_order_ids, d_order_ids, relevant_docs)

        avg_first_ratio = np.mean(contrib["first_half_ratio"])
        avg_first_count = np.mean(contrib["first_half_count"])
        avg_second_count = np.mean(contrib["second_half_count"])

        print(f"{k:>4} {ndcg_full:>10.4f} {ndcg_first:>10.4f} {ndcg_second:>10.4f} "
              f"{avg_first_ratio:>10.2%} {1-avg_first_ratio:>10.2%} "
              f"{avg_first_count:>8.2f} {avg_second_count:>8.2f}")

        results[k] = {
            "ndcg_full": ndcg_full,
            "ndcg_first_only": ndcg_first,
            "ndcg_second_only": ndcg_second,
            "avg_first_ratio": float(avg_first_ratio),
            "avg_first_count": float(avg_first_count),
            "avg_second_count": float(avg_second_count),
        }

    return results


def analyze_tertiles(q_embeds, d_embeds, relevant_docs, q_order_ids, d_order_ids, k):
    """Break down by three regions (first/middle/last third)."""
    num_docs = d_embeds.size(0)
    seq_d = d_embeds.size(1)
    doc_positions = torch.arange(seq_d, device=d_embeds.device, dtype=torch.float32)
    doc_weights = torch.exp(doc_positions / doc_positions.max())
    doc_weights = doc_weights / doc_weights.sum()

    t1_end = seq_d // 3
    t2_end = 2 * seq_d // 3

    all_t1 = []
    all_t2 = []
    all_t3 = []
    all_total = []

    corpus_chunk_size = 200
    query_step = 4

    for chunk_start in range(0, num_docs, corpus_chunk_size):
        chunk_end = min(chunk_start + corpus_chunk_size, num_docs)
        d_chunk = d_embeds[chunk_start:chunk_end]
        chunk_t1, chunk_t2, chunk_t3, chunk_total = [], [], [], []

        for q_start in range(0, q_embeds.size(0), query_step):
            q_end = min(q_start + query_step, q_embeds.size(0))
            q_batch = q_embeds[q_start:q_end]

            scores = torch.einsum("ash,bth->abst", q_batch, d_chunk)
            topk_scores, topk_idxs = scores.topk(k=k, axis=-1)
            topk_weights = doc_weights[-k:].unsqueeze(0).unsqueeze(0).unsqueeze(0)
            weighted_topk = topk_scores * topk_weights

            t1_mask = (topk_idxs < t1_end).float()
            t2_mask = ((topk_idxs >= t1_end) & (topk_idxs < t2_end)).float()
            t3_mask = (topk_idxs >= t2_end).float()

            t1_c = (weighted_topk * t1_mask).sum(axis=-1).sum(axis=2).cpu()
            t2_c = (weighted_topk * t2_mask).sum(axis=-1).sum(axis=2).cpu()
            t3_c = (weighted_topk * t3_mask).sum(axis=-1).sum(axis=2).cpu()
            total_c = t1_c + t2_c + t3_c

            chunk_t1.append(t1_c)
            chunk_t2.append(t2_c)
            chunk_t3.append(t3_c)
            chunk_total.append(total_c)

        all_t1.append(torch.cat(chunk_t1, dim=0))
        all_t2.append(torch.cat(chunk_t2, dim=0))
        all_t3.append(torch.cat(chunk_t3, dim=0))
        all_total.append(torch.cat(chunk_total, dim=0))

    t1_n = torch.cat(all_t1, dim=1).numpy()
    t2_n = torch.cat(all_t2, dim=1).numpy()
    t3_n = torch.cat(all_t3, dim=1).numpy()
    total_n = torch.cat(all_total, dim=1).numpy()

    total_sum = t1_n + t2_n + t3_n + 1e-10
    t1_r = t1_n / total_sum
    t2_r = t2_n / total_sum
    t3_r = t3_n / total_sum

    print(f"\n  Tertile Analysis (k={k}, seq_d={seq_d}):")
    print(f"    Region 1 (0-{t1_end-1}): ratio={np.mean(t1_r):.2%}, avg_contrib={np.mean(t1_n):.4f}")
    print(f"    Region 2 ({t1_end}-{t2_end-1}): ratio={np.mean(t2_r):.2%}, avg_contrib={np.mean(t2_n):.4f}")
    print(f"    Region 3 ({t2_end}-{seq_d-1}): ratio={np.mean(t3_r):.2%}, avg_contrib={np.mean(t3_n):.4f}")

    ndcg_full = compute_ndcg(total_n, q_order_ids, d_order_ids, relevant_docs)
    ndcg_1 = compute_ndcg(t1_n, q_order_ids, d_order_ids, relevant_docs)
    ndcg_2 = compute_ndcg(t2_n, q_order_ids, d_order_ids, relevant_docs)
    ndcg_3 = compute_ndcg(t3_n, q_order_ids, d_order_ids, relevant_docs)

    print(f"    NDCG@10: full={ndcg_full:.4f}, R1_only={ndcg_1:.4f}, R2_only={ndcg_2:.4f}, R3_only={ndcg_3:.4f}")

    return {
        "t1_ratio": float(np.mean(t1_r)),
        "t2_ratio": float(np.mean(t2_r)),
        "t3_ratio": float(np.mean(t3_r)),
        "ndcg_full": ndcg_full,
        "ndcg_r1_only": ndcg_1,
        "ndcg_r2_only": ndcg_2,
        "ndcg_r3_only": ndcg_3,
    }


def main():
    accelerator = Accelerator()
    if accelerator.num_processes != 1:
        raise RuntimeError("Only single process supported")

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

    print(f"Query: {query_embeds.shape}, Doc: {doc_embeds.shape}")

    # Region analysis
    print("\n=== Region Contribution Analysis (First Half vs Second Half) ===")
    print(f"{'k':>4} {'NDCG@10':>10} {'NDCG(1st)':>10} {'NDCG(2nd)':>10} {'1st ratio':>10} {'2nd ratio':>10} {'1st cnt':>8} {'2nd cnt':>8}")
    print("-" * 80)

    k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20, 30, 50, 100]
    region_results = analyze_region_contributions(query_embeds, doc_embeds, relevant_docs, q_order_ids, d_order_ids, k_values)

    # Tertile analysis for key k
    print("\n" + "="*60)
    for k in [4, 8, 12, 16]:
        analyze_tertiles(query_embeds, doc_embeds, relevant_docs, q_order_ids, d_order_ids, k)

    # Save results
    output_path = "./k_sweep_results/region_contribution_analysis.json"
    with open(output_path, "w") as f:
        json.dump(region_results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
