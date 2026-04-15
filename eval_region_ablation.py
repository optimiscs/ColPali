#!/usr/bin/env python3
"""直接测试只用前半/后半/各1/3区域的token做max_sim，验证NDCG"""
import sys
sys.path.insert(0, "/home/moxu/MMRAG/otherExp/colpali/mteb")

import numpy as np
import torch
from pathlib import Path
from accelerate import Accelerator
from collections import defaultdict
from mteb.get_tasks import get_tasks
from mteb.similarity_functions import _convert_to_tensor
from datasets import load_dataset


def max_sim_base(a, b):
    """Original max_sim."""
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
    """Original max_sim similarity."""
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
    """Compute max_sim using only a subset of doc tokens [start, end)."""
    num_docs = d_embeds.size(0)
    d_subset = d_embeds[:, token_range[0]:token_range[1], :]
    final_scores = None
    for chunk_start in range(0, num_docs, corpus_chunk_size):
        chunk_end = min(chunk_start + corpus_chunk_size, num_docs)
        d_chunk = d_subset[chunk_start:chunk_end]
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

    # Baseline (all tokens)
    print("\n=== Baseline (all tokens) ===")
    scores_full = compute_similarity_base(query_embeds, doc_embeds)
    ndcg_full = compute_ndcg(scores_full.cpu().numpy(), q_order_ids, d_order_ids, relevant_docs)
    print(f"NDCG@10: {ndcg_full:.4f}")

    # Test different token ranges
    print("\n=== Token Range Ablation ===")
    print(f"{'Range':>20} {'#Tokens':>8} {'NDCG@10':>10} {'vs Full':>10}")
    print("-" * 50)

    ranges = [
        ("first 8", 0, 8),
        ("first 16", 0, 16),
        ("first 32", 0, 32),
        ("first 64", 0, 64),
        ("first 128", 0, 128),
        ("first 256", 0, 256),
        ("first 366 (half)", 0, seq_d // 2),
        ("last 8", seq_d - 8, seq_d),
        ("last 16", seq_d - 16, seq_d),
        ("last 32", seq_d - 32, seq_d),
        ("last 64", seq_d - 64, seq_d),
        ("last 128", seq_d - 128, seq_d),
        ("last 256", seq_d - 256, seq_d),
        ("last 366 (half)", seq_d // 2, seq_d),
        ("middle 366", seq_d // 4, seq_d - seq_d // 4),
        ("first 1/3 (244)", 0, seq_d // 3),
        ("middle 1/3 (244)", seq_d // 3, 2 * seq_d // 3),
        ("last 1/3 (244)", 2 * seq_d // 3, seq_d),
    ]

    for name, start, end in ranges:
        n_tokens = end - start
        if n_tokens < 1:
            continue
        scores = compute_similarity_subset(query_embeds, doc_embeds, (start, end))
        ndcg = compute_ndcg(scores.cpu().numpy(), q_order_ids, d_order_ids, relevant_docs)
        delta = ndcg - ndcg_full
        print(f"{name:>20} {n_tokens:>8} {ndcg:>10.4f} {delta:>+10.4f}")


if __name__ == "__main__":
    main()
