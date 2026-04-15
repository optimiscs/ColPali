#!/usr/bin/env python3
"""指数位置加权 MaxSim 评测 - 支持多 alpha 一次性测试

Phase 1: 编码 queries 和 docs，持久化为 int8 格式
Phase 2: 对每个 alpha 用 max_sim_exp 计算相似度并评测
"""
import sys
sys.path.insert(0, "/home/moxu/MMRAG/otherExp/colpali/mteb")

import argparse
import json
import os
import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
from pathlib import Path
from collections import defaultdict
from accelerate import Accelerator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mteb.models.model_implementations.colqwen_models import ColQwen3VLEmbeddingWrapper
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.similarity_functions import _convert_to_tensor
from mteb.get_tasks import get_tasks
from datasets import load_dataset


# ========== MaxSim with Exponential Position Weighting ==========

def max_sim_exp(a, b, alpha: float = 2.0, strategy: str = "weighted_avg"):
    """Compute max-sim variants with exponential doc-position weighting.

    Strategies:
      - "weighted_avg": weighted average of top-k doc tokens, doc weights = exp(alpha * pos / L)
      - "topk_max": keep only top-k doc tokens per query, then uniform max
      - "exp_max": max with exp-weighted doc position (alpha controls how much later tokens dominate)
    """
    a = _convert_to_tensor(a)
    b = _convert_to_tensor(b)

    if len(a.shape) == 2:
        a = a.reshape(1, *a.shape)
    if len(b.shape) == 2:
        b = b.reshape(1, *b.shape)

    # scores: (batch_a, batch_b, seq_a, seq_b)
    scores = torch.einsum("ash,bth->abst", a, b)
    seq_a = a.shape[1]
    seq_b = b.shape[1]

    if strategy == "exp_max":
        # Max over doc, but doc position is used to bias which token wins
        # Using softmax with temperature: higher alpha = more weight to later tokens
        doc_positions = torch.arange(seq_b, device=b.device, dtype=torch.float32)
        doc_weights = torch.exp(alpha * doc_positions / seq_b)
        doc_weights = doc_weights / doc_weights.sum()
        # Softmax-like reweighting: amplify later tokens exponentially
        biased_scores = scores * doc_weights.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        max_scores = biased_scores.max(axis=-1).values
        return max_scores.sum(axis=-1)

    elif strategy == "topk_max":
        # Keep only top-k doc tokens (by score), then do uniform max
        k = min(max(1, int(alpha)), seq_b)
        topk_scores, _ = scores.topk(k=k, axis=-1)
        max_scores = topk_scores.max(axis=-1).values
        return max_scores.sum(axis=-1)

    elif strategy == "weighted_avg":
        # Weighted average of top-k doc tokens
        k = min(max(1, int(alpha)), seq_b)
        doc_positions = torch.arange(seq_b, device=b.device, dtype=torch.float32)
        # Mild exponential doc weighting based on position
        doc_weights = torch.exp(doc_positions / seq_b)
        doc_weights = doc_weights / doc_weights.sum()

        # Top-k doc tokens per (query, doc, query_token)
        topk_scores, _ = scores.topk(k=k, axis=-1)  # (batch_a, batch_b, seq_a, k)
        # Apply doc weights to top-k
        topk_weights = doc_weights[-k:].unsqueeze(0).unsqueeze(0).unsqueeze(0)
        weighted_topk = topk_scores * topk_weights
        avg_scores = weighted_topk.sum(axis=-1) / k
        return avg_scores.sum(axis=-1)

    elif strategy == "softmax_avg":
        # Weighted average using softmax over doc positions
        doc_positions = torch.arange(seq_b, device=b.device, dtype=torch.float32)
        # alpha controls how peaked the weight distribution is
        # high alpha = more weight on later tokens
        doc_weights = torch.softmax(alpha * doc_positions / seq_b, dim=0)
        weighted_scores = scores * doc_weights.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        avg_scores = weighted_scores.sum(axis=-1)
        return avg_scores.sum(axis=-1)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def max_sim_exp_with_argmax(a, b, alpha: float = 2.0):
    """max_sim_exp with doc-weighted max and argmax tracking.

    Returns (scores, doc_argmax_positions).
    doc_argmax_positions: (batch_a, batch_b, seq_a) - which doc token was selected.
    """
    a = _convert_to_tensor(a)
    b = _convert_to_tensor(b)

    if len(a.shape) == 2:
        a = a.reshape(1, *a.shape)
    if len(b.shape) == 2:
        b = b.reshape(1, *b.shape)

    scores = torch.einsum("ash,bth->abst", a, b)

    # Doc position weighting: later doc tokens get higher weight
    seq_d = b.shape[1]
    doc_positions = torch.arange(seq_d, device=b.device, dtype=torch.float32)
    doc_weights = torch.exp(alpha * doc_positions / seq_d)
    doc_weights = doc_weights / doc_weights.sum()

    # Apply doc weights before max
    weighted_scores = scores * doc_weights.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    max_scores = weighted_scores.max(axis=-1).values
    argmax_idxs = scores.argmax(axis=-1)

    return max_scores.sum(axis=-1), argmax_idxs


def visualize_maxsim_patterns(q_embeds, d_embeds, q_ids, d_ids, relevant_docs,
                               alpha: float, output_dir: str):
    """可视化 max_sim 中 doc token 的匹配分布。

    关键问题：哪些 doc tokens 在 max_sim 中贡献最大？
    - 不是"被选中的频率"（argmax），而是"被选中时的 score 大小"
    """
    os.makedirs(output_dir, exist_ok=True)

    # 只取前20个query和前200个doc做可视化
    n_q_vis, n_d_vis = min(20, q_embeds.size(0)), min(200, d_embeds.size(0))
    q_batch = q_embeds[:n_q_vis]
    d_batch = d_embeds[:n_d_vis]

    # raw_scores: (n_q, n_d, seq_q, seq_d)
    raw_scores = torch.einsum("ash,bth->abst", q_batch, d_batch)

    seq_q = raw_scores.size(2)
    seq_d = raw_scores.size(3)

    # 对于每个 query token position，找出哪个 doc token position 的 score 最高
    # max over doc index (axis=1) -> (n_q, seq_q, seq_d)
    # 含义：对于每个 query 的每个 query token，每个 doc token 位置的 max similarity
    max_scores_per_doc = raw_scores.max(axis=1).values  # (n_q, seq_q, seq_d)

    # 对每个 query token position，统计各 doc token 位置的 score 均值
    mean_score_per_doc_pos = max_scores_per_doc.mean(axis=0).cpu().numpy()  # (seq_q, seq_d)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"MaxSim score analysis (alpha={alpha})\n"
                 f"queries={n_q_vis}, docs={n_d_vis}", fontsize=14)

    # Plot 1: heatmap - mean max-sim score for each (query_pos, doc_pos)
    ax = axes[0, 0]
    im = ax.imshow(mean_score_per_doc_pos.T, aspect="auto", cmap="plasma", origin="lower")
    ax.set_xlabel("Query Token Position")
    ax.set_ylabel("Doc Token Position")
    ax.set_title("Mean max-sim score: query_pos vs doc_pos\n(plasma = higher score)")
    plt.colorbar(im, ax=ax, label="Mean Score")

    # Plot 2: which doc positions have the HIGHEST SCORES (not just most selected)
    ax = axes[0, 1]
    # 对于每个 doc position，平均 score 是多少
    mean_per_doc = max_scores_per_doc.mean(axis=(0, 1)).cpu().numpy()  # (seq_d,)
    ax.plot(np.arange(len(mean_per_doc)), mean_per_doc, color="teal", linewidth=1.5)
    ax.axvline(x=np.argmax(mean_per_doc), color="red", linestyle="--", label=f"max at {np.argmax(mean_per_doc)}")
    ax.set_xlabel("Doc Token Position")
    ax.set_ylabel("Mean Max-Sim Score")
    ax.set_title("Which doc tokens have highest max-sim scores?\n"
                 f"Peak at doc pos {np.argmax(mean_per_doc)}, seq_d={seq_d}")
    ax.legend()

    # Plot 3: per query token position - which doc positions score highest
    ax = axes[1, 0]
    # 对每个 query token position，哪个 doc token 位置 score 最高
    best_doc_per_q = mean_score_per_doc_pos.argmax(axis=1)  # (seq_q,)
    ax.plot(np.arange(len(best_doc_per_q)), best_doc_per_q, color="steelblue", linewidth=2)
    ax.fill_between(np.arange(len(best_doc_per_q)), 0, best_doc_per_q, alpha=0.2, color="steelblue")
    ax.set_xlabel("Query Token Position")
    ax.set_ylabel("Best Doc Token Position")
    ax.set_title("Best matching doc token for each query token position")
    ax.set_ylim(0, seq_d)

    # Plot 4: query position weight comparison
    ax = axes[1, 1]
    unweighted = np.ones(seq_q) / seq_q
    positions = np.arange(seq_q)
    weights = np.exp(alpha * positions / seq_q)
    weights = weights / weights.sum()
    x = np.arange(seq_q)
    ax.bar(x - 0.2, unweighted * seq_q, width=0.4, alpha=0.6, label="uniform", color="gray")
    ax.bar(x + 0.2, weights * seq_q, width=0.4, alpha=0.6, label=f"exp α={alpha}", color="coral")
    ax.set_xlabel("Query Token Position")
    ax.set_ylabel("Relative Weight")
    ax.set_title("Query token position weights")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, f"maxsim_pattern_alpha_{alpha}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    # Plot 5: doc token cumulative score contribution
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    cumulative = np.cumsum(mean_per_doc)
    cumulative = cumulative / cumulative[-1]
    ax2.plot(np.arange(len(cumulative)), cumulative, color="purple", linewidth=2)
    # 找到前10%的doc tokens贡献了多少score
    n_10pct = seq_d // 10
    score_at_10pct = cumulative[n_10pct - 1] if n_10pct > 0 else 0
    n_50pct = seq_d // 2
    score_at_50pct = cumulative[n_50pct - 1] if n_50pct > 0 else 0
    ax2.axvline(x=n_10pct, color="red", linestyle="--", alpha=0.7)
    ax2.axvline(x=n_50pct, color="orange", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Doc Token Position (sorted by score)")
    ax2.set_ylabel("Cumulative Score Contribution")
    ax2.set_title(f"Doc token score concentration\n"
                  f"First 10% doc tokens = {score_at_10pct:.1%} of score\n"
                  f"First 50% doc tokens = {score_at_50pct:.1%} of score")
    ax2.legend()

    path2 = os.path.join(output_dir, f"doc_score_concentration_alpha_{alpha}.png")
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"  Saved: {path2}")

    # Plot 6: Compare score distribution for "first" vs "last" doc tokens
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    # 只取前20和后20个doc tokens，对比score分布
    n_edge = min(20, seq_d)
    first_scores = mean_per_doc[:n_edge]
    last_scores = mean_per_doc[-n_edge:]
    x_first = np.arange(n_edge)
    x_last = np.arange(seq_d - n_edge, seq_d)
    ax3.bar(x_first, first_scores, width=1.0, alpha=0.7, label=f"first {n_edge}", color="blue")
    ax3.bar(x_last, last_scores, width=1.0, alpha=0.7, label=f"last {n_edge}", color="red")
    ax3.set_xlabel("Doc Token Position")
    ax3.set_ylabel("Mean Max-Sim Score")
    ax3.set_title(f"First vs Last doc token scores\n"
                  f"First {n_edge} avg: {first_scores.mean():.4f}, Last {n_edge} avg: {last_scores.mean():.4f}")
    ax3.legend()

    path3 = os.path.join(output_dir, f"first_vs_last_scores_alpha_{alpha}.png")
    plt.savefig(path3, dpi=150)
    plt.close()
    print(f"  Saved: {path3}")


def max_sim_base(a, b):
    """Original max_sim (no weighting)."""
    a = _convert_to_tensor(a)
    b = _convert_to_tensor(b)
    if len(a.shape) == 2:
        a = a.reshape(1, *a.shape)
    if len(b.shape) == 2:
        b = b.reshape(1, *b.shape)
    scores = torch.einsum("ash,bth->abst", a, b)
    max_scores = scores.max(axis=-1).values
    return max_scores.sum(axis=-1)


def compute_similarity_baseline(q_embeds, d_embeds):
    """Original max_sim similarity (no weighting)."""
    corpus_chunk_size = 200
    query_step = 4
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


def compute_similarity_exp(q_embeds, d_embeds, alpha: float, strategy: str = "weighted_avg"):
    """Compute similarity with exponential weighted max_sim."""
    corpus_chunk_size = 200
    query_step = 4
    num_docs = d_embeds.size(0)
    final_scores = None

    for chunk_start in range(0, num_docs, corpus_chunk_size):
        chunk_end = min(chunk_start + corpus_chunk_size, num_docs)
        d_chunk = d_embeds[chunk_start:chunk_end]
        chunk_scores_list = []

        for q_start in range(0, q_embeds.size(0), query_step):
            q_end = min(q_start + query_step, q_embeds.size(0))
            q_batch = q_embeds[q_start:q_end]
            chunk_scores_list.append(max_sim_exp(q_batch, d_chunk, alpha=alpha, strategy=strategy))

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
    """Compute retrieval metrics (MRR, Recall@K, NDCG@K)."""
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


# ========== Data Loading ==========

def load_task_data(task_name: str):
    """Load queries, corpus, and relevant_docs using load_dataset."""
    tasks = get_tasks(tasks=[task_name])
    if not tasks:
        raise ValueError(f"Task {task_name} not found in MTEB")
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


# ========== Encoding ==========

def encode_queries(model, queries_dict, batch_size=16):
    """Encode queries. Returns (embeddings_dict, embeddings_tensor, id_list)."""
    query_ids = list(queries_dict.keys())
    all_embeds = []
    with torch.no_grad():
        for i in range(0, len(query_ids), batch_size):
            batch_ids = query_ids[i:i+batch_size]
            batch_texts = [queries_dict[qid].get("text", "") or "" for qid in batch_ids]
            inputs = model.processor.process_queries(texts=batch_texts)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outs = model._encode_inputs(inputs)
            all_embeds.extend(outs.cpu().to(torch.float32))
    embeddings = rnn_utils.pad_sequence(all_embeds, batch_first=True, padding_value=0)
    embeddings_dict = {}
    for i, qid in enumerate(query_ids):
        embeddings_dict[qid] = embeddings[i]
    return embeddings_dict, embeddings, query_ids


def encode_corpus(model, corpus_dict, batch_size=4):
    """Encode corpus documents. Returns (embeddings_dict, embeddings_tensor, id_list)."""
    import torchvision.transforms.functional as F
    from PIL import Image

    doc_ids = list(corpus_dict.keys())
    all_embeds = []
    with torch.no_grad():
        for i in range(0, len(doc_ids), batch_size):
            batch_ids = doc_ids[i:i+batch_size]
            batch_docs = [corpus_dict[did] for did in batch_ids]
            imgs = []
            for d in batch_docs:
                if "image" in d and d["image"]:
                    img = d["image"]
                    if not isinstance(img, Image.Image):
                        img = F.to_pil_image(img)
                    imgs.append(img.convert("RGB"))
                else:
                    imgs.append(None)
            inputs = model.processor.process_images(imgs)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outs = model._encode_inputs(inputs)
            all_embeds.extend(outs.cpu().to(torch.float32))
    embeddings = rnn_utils.pad_sequence(all_embeds, batch_first=True, padding_value=0)
    embeddings_dict = {}
    for i, did in enumerate(doc_ids):
        embeddings_dict[did] = embeddings[i]
    return embeddings_dict, embeddings, doc_ids


# ========== Int8 Persistence ==========

def quantize_int8(t):
    """Per-tensor int8 quantization with scale factor."""
    scale = t.abs().max() / 127.0
    data = (t / scale).round().clamp(-127, 127).to(torch.int8)
    return data, scale


# ========== Args ==========

def parse_args():
    p = argparse.ArgumentParser(description="MTEB 指数位置加权 MaxSim 评测")
    p.add_argument("--model-path", type=str,
                   default="/home/moxu/MMRAG/otherExp/colpali/merged_qwen3_vl_stage1")
    p.add_argument("--hub-model-id", type=str, default="Qwen/Qwen3-VL-Embedding-2B")
    p.add_argument("--max-num-visual-tokens", type=int, default=768)
    p.add_argument("--encode-batch-size", type=int, default=16)
    p.add_argument("--alpha", type=str, default="4,8,16,32,64,128",
                   help="逗号分隔的 alpha 列表")
    p.add_argument("--embeddings-cache-dir", type=str, default="./embeddings_cache")
    p.add_argument("--task", type=str, default="Vidore3PhysicsRetrieval.v2")
    p.add_argument("--num-queries", type=int, default=None)
    p.add_argument("--output-dir", type=str, default="./weighted_maxsim_results")
    return p.parse_args()


# ========== Main ==========

def main():
    args = parse_args()
    accelerator = Accelerator()

    if accelerator.num_processes != 1:
        raise RuntimeError(f"仅支持单进程，当前 num_processes={accelerator.num_processes}")

    alphas = [float(a.strip()) for a in args.alpha.split(",")]
    cache_dir = Path(args.embeddings_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Accelerate: {accelerator.device}, alphas: {alphas}")
    print(f"Cache dir: {cache_dir}")

    # ========== Phase 1: 编码 ==========
    print(f"\n{'='*60}")
    print("Phase 1: Encoding (int8 persist)")
    print(f"{'='*60}")

    model = ColQwen3VLEmbeddingWrapper(
        model_name="Qwen/Qwen3-VL-Embedding-2B-colpali",
        hub_model_id=args.hub_model_id,
        peft_adapter_path=args.model_path,
        max_num_visual_tokens=args.max_num_visual_tokens,
        device=str(accelerator.device),
        similarity_use_max_sim=True,
        attn_implementation="sdpa",
        use_cache=False,
    )

    task_name = args.task
    print(f"\nLoading task: {task_name}")
    queries_dict, corpus_dict, relevant_docs = load_task_data(task_name)

    query_ids = list(queries_dict.keys())
    doc_ids = list(corpus_dict.keys())
    if args.num_queries is not None:
        query_ids = query_ids[:args.num_queries]

    q_cache = cache_dir / f"{task_name}_queries.pt"
    d_cache = cache_dir / f"{task_name}_docs.pt"

    if q_cache.exists() and d_cache.exists():
        print("Loading cached embeddings...")
        q_loaded = torch.load(q_cache)
        d_loaded = torch.load(d_cache)
        query_embeds = q_loaded["data"].to(accelerator.device).float() * q_loaded["scale"].to(accelerator.device)
        doc_embeds = d_loaded["data"].to(accelerator.device).float() * d_loaded["scale"].to(accelerator.device)
        q_order_ids = q_loaded["ids"]
        d_order_ids = d_loaded["ids"]
    else:
        print(f"Encoding {len(query_ids)} queries...")
        _, query_embeds, q_order_ids = encode_queries(model, queries_dict, batch_size=args.encode_batch_size)
        if args.num_queries is not None:
            query_embeds = query_embeds[:args.num_queries]
            q_order_ids = q_order_ids[:args.num_queries]

        print(f"Encoding {len(doc_ids)} documents...")
        _, doc_embeds, d_order_ids = encode_corpus(model, corpus_dict, batch_size=4)

        print("Quantizing and saving...")
        q_data, q_scale = quantize_int8(query_embeds.cpu())
        d_data, d_scale = quantize_int8(doc_embeds.cpu())
        torch.save({"data": q_data, "scale": q_scale, "ids": q_order_ids}, q_cache)
        torch.save({"data": d_data, "scale": d_scale, "ids": d_order_ids}, d_cache)
        print(f"Saved to {cache_dir}")

    print(f"Query embeds: {query_embeds.shape}, Doc embeds: {doc_embeds.shape}")

    # ========== Phase 2: 多 alpha 评测 ==========
    print(f"\n{'='*60}")
    print("Phase 2: Computing for each alpha")
    print(f"{'='*60}")

    # Baseline: original max_sim (no weighting)
    print(f"\n--- baseline (original max_sim) ---")
    scores_base = compute_similarity_baseline(query_embeds, doc_embeds)
    metrics_base = compute_retrieval_metrics(
        scores_base, q_order_ids, d_order_ids, relevant_docs, k_values=[1, 5, 10]
    )
    print(f"  MRR:      {metrics_base['mrr']:.4f}")
    print(f"  Recall@5: {metrics_base['recall@5']:.4f}")
    print(f"  Recall@10:{metrics_base['recall@10']:.4f}")
    print(f"  NDCG@5:   {metrics_base['ndcg@5']:.4f}")
    print(f"  NDCG@10:  {metrics_base['ndcg@10']:.4f}")

    # 保存 baseline 结果
    result_base = {
        "task": task_name,
        "alpha": "baseline",
        "metrics": {k: float(v) for k, v in metrics_base.items()},
    }
    out_path_base = os.path.join(args.output_dir, "alpha_baseline.json")
    with open(out_path_base, "w") as f:
        json.dump(result_base, f, indent=2)
    print(f"  Saved: {out_path_base}")

    # Baseline 可视化 (alpha=0)
    visualize_maxsim_patterns(
        query_embeds, doc_embeds, q_order_ids, d_order_ids,
        relevant_docs, alpha=0, output_dir=args.output_dir
    )

    for alpha in alphas:
        # Test multiple strategies
        for strategy in ["exp_max", "topk_max", "weighted_avg", "softmax_avg"]:
            label = f"alpha={alpha}_strategy={strategy}"
            print(f"\n--- {label} ---")
            scores = compute_similarity_exp(
                query_embeds, doc_embeds, alpha=alpha, strategy=strategy
            )

            metrics = compute_retrieval_metrics(
                scores, q_order_ids, d_order_ids, relevant_docs, k_values=[1, 5, 10]
            )

            print(f"  MRR:      {metrics['mrr']:.4f}")
            print(f"  Recall@5: {metrics['recall@5']:.4f}")
            print(f"  Recall@10:{metrics['recall@10']:.4f}")
            print(f"  NDCG@5:   {metrics['ndcg@5']:.4f}")
            print(f"  NDCG@10:  {metrics['ndcg@10']:.4f}")

            result = {
                "task": task_name,
                "alpha": alpha,
                "strategy": strategy,
                "metrics": {k: float(v) for k, v in metrics.items()},
            }
            safe_label = f"alpha_{alpha}_strategy_{strategy}"
            out_path = os.path.join(args.output_dir, f"{safe_label}.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved: {out_path}")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
