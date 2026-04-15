#!/usr/bin/env python3
"""Query Token Contribution Analysis for ColQwen3 (late-interaction / MaxSim).

This script analyzes:
1. Token-level contributions in MaxSim computation
2. Bad case analysis (noise token detection)
3. Ablation experiments (using top-k% tokens to approximate full performance)
python eval_token_contribution.py \
    --model-path /home/moxu/MMRAG/otherExp/colpali/output/colqwen3_vl_embedding_2B_lr5e-6_dim768/checkpoint-462 \
    --task Vidore3PhysicsRetrieval.v2 \
    --num-queries 302 \
    --encode-batch-size 16 \
    --output-dir ./token_contribution_results
    --strategies simple
"""

import sys
sys.path.insert(0, "/home/moxu/MMRAG/otherExp/colpali/mteb")

import argparse
import json
import os
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

import mteb
from mteb.models.model_implementations.colqwen_models import ColQwen3VLEmbeddingWrapper
from mteb.models.model_meta import ModelMeta, ScoringFunction


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Query Token Contribution Analysis")
    p.add_argument(
        "--model-path",
        type=str,
        default="/home/moxu/MMRAG/otherExp/colpali/output/colqwen3_vl_2B_lr5e-6_dim2048_stage3/checkpoint-462",
        help="Path to model checkpoint",
    )
    p.add_argument(
        "--hub-model-id",
        type=str,
        default="Qwen/Qwen3-VL-Embedding-2B",
        help="Base model HuggingFace id",
    )
    p.add_argument(
        "--task",
        type=str,
        default="Vidore3PhysicsRetrieval.v2",
        help="MTEB task to analyze",
    )
    p.add_argument(
        "--ablation-ratios",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
        help="Token ratios to test in ablation",
    )
    p.add_argument("--max-num-visual-tokens", type=int, default=1280)
    p.add_argument("--encode-batch-size", type=int, default=16)
    p.add_argument(
        "--output-dir",
        type=str,
        default="./token_contribution_results",
        help="Output directory for results",
    )
    p.add_argument(
        "--num-queries",
        type=int,
        default=None,
        help="Limit number of queries to analyze (default: all)",
    )
    p.add_argument(
        "--language",
        type=str,
        default="english",
        help="Language config to use (default: english)",
    )
    p.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=["all"],
        choices=["all", "uniform", "random", "first", "last", "importance", "kmeans", "simple"],
        help="Which strategies to test: all, simple (uniform+random+first+last), or specific strategies",
    )
    return p.parse_args()


def get_task_data(task_name: str, language: str = "english"):
    """已修正的 Vidore 数据加载函数"""
    from mteb.get_tasks import get_tasks
    from datasets import load_dataset

    # 1. 语言代码映射：解决 eng-Latn -> english 的问题
    lang_map = {
        "eng-latn": "english",
        "english": "english",
        "fra-latn": "french",
        "french": "french",
    }
    
    # 统一转换为小写并匹配
    clean_lang = language.lower().replace("_", "-")
    lang_suffix = lang_map.get(clean_lang, "english") # 默认回退到 english

    # 2. 获取任务元数据
    tasks = mteb.get_tasks(tasks=[task_name])
    if not tasks:
        raise ValueError(f"Task {task_name} not found in MTEB")
    task = tasks[0]

    dataset_path = task.metadata.dataset["path"]
    revision = task.metadata.dataset["revision"]
    
    print(f"DEBUG: Loading from {dataset_path} with config prefix '{lang_suffix}'")

    # 3. 加载数据集的三部分
    # 注意：Vidore 的规范通常是 english-corpus, english-queries, english-qrels
    corpus_ds = load_dataset(
        dataset_path,
        name=f"{lang_suffix}-corpus",
        revision=revision,
        split="test",
        trust_remote_code=True, # Vidore 任务通常需要 True
    )

    queries_ds = load_dataset(
        dataset_path,
        name=f"{lang_suffix}-queries",
        revision=revision,
        split="test",
        trust_remote_code=True,
    )

    qrels_ds = load_dataset(
        dataset_path,
        name=f"{lang_suffix}-qrels",
        revision=revision,
        split="test",
        trust_remote_code=True,
    )

    # 转换格式供后续分析使用
    corpus = {row["id"]: row for row in corpus_ds}
    queries = {row["id"]: row for row in queries_ds}

    relevant_docs = defaultdict(dict)
    for row in qrels_ds:
        relevant_docs[row["query-id"]][row["corpus-id"]] = row["score"]

    data = {
        "corpus": corpus,
        "queries": queries,
        "relevant_docs": dict(relevant_docs),
    }

    return task, data

def encode_queries(model, queries_dict, batch_size=16):
    """Encode queries and return embeddings.

    Args:
        model: The ColQwen3VLEmbeddingWrapper model
        queries_dict: Dict of query_id -> {'id', 'text', 'image' (optional)}
        batch_size: Batch size for encoding

    Returns:
        Tuple of (embeddings_dict, embeddings_tensor, id_list)
    """
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

    embeddings = torch.nn.utils.rnn.pad_sequence(all_embeds, batch_first=True, padding_value=0)

    embeddings_dict = {}
    for i, qid in enumerate(query_ids):
        embeddings_dict[qid] = embeddings[i]

    return embeddings_dict, embeddings, query_ids


def encode_corpus(model, corpus_dict, batch_size=4):
    """Encode corpus documents and return embeddings.

    Args:
        model: The ColQwen3VLEmbeddingWrapper model
        corpus_dict: Dict of doc_id -> {'id', 'text', 'image'}
        batch_size: Batch size for encoding

    Returns:
        Tuple of (embeddings_dict, embeddings_tensor, id_list)
    """
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

    embeddings = torch.nn.utils.rnn.pad_sequence(all_embeds, batch_first=True, padding_value=0)

    embeddings_dict = {}
    for i, did in enumerate(doc_ids):
        embeddings_dict[did] = embeddings[i]

    return embeddings_dict, embeddings, doc_ids


def compute_token_contributions_from_tensors(model, query_embeds, doc_embeds, top_k=None):
    """Compute token contributions from embedding tensors.

    Args:
        model: The ColQwen3VLEmbeddingWrapper model
        query_embeds: Query embeddings tensor (num_queries, num_tokens, dim)
        doc_embeds: Document embeddings tensor (num_docs, num_tokens, dim)
        top_k: If specified, only use top-k tokens by contribution

    Returns:
        Tuple of (scores, contribs, max_idxs)
    """
    return model.similarity_with_attribution(query_embeds, doc_embeds, top_k=top_k)


def analyze_token_contributions(
    scores: torch.Tensor,
    token_contributions: torch.Tensor,
    query_ids: list[str],
    queries_dict: dict,
    threshold_percentile: float = 25,
) -> dict[str, Any]:
    """Analyze token contributions across all queries.

    Args:
        token_contributions: (num_queries, num_docs_best_match, num_query_tokens)
        query_ids: List of query IDs
        queries_dict: Dict of query data
        threshold_percentile: Percentile below which tokens are considered low-contribution

    Returns:
        Dictionary with analysis results
    """
    num_queries = token_contributions.shape[0]

    results = {
        "per_query": [],
        "summary": {},
    }

    total_contribs = []
    for i in range(num_queries):
        best_doc_idx = scores[i].argmax().item()
        contribs = token_contributions[i, best_doc_idx].numpy()
        query_len = len(contribs)

        if query_len == 0:
            continue

        qid = query_ids[i]
        query_text = queries_dict.get(qid, {}).get("text", "")

        contrib_sum = contribs.sum()
        normalized_contribs = contribs / contribs.sum() if contrib_sum > 0 else contribs

        threshold = np.percentile(normalized_contribs, threshold_percentile)
        low_contrib_indices = np.where(normalized_contribs < threshold)[0]

        total_contribs.extend(normalized_contribs.tolist())

        results["per_query"].append({
            "query_id": qid,
            "query_text": query_text[:100] if query_text else "",
            "num_tokens": query_len,
            "total_score": float(contrib_sum),
            "mean_contribution": float(normalized_contribs.mean()),
            "max_contribution": float(normalized_contribs.max()),
            "min_contribution": float(normalized_contribs.min()),
            "low_contrib_token_count": len(low_contrib_indices),
            "low_contrib_ratio": len(low_contrib_indices) / query_len,
            "contribution_std": float(normalized_contribs.std()),
        })

    total_contribs = np.array(total_contribs)
    results["summary"] = {
        "num_queries": num_queries,
        "avg_query_tokens": float(np.mean([r["num_tokens"] for r in results["per_query"]])),
        "avg_low_contrib_ratio": float(np.mean([r["low_contrib_ratio"] for r in results["per_query"]])),
        "global_low_contrib_ratio": float((total_contribs < np.percentile(total_contribs, threshold_percentile)).mean()),
        "global_contribution_mean": float(total_contribs.mean()),
        "global_contribution_std": float(total_contribs.std()),
    }

    return results


def analyze_bad_cases(
    scores: torch.Tensor,
    queries_dict: dict,
    corpus_dict: dict,
    relevant_docs: dict[str, dict[str, int]],
    token_contributions: torch.Tensor,
    query_ids: list[str],
    doc_ids: list[str],
    top_k: int = 5,
) -> dict[str, Any]:
    """Analyze bad cases to detect potential noise tokens.

    Args:
        scores: (num_queries, num_docs) similarity scores
        queries_dict: Dict of query data
        corpus_dict: Dict of corpus data
        relevant_docs: Dict of query_id -> {doc_id: score}
        token_contributions: (num_queries, num_docs, num_query_tokens)
        query_ids: List of query IDs
        doc_ids: List of doc IDs
        top_k: Number of top bad cases to analyze

    Returns:
        Dictionary with bad case analysis
    """
    results = {
        "low_contribution_cases": [],
        "ranking_failures": [],
    }

    for i in range(len(query_ids)):
        qid = query_ids[i]
        if qid not in relevant_docs:
            continue

        relevant = set(relevant_docs[qid].keys())
        if not relevant:
            continue

        query_score = scores[i]
        best_doc_idx = query_score.argmax().item()
        best_doc = doc_ids[best_doc_idx]
        best_doc_score = query_score[best_doc_idx].item()

        is_correct = best_doc in relevant

        if not is_correct:
            contribs = token_contributions[i, best_doc_idx].numpy()
            normalized = contribs / contribs.sum() if contribs.sum() > 0 else contribs

            top_contrib_idx = np.argsort(normalized)[::-1][:top_k]
            bottom_contrib_idx = np.argsort(normalized)[:top_k]

            query_text = queries_dict.get(qid, {}).get("text", "")

            results["ranking_failures"].append({
                "query_id": qid,
                "query_text": query_text[:200] if query_text else "",
                "num_query_tokens": len(contribs),
                "predicted_doc": best_doc,
                "predicted_score": float(best_doc_score),
                "relevant_docs": list(relevant),
                "top_contrib_indices": top_contrib_idx.tolist(),
                "bottom_contrib_indices": bottom_contrib_idx.tolist(),
                "top_contrib_values": [float(normalized[j]) for j in top_contrib_idx],
                "bottom_contrib_values": [float(normalized[j]) for j in bottom_contrib_idx],
            })

        # Mean across all docs for each query token, then find low-contribution ratio
        mean_contrib = token_contributions[i].mean(dim=0)  # (num_query_tokens,)
        threshold = np.percentile(mean_contrib.numpy(), 25)
        low_contrib_ratio = (mean_contrib < threshold).float().mean().item()

        if low_contrib_ratio > 0.5:
            contribs = token_contributions[i, best_doc_idx].numpy()
            normalized = contribs / contribs.sum() if contribs.sum() > 0 else contribs
            bottom_idx = np.argsort(normalized)[:3]
            query_text = queries_dict.get(qid, {}).get("text", "")

            results["low_contribution_cases"].append({
                "query_id": qid,
                "query_text": query_text[:200] if query_text else "",
                "num_query_tokens": len(contribs),
                "low_contrib_ratio": float(low_contrib_ratio),
                "potential_noise_indices": bottom_idx.tolist(),
                "potential_noise_values": [float(normalized[j]) for j in bottom_idx],
            })

    results["ranking_failures"] = sorted(
        results["ranking_failures"],
        key=lambda x: x["predicted_score"],
        reverse=True
    )[:top_k]

    results["low_contribution_cases"] = sorted(
        results["low_contribution_cases"],
        key=lambda x: x["low_contrib_ratio"],
        reverse=True
    )[:top_k]

    return results


def analyze_doc_token_coverage(
    token_contributions: torch.Tensor,
    max_doc_indices: torch.Tensor,
    scores: torch.Tensor,
    query_ids: list[str],
    doc_ids: list[str],
    corpus_dict: dict,
    relevant_docs: dict[str, dict[str, int]],
    top_k: int = 5,
) -> dict[str, Any]:
    """Analyze which doc tokens are matched by query tokens.

    Args:
        token_contributions: (num_queries, num_docs, num_query_tokens)
        max_doc_indices: (num_queries, num_docs, num_query_tokens) which doc token each query token matched to
        scores: (num_queries, num_docs) similarity scores
        query_ids: List of query IDs
        doc_ids: List of doc IDs
        corpus_dict: Dict of corpus data
        relevant_docs: Dict of query_id -> {doc_id: score}
        top_k: Number of top docs to analyze per query

    Returns:
        Dictionary with doc token coverage analysis
    """
    results = {
        "per_doc": [],
        "summary": {},
        "dark_tokens": [],  # Doc tokens never matched
        "hot_tokens": [],   # Most frequently matched doc tokens
    }

    num_queries, num_docs, num_query_tokens = token_contributions.shape

    # Aggregate doc token match counts across all queries
    doc_token_hit_counts = defaultdict(list)  # doc_id -> list of (token_idx, hit_count, total_contrib)

    for i in range(min(num_queries, 100)):  # Sample first 100 queries for analysis
        qid = query_ids[i]
        if qid not in relevant_docs:
            continue

        relevant = set(relevant_docs[qid].keys())

        # Get top-k docs for this query
        query_scores = scores[i]
        top_doc_indices = torch.argsort(query_scores, descending=True)[:top_k].tolist()

        for doc_idx in top_doc_indices:
            did = doc_ids[doc_idx]
            doc_token_hits = max_doc_indices[i, doc_idx].numpy()
            doc_token_contribs = token_contributions[i, doc_idx].numpy()

            # Count unique tokens matched
            unique_tokens = set(doc_token_hits.tolist())
            doc_len = len(doc_token_contribs)
            coverage_ratio = len(unique_tokens) / doc_len if doc_len > 0 else 0

            is_relevant = did in relevant

            if did not in doc_token_hit_counts:
                doc_token_hit_counts[did] = [0] * doc_len

            # Update hit counts for this doc
            for tok_idx in unique_tokens:
                if tok_idx < doc_len:
                    doc_token_hit_counts[did][tok_idx] += 1

            if len(results["per_doc"]) < 50:  # Keep only first 50 docs for detail
                results["per_doc"].append({
                    "query_id": qid,
                    "doc_id": did,
                    "is_relevant": is_relevant,
                    "doc_length": doc_len,
                    "unique_tokens_matched": len(unique_tokens),
                    "coverage_ratio": float(coverage_ratio),
                    "top_matched_token_indices": list(unique_tokens)[:20],
                })

    # Analyze dark tokens (never matched) and hot tokens (frequently matched)
    all_doc_hits = {}
    for did, hits in doc_token_hit_counts.items():
        if len(hits) > 0:
            all_doc_hits[did] = hits

    if all_doc_hits:
        # Flatten all hits
        flat_hits = []
        for did, hits in all_doc_hits.items():
            for tok_idx, count in enumerate(hits):
                flat_hits.append((did, tok_idx, count))

        if flat_hits:
            # Most matched tokens
            flat_hits.sort(key=lambda x: x[2], reverse=True)
            results["hot_tokens"] = [
                {"doc_id": did, "token_idx": tok_idx, "hit_count": count}
                for did, tok_idx, count in flat_hits[:20]
            ]

            # Least matched tokens (dark tokens) - filter first, then take top 20
            flat_hits.sort(key=lambda x: x[2])
            dark_only = [(did, tok_idx, count) for did, tok_idx, count in flat_hits if count == 0]
            results["dark_tokens"] = [
                {"doc_id": did, "token_idx": tok_idx, "hit_count": count}
                for did, tok_idx, count in dark_only[:20]
            ]

    # Summary statistics
    coverage_ratios = [p["coverage_ratio"] for p in results["per_doc"]]
    results["summary"] = {
        "num_docs_analyzed": len(results["per_doc"]),
        "avg_doc_coverage_ratio": float(np.mean(coverage_ratios)) if coverage_ratios else 0,
        "min_doc_coverage_ratio": float(np.min(coverage_ratios)) if coverage_ratios else 0,
        "max_doc_coverage_ratio": float(np.max(coverage_ratios)) if coverage_ratios else 0,
        "num_dark_tokens_found": len(results["dark_tokens"]),
        "num_hot_tokens_found": len(results["hot_tokens"]),
    }

    return results


def compute_retrieval_metrics(
    scores: torch.Tensor,
    query_ids: list[str],
    doc_ids: list[str],
    relevant_docs: dict[str, dict[str, int]],
    k_values: list[int] = [5, 10, 20, 100],
) -> dict[str, float]:
    """Compute retrieval metrics (MRR, Recall@K, NDCG@K).

    Args:
        scores: (num_queries, num_docs) similarity scores
        query_ids: List of query IDs
        doc_ids: List of doc IDs
        relevant_docs: Dict of query_id -> {doc_id: score}
        k_values: K values for Recall@K and NDCG@K

    Returns:
        Dictionary with metrics
    """
    metrics = {}

    mrr = 0.0
    recall_at_k = {k: 0.0 for k in k_values}
    ndcg_at_k = {k: 0.0 for k in k_values}
    num_evaluated = 0

    for i, qid in enumerate(query_ids):
        if qid not in relevant_docs:
            continue

        # relevant: doc_id -> relevance score (typically 1 for binary)
        relevant = relevant_docs[qid]  # keep as dict to get scores
        if not relevant:
            continue

        num_evaluated += 1

        ranking = torch.argsort(scores[i], descending=True)
        best_rank = float('inf')

        # Compute DCG and ideal DCG for NDCG
        dcg = {k: 0.0 for k in k_values}
        idcg = {k: 0.0 for k in k_values}

        # Sort relevance scores for IDCG (ideal ranking)
        sorted_rel_scores = sorted(relevant.values(), reverse=True)

        for rank, doc_idx in enumerate(ranking.tolist(), 1):
            did = doc_ids[doc_idx]

            # MRR and Recall@K
            if did in relevant:
                if best_rank == float('inf'):
                    best_rank = rank
                for k in k_values:
                    if rank <= k:
                        # DCG: rel_i / log2(rank+1), standard DCG formula
                        rel = relevant[did]
                        dcg[k] += rel / np.log2(rank + 1) if rank > 0 else rel

        # Compute IDCG for each k
        for k in k_values:
            idcg_rel = sorted_rel_scores[:k]
            for rank, rel in enumerate(idcg_rel, 1):
                idcg[k] += rel / np.log2(rank + 1) if rank > 0 else rel

        # Update metrics
        if best_rank < float('inf'):
            mrr += 1.0 / best_rank

            for k in k_values:
                top_k_docs = set(ranking[:k].tolist())
                top_k_dids = {doc_ids[idx] for idx in top_k_docs}
                if set(relevant.keys()) & top_k_dids:
                    recall_at_k[k] += 1

        # NDCG@K
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


def select_doc_tokens(
    doc_embeds: torch.Tensor,
    num_tokens: int,
    strategy: str = "uniform",
    token_importance: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select a subset of document tokens for reduced index storage.

    Args:
        doc_embeds: Document embeddings tensor (num_docs, num_tokens, dim)
        num_tokens: Number of tokens to keep per document
        strategy: Selection strategy:
            - "uniform": Uniformly sample tokens across the sequence
            - "random": Random selection
            - "first": Keep first N tokens (likely special tokens)
            - "last": Keep last N tokens (likely content tokens)
            - "importance": Select tokens by precomputed importance scores
            - "kmeans": Select cluster centers using K-means clustering
        token_importance: Optional tensor (num_tokens,) with importance scores for "importance" strategy

    Returns:
        Reduced document embeddings tensor (num_docs, num_tokens, dim)
    """
    num_docs, total_tokens, dim = doc_embeds.shape

    if num_tokens >= total_tokens:
        return doc_embeds

    if strategy == "uniform":
        # Uniformly sample tokens across the sequence
        indices = torch.linspace(0, total_tokens - 1, num_tokens).long()
        indices = indices.clamp(0, total_tokens - 1)
    elif strategy == "random":
        # Random selection
        indices = torch.randperm(total_tokens)[:num_tokens]
    elif strategy == "first":
        # Keep first N tokens (e.g., special tokens like [CLS])
        indices = torch.arange(min(num_tokens, total_tokens))
    elif strategy == "last":
        # Keep last N tokens (e.g., content tokens)
        indices = torch.arange(total_tokens - min(num_tokens, total_tokens), total_tokens)
    elif strategy == "importance":
        if token_importance is None:
            raise ValueError("token_importance tensor required for 'importance' strategy")
        # Select top-k tokens by importance score
        _, indices = torch.topk(token_importance, k=min(num_tokens, len(token_importance)))
    elif strategy == "kmeans":
        # K-means clustering: find cluster centers that represent all tokens
        from sklearn.cluster import MiniBatchKMeans

        # Flatten doc embeddings: (num_docs * num_tokens, dim)
        all_tokens = doc_embeds.reshape(-1, dim).numpy()

        # K-means clustering
        kmeans = MiniBatchKMeans(n_clusters=num_tokens, random_state=42, batch_size=1024)
        kmeans.fit(all_tokens)

        # For each doc, select the token closest to each cluster center
        # This gives us num_tokens representatives per doc
        indices = []
        for d_idx in range(num_docs):
            start = d_idx * total_tokens
            end = start + total_tokens
            doc_tokens = all_tokens[start:end]

            # Find closest token to each cluster center
            cluster_centers = kmeans.cluster_centers_
            doc_indices = []
            for c_idx in range(num_tokens):
                # Distance from each token to this cluster center
                distances = np.linalg.norm(doc_tokens - cluster_centers[c_idx], axis=1)
                closest = distances.argmin()
                doc_indices.append(closest)
            indices.append(sorted(doc_indices))  # Sort for consistency

        # indices is now (num_docs, num_tokens)
        indices = torch.tensor(indices, dtype=torch.long)
        return doc_embeds[torch.arange(num_docs).unsqueeze(1), indices]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return doc_embeds[:, indices, :]


def compute_similarity_with_reduced_docs(
    model,
    query_embeds: torch.Tensor,
    doc_embeds_reduced: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute similarity with reduced document embeddings.

    Args:
        model: The ColQwen3VLEmbeddingWrapper model
        query_embeds: Query embeddings tensor (num_queries, num_query_tokens, dim)
        doc_embeds_reduced: Reduced document embeddings (num_docs, num_doc_tokens_reduced, dim)

    Returns:
        Tuple of (scores, contribs, max_idxs)
    """
    return model.similarity_with_attribution(query_embeds, doc_embeds_reduced, top_k=None)


def compute_token_importance_from_training(
    model,
    query_embeds: torch.Tensor,
    doc_embeds: torch.Tensor,
) -> torch.Tensor:
    """Compute token importance based on average contribution across all query-doc pairs.

    Args:
        model: The model wrapper
        query_embeds: Query embeddings tensor (num_queries, num_query_tokens, dim)
        doc_embeds: Document embeddings tensor (num_docs, num_doc_tokens, dim)

    Returns:
        Importance scores tensor (num_doc_tokens,) - average contribution per doc token position
    """
    print("Computing token importance from query-doc interactions...")

    # Compute token contributions for a sample of query-doc pairs
    scores, token_contribs, max_idxs = model.similarity_with_attribution(
        query_embeds[:min(50, len(query_embeds))],  # Sample queries for speed
        doc_embeds[:min(100, len(doc_embeds))],     # Sample docs
        top_k=None
    )

    # token_contribs: (num_sampled_queries, num_sampled_docs, num_query_tokens)
    # max_idxs: (num_sampled_queries, num_sampled_docs, num_query_tokens)

    # For each doc token position, compute how often it's selected (average contribution)
    # Shape: (num_sampled_docs, num_doc_tokens)
    importance_by_position = token_contribs.mean(dim=(0, 2))  # Average over queries and query_tokens

    # Global average across all sampled docs
    global_importance = importance_by_position.mean(dim=0)  # (num_doc_tokens,)

    print(f"Computed importance scores for {len(global_importance)} token positions")
    return global_importance


def ablation_doc_token_count(
    model,
    query_embeds: torch.Tensor,
    doc_embeds: torch.Tensor,
    query_ids: list[str],
    doc_ids: list[str],
    relevant_docs: dict[str, dict[str, int]],
    num_tokens_list: list[int],
    strategy: str = "uniform",
    compute_importance: bool = False,
) -> dict[str, Any]:
    """Run doc-side ablation experiments to study index size vs performance trade-off.

    This explores how many doc tokens are needed to approximate full late-interaction
    performance, with the goal of reducing index storage size.

    Args:
        model: The model wrapper
        query_embeds: Query embeddings tensor (all query tokens)
        doc_embeds: Full document embeddings tensor (num_docs, num_doc_tokens, dim)
        query_ids: List of query IDs
        doc_ids: List of doc IDs
        relevant_docs: Dict of query_id -> {doc_id: score}
        num_tokens_list: List of num_tokens to test (e.g., [8, 16, 32, 64, 128])
        strategy: Selection strategy for doc tokens:
            - "uniform": Uniform sampling across sequence
            - "random": Random selection
            - "first": Keep first N tokens
            - "last": Keep last N tokens
            - "importance": Select by precomputed importance scores
            - "kmeans": K-means cluster centers
        compute_importance: If True, compute importance scores for "importance" strategy

    Returns:
        Dictionary with ablation results
    """
    # Compute full (baseline) metrics with all doc tokens
    print("Computing baseline with full doc tokens...")
    full_scores, _, _ = compute_similarity_with_reduced_docs(
        model, query_embeds, doc_embeds
    )
    full_metrics = compute_retrieval_metrics(full_scores, query_ids, doc_ids, relevant_docs)

    num_docs, full_doc_tokens, dim = doc_embeds.shape
    index_size_full = num_docs * full_doc_tokens * dim * 4  # float32 = 4 bytes

    # Precompute importance scores if using importance strategy
    token_importance = None
    if strategy == "importance" or compute_importance:
        token_importance = compute_token_importance_from_training(model, query_embeds, doc_embeds)

    results = {
        "strategy": strategy,
        "full_doc_tokens": full_doc_tokens,
        "full_index_size_mb": index_size_full / (1024 * 1024),
        "full_metrics": full_metrics,
        "ablations": [],
    }

    print(f"\nFull doc: {full_doc_tokens} tokens/doc, index size: {index_size_full / (1024*1024):.2f} MB")
    print(f"NDCG@10: {full_metrics['ndcg@10']:.4f}")
    print(f"\nRunning doc-side ablation experiments (strategy={strategy})...")

    for num_tokens in sorted(num_tokens_list):
        if num_tokens >= full_doc_tokens:
            continue

        print(f"\n  Testing num_tokens={num_tokens} (ratio={num_tokens/full_doc_tokens:.2%})...")

        # Reduce doc tokens
        doc_embeds_reduced = select_doc_tokens(
            doc_embeds,
            num_tokens=num_tokens,
            strategy=strategy,
            token_importance=token_importance,
        )

        # Compute index size for this configuration
        index_size = num_docs * num_tokens * dim * 4
        compression_ratio = index_size / index_size_full

        # Compute metrics with reduced docs
        scores, _, _ = compute_similarity_with_reduced_docs(
            model, query_embeds, doc_embeds_reduced
        )

        metrics = compute_retrieval_metrics(scores, query_ids, doc_ids, relevant_docs)

        # Compute delta from full
        delta = {}
        for key in metrics:
            if full_metrics[key] > 0:
                delta[key] = metrics[key] - full_metrics[key]

        result_entry = {
            "num_tokens": num_tokens,
            "ratio": num_tokens / full_doc_tokens,
            "index_size_mb": index_size / (1024 * 1024),
            "compression_ratio": compression_ratio,
            "metrics": metrics,
            "delta_from_full": delta,
        }
        results["ablations"].append(result_entry)

        print(f"    Index size: {index_size / (1024*1024):.2f} MB (compressed to {compression_ratio:.1%})")
        print(f"    NDCG@10: {metrics['ndcg@10']:.4f} (delta: {delta.get('ndcg@10', 0):+.4f})")

    return results


def ablation_all_simple_strategies(
    model,
    query_embeds: torch.Tensor,
    doc_embeds: torch.Tensor,
    query_ids: list[str],
    doc_ids: list[str],
    relevant_docs: dict[str, dict[str, int]],
    num_tokens_list: list[int],
) -> dict[str, Any]:
    """Run ablation for all simple strategies in ONE pass (uniform, random, first, last).

    This is more efficient than running each strategy separately because:
    - Doc embeddings are computed only once
    - Similarity is computed per (num_tokens, strategy) combination

    Args:
        model: The model wrapper
        query_embeds: Query embeddings tensor
        doc_embeds: Full document embeddings tensor
        query_ids: List of query IDs
        doc_ids: List of doc IDs
        relevant_docs: Dict of query_id -> {doc_id: score}
        num_tokens_list: List of num_tokens to test

    Returns:
        Dictionary with all strategy results
    """
    simple_strategies = ["uniform", "random", "first", "last"]

    # Compute full baseline
    print("Computing baseline with full doc tokens...")
    full_scores, _, _ = compute_similarity_with_reduced_docs(model, query_embeds, doc_embeds)
    full_metrics = compute_retrieval_metrics(full_scores, query_ids, doc_ids, relevant_docs)

    num_docs, full_doc_tokens, dim = doc_embeds.shape
    index_size_full = num_docs * full_doc_tokens * dim * 4 / (1024 * 1024)

    results = {
        "full_doc_tokens": full_doc_tokens,
        "full_index_size_mb": index_size_full,
        "full_metrics": full_metrics,
        "strategies": {s: {"ablations": []} for s in simple_strategies},
    }

    print(f"\nFull doc: {full_doc_tokens} tokens/doc, Index: {index_size_full:.2f} MB, NDCG@10: {full_metrics['ndcg@10']:.4f}")
    print(f"\nRunning ablation for all simple strategies: {simple_strategies}")

    for num_tokens in sorted(num_tokens_list):
        if num_tokens >= full_doc_tokens:
            continue

        ratio = num_tokens / full_doc_tokens
        index_size = num_docs * num_tokens * dim * 4 / (1024 * 1024)
        compression_ratio = num_tokens / full_doc_tokens

        print(f"\n  [{num_tokens} tokens, {ratio:.1%}]")

        for strat in simple_strategies:
            # Select tokens for this strategy
            doc_embeds_reduced = select_doc_tokens(
                doc_embeds, num_tokens=num_tokens, strategy=strat
            )

            # Compute similarity and metrics
            scores, _, _ = compute_similarity_with_reduced_docs(model, query_embeds, doc_embeds_reduced)
            metrics = compute_retrieval_metrics(scores, query_ids, doc_ids, relevant_docs)

            delta = {k: metrics[k] - full_metrics[k] for k in metrics if full_metrics[k] > 0}

            results["strategies"][strat]["ablations"].append({
                "num_tokens": num_tokens,
                "ratio": ratio,
                "index_size_mb": index_size,
                "compression_ratio": compression_ratio,
                "metrics": metrics,
                "delta_from_full": delta,
            })

            print(f"    {strat:>8}: NDCG@10={metrics['ndcg@10']:.4f} (delta={delta.get('ndcg@10', 0):+.4f})")

    return results


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from: {args.model_path}")
    model = ColQwen3VLEmbeddingWrapper(
        model_name="Qwen/Qwen3-VL-Embedding-2B-colpali",
        hub_model_id=args.hub_model_id,
        peft_adapter_path=args.model_path,
        max_num_visual_tokens=args.max_num_visual_tokens,
        device="cuda" if torch.cuda.is_available() else "cpu",
        similarity_use_max_sim=True,
        attn_implementation="sdpa",
        use_cache=False,
    )

    model.mteb_model_meta = ModelMeta(
        name="moxu/colqwen3_vl_embedding_2B_lr5e-6_dim768_stage0",
        revision="v1",
        release_date="2026-03-24",
        languages=["eng-Latn"],
        framework=["PyTorch", "ColPali"],
        similarity_fn_name=ScoringFunction.MAX_SIM,
        modalities=["text", "image"],
        model_type=["late-interaction"],
        loader=None,
        n_parameters=2_000_000_000,
        memory_usage_mb=8000.0,
        max_tokens=args.max_num_visual_tokens,
        embed_dim=2048,
        license="apache-2.0",
        open_weights=True,
        public_training_code=None,
        public_training_data=None,
        use_instructions=False,
        training_datasets=None,
    )

    print(f"Loading task: {args.task} (language: {args.language})")
    task, data = get_task_data(args.task, language=args.language)

    queries_dict = data["queries"]
    corpus_dict = data["corpus"]
    relevant_docs = data["relevant_docs"]

    query_ids = list(queries_dict.keys())
    doc_ids = list(corpus_dict.keys())

    if args.num_queries is not None:
        query_ids = query_ids[:args.num_queries]

    print(f"\nEncoding {len(query_ids)} queries...")
    _, query_embeds, _ = encode_queries(model, queries_dict, batch_size=args.encode_batch_size)
    if args.num_queries is not None:
        query_embeds = query_embeds[:args.num_queries]
    print(f"Query embeddings shape: {query_embeds.shape}")

    print(f"\nEncoding {len(doc_ids)} documents...")
    _, doc_embeds, _ = encode_corpus(model, corpus_dict, batch_size=args.encode_batch_size)
    print(f"Document embeddings shape: {doc_embeds.shape}")

    print("\nComputing token contributions...")
    scores, contribs, max_idxs = compute_token_contributions_from_tensors(
        model, query_embeds, doc_embeds, top_k=None
    )

    print("\n=== Token Contribution Analysis ===")
    contrib_analysis = analyze_token_contributions(
        scores,
        contribs,
        query_ids,
        queries_dict,
        threshold_percentile=25,
    )

    print(f"Average query tokens: {contrib_analysis['summary']['avg_query_tokens']:.1f}")
    print(f"Average low contribution ratio: {contrib_analysis['summary']['avg_low_contrib_ratio']:.2%}")

    with open(os.path.join(args.output_dir, "token_contributions.json"), "w") as f:
        json.dump(contrib_analysis, f, indent=2)
    print(f"\nToken contributions saved to {args.output_dir}/token_contributions.json")

    print("\n=== Bad Case Analysis ===")
    bad_case_analysis = analyze_bad_cases(
        scores,
        queries_dict,
        corpus_dict,
        relevant_docs,
        contribs,
        query_ids,
        doc_ids,
        top_k=5,
    )

    print(f"Ranking failures analyzed: {len(bad_case_analysis['ranking_failures'])}")
    print(f"Low contribution cases: {len(bad_case_analysis['low_contribution_cases'])}")

    with open(os.path.join(args.output_dir, "bad_case_analysis.json"), "w") as f:
        json.dump(bad_case_analysis, f, indent=2, default=str)
    print(f"Bad case analysis saved to {args.output_dir}/bad_case_analysis.json")

    print("\n=== Document Token Coverage Analysis ===")
    doc_token_analysis = analyze_doc_token_coverage(
        contribs,
        max_idxs,
        scores,
        query_ids,
        doc_ids,
        corpus_dict,
        relevant_docs,
        top_k=5,
    )

    print(f"Average doc token coverage ratio: {doc_token_analysis['summary']['avg_doc_coverage_ratio']:.2%}")
    print(f"Dark tokens (never matched): {doc_token_analysis['summary']['num_dark_tokens_found']}")
    print(f"Hot tokens (frequently matched): {doc_token_analysis['summary']['num_hot_tokens_found']}")

    with open(os.path.join(args.output_dir, "doc_token_coverage.json"), "w") as f:
        json.dump(doc_token_analysis, f, indent=2, default=str)
    print(f"Doc token coverage saved to {args.output_dir}/doc_token_coverage.json")

    print("\n=== Doc-Side Ablation Experiments ===")
    print("Goal: Find minimal doc tokens needed to approximate full late-interaction performance")
    print("This reduces index storage size while preserving query-side late interaction\n")

    # Define num_tokens to test
    doc_token_counts = [8, 16, 32, 64, 128, 256, 512]

    # Get model name for output directory naming
    model_name = model.mteb_model_meta.name if hasattr(model, 'mteb_model_meta') else "colqwen3_vl"
    model_name_safe = model_name.replace("/", "_").replace("-", "_")

    # Parse strategies to run
    strategies_requested = [s.lower() for s in args.strategies]
    if "all" in strategies_requested:
        run_simple = True
        run_importance = True
        run_kmeans = True
        specific_strategies = []
    elif "simple" in strategies_requested:
        run_simple = True
        run_importance = False
        run_kmeans = False
        specific_strategies = []
    else:
        run_simple = any(s in ["uniform", "random", "first", "last"] for s in strategies_requested)
        run_importance = "importance" in strategies_requested
        run_kmeans = "kmeans" in strategies_requested
        specific_strategies = [s for s in strategies_requested if s not in ["uniform", "random", "first", "last", "importance", "kmeans"]]

    print(f"Strategies requested: {args.strategies}")
    print(f"Will run: {'simple' if run_simple else ''} {'importance' if run_importance else ''} {'kmeans' if run_kmeans else ''}".strip())

    all_results = {}
    strategy_dirs = {}
    simple_results = None
    importance_results = None
    kmeans_results = None

    # Step 1: Test simple strategies (uniform, random, first, last)
    if run_simple:
        print("\n" + "="*70)
        print("STEP 1: Testing simple strategies (uniform, random, first, last)")
        print("="*70)
        print("Note: Doc embeddings computed once, strategies differ only in token selection\n")

        simple_results = ablation_all_simple_strategies(
            model, query_embeds, doc_embeds,
            query_ids, doc_ids, relevant_docs,
            num_tokens_list=doc_token_counts
        )
        all_results["simple"] = simple_results

        # Create directories and save simple strategies results
        for strat in ["uniform", "random", "first", "last"]:
            strat_dir = os.path.join(args.output_dir, f"{model_name_safe}_{strat}")
            os.makedirs(strat_dir, exist_ok=True)
            strategy_dirs[strat] = strat_dir

            strat_result = {
                "strategy": strat,
                "full_doc_tokens": simple_results["full_doc_tokens"],
                "full_index_size_mb": simple_results["full_index_size_mb"],
                "full_metrics": simple_results["full_metrics"],
                "ablations": simple_results["strategies"][strat]["ablations"]
            }
            with open(os.path.join(strat_dir, "results.json"), "w") as f:
                json.dump(strat_result, f, indent=2, default=str)
            print(f"  {strat}: saved to {strat_dir}/")

        with open(os.path.join(args.output_dir, f"{model_name_safe}_simple_strategies.json"), "w") as f:
            json.dump(simple_results, f, indent=2, default=str)
        print(f"\nSimple strategies combined results saved")

    # Step 2: Test importance strategy
    if run_importance:
        print("\n" + "="*70)
        print("STEP 2: Testing importance strategy")
        print("="*70)
        importance_results = ablation_doc_token_count(
            model, query_embeds, doc_embeds,
            query_ids, doc_ids, relevant_docs,
            num_tokens_list=doc_token_counts,
            strategy="importance",
            compute_importance=True,
        )
        all_results["importance"] = importance_results

        importance_dir = os.path.join(args.output_dir, f"{model_name_safe}_importance")
        os.makedirs(importance_dir, exist_ok=True)
        strategy_dirs["importance"] = importance_dir
        with open(os.path.join(importance_dir, "results.json"), "w") as f:
            json.dump(importance_results, f, indent=2, default=str)
        print(f"Importance results saved to {importance_dir}/")

    # Step 3: Test kmeans strategy
    if run_kmeans:
        print("\n" + "="*70)
        print("STEP 3: Testing kmeans strategy")
        print("="*70)
        kmeans_results = ablation_doc_token_count(
            model, query_embeds, doc_embeds,
            query_ids, doc_ids, relevant_docs,
            num_tokens_list=doc_token_counts,
            strategy="kmeans",
            compute_importance=False,
        )
        all_results["kmeans"] = kmeans_results

        kmeans_dir = os.path.join(args.output_dir, f"{model_name_safe}_kmeans")
        os.makedirs(kmeans_dir, exist_ok=True)
        strategy_dirs["kmeans"] = kmeans_dir
        with open(os.path.join(kmeans_dir, "results.json"), "w") as f:
            json.dump(kmeans_results, f, indent=2, default=str)
        print(f"Kmeans results saved to {kmeans_dir}/")

    # Save combined results
    if all_results:
        with open(os.path.join(args.output_dir, f"{model_name_safe}_all_results.json"), "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nAll results saved to {args.output_dir}/{model_name_safe}_all_results.json")

    # Compare all strategies summary
    print("\n" + "="*120)
    print("STRATEGY COMPARISON SUMMARY (NDCG@10)")
    print("="*120)

    if simple_results is None and importance_results is None and kmeans_results is None:
        print("No strategies run!")
        return

    # Get baseline info
    if simple_results:
        full_ndcg = simple_results['full_metrics']['ndcg@10']
        full_doc_tokens = simple_results['full_doc_tokens']
        index_size_full = simple_results['full_index_size_mb']
    elif importance_results:
        full_ndcg = importance_results['full_metrics']['ndcg@10']
        full_doc_tokens = importance_results['full_doc_tokens']
        index_size_full = importance_results['full_index_size_mb']
    elif kmeans_results:
        full_ndcg = kmeans_results['full_metrics']['ndcg@10']
        full_doc_tokens = kmeans_results['full_doc_tokens']
        index_size_full = kmeans_results['full_index_size_mb']

    # Build header
    header_strategies = []
    if run_simple:
        header_strategies.extend(["uniform", "random", "first", "last"])
    if run_importance:
        header_strategies.append("importance")
    if run_kmeans:
        header_strategies.append("kmeans")

    header = f"{'num_tokens':>10} | {'ratio':>8}"
    for s in header_strategies:
        header += f" | {s:>8}"
    print(f"\nBaseline: {full_doc_tokens} tokens/doc, Index: {index_size_full:.2f} MB, NDCG@10: {full_ndcg:.4f}")
    print("-"*len(header))
    print(header)
    print("-"*len(header))

    for i, num_tokens in enumerate(doc_token_counts):
        if num_tokens >= full_doc_tokens:
            continue
        ratio = num_tokens / full_doc_tokens

        row = f"{num_tokens:>10} | {ratio:>7.1%}"
        for s in header_strategies:
            if s in ["uniform", "random", "first", "last"] and simple_results:
                ndcg = simple_results['strategies'][s]['ablations'][i]['metrics']['ndcg@10']
            elif s == "importance" and importance_results:
                ndcg = importance_results['ablations'][i]['metrics']['ndcg@10']
            elif s == "kmeans" and kmeans_results:
                ndcg = kmeans_results['ablations'][i]['metrics']['ndcg@10']
            else:
                ndcg = float('nan')
            row += f" | {ndcg:>8.4f}"
        print(row)
    print("-"*len(header))

    # Print directory structure
    print("\n" + "="*80)
    print("OUTPUT DIRECTORY STRUCTURE")
    print("="*80)
    print(f"\n{args.output_dir}/")

    if run_simple:
        for s in ["uniform", "random", "first", "last"]:
            print(f"├── {model_name_safe}_{s}/")
            print(f"│   └── results.json")
        print(f"├── {model_name_safe}_simple_strategies.json")

    if run_importance:
        print(f"├── {model_name_safe}_importance/")
        print(f"│   └── results.json")

    if run_kmeans:
        print(f"├── {model_name_safe}_kmeans/")
        print(f"│   └── results.json")

    if all_results:
        print(f"└── {model_name_safe}_all_results.json")

    print(f"\nAll results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
