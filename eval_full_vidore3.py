#!/usr/bin/env python3
"""在完整Vidore3上评估 top-k=4 + position weighted 方法 - 修复OOM"""
import sys
sys.path.insert(0, "/home/moxu/MMRAG/otherExp/colpali/mteb")

import json
import gc
import shutil
import numpy as np
import torch
from pathlib import Path
from accelerate import Accelerator
from collections import defaultdict
from mteb.get_tasks import get_tasks
from mteb.similarity_functions import _convert_to_tensor
from datasets import load_dataset
import torch.nn.utils.rnn as rnn_utils


def cleanup_memory():
    """Force release all possible memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # "Vidore3ComputerScienceRetrieval.v2",
    # "Vidore3EnergyRetrieval.v2",
    # "Vidore3FinanceEnRetrieval.v2",
    # "Vidore3FinanceFrRetrieval.v2",
    # "Vidore3HrRetrieval.v2",
    # "Vidore3IndustrialRetrieval.v2",
    # "Vidore3PharmaceuticalsRetrieval.v2",
    # "Vidore3PhysicsRetrieval.v2",
TASKS = [
    "Vidore3ComputerScienceRetrieval.v2",
    "Vidore3EnergyRetrieval.v2",
    "Vidore3FinanceEnRetrieval.v2",
    "Vidore3FinanceFrRetrieval.v2",
    "Vidore3HrRetrieval.v2",
    "Vidore3IndustrialRetrieval.v2",
    "Vidore3PharmaceuticalsRetrieval.v2",
    "Vidore3PhysicsRetrieval.v2",
]


def max_sim_base(a, b):
    a = _convert_to_tensor(a)
    b = _convert_to_tensor(b)
    if len(a.shape) == 2:
        a = a.reshape(1, *a.shape)
    if len(b.shape) == 2:
        b = b.reshape(1, *b.shape)
    scores = torch.einsum("ash,bth->abst", a, b)
    max_scores = scores.max(axis=-1).values
    del scores  # Free memory immediately
    return max_scores.sum(axis=-1)


def topk_position_weighted_max_sim(q_embeds, d_embeds, k=4, alpha=1.0, corpus_chunk_size=200, query_step=4):
    """Top-k + position weighted max_sim."""
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
            topk_scores, _ = scores.topk(k=k, axis=-1)
            del scores  # Free memory immediately

            topk_weights = doc_weights[-k:].unsqueeze(0).unsqueeze(0).unsqueeze(0)
            weighted_topk = topk_scores * topk_weights

            avg_scores = weighted_topk.sum(axis=-1) / k
            final_q = avg_scores.sum(axis=-1)
            del weighted_topk, avg_scores, topk_scores  # Free memory
            chunk_scores_list.append(final_q)

        chunk_scores = torch.cat(chunk_scores_list, dim=0)
        del chunk_scores_list
        if final_scores is None:
            final_scores = chunk_scores
        else:
            final_scores = torch.cat([final_scores, chunk_scores], dim=1)
            del chunk_scores
        gc.collect()

    return final_scores


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
        del chunk_scores_list
        if final_scores is None:
            final_scores = chunk_scores
        else:
            final_scores = torch.cat([final_scores, chunk_scores], dim=1)
            del chunk_scores
        gc.collect()
    return final_scores


def compute_ndcg(scores_matrix, q_order_ids, d_order_ids, relevant_docs, ndcg_k=10):
    """Compute NDCG incrementally without storing full matrix."""
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


def compute_mrr(scores_matrix, q_order_ids, d_order_ids, relevant_docs):
    """Compute MRR incrementally without storing full matrix."""
    mrr_sum = 0.0
    n_evaluated = 0
    for i, qid in enumerate(q_order_ids):
        if qid not in relevant_docs:
            continue
        relevant = relevant_docs[qid]
        if not relevant:
            continue
        n_evaluated += 1

        ranking = np.argsort(-scores_matrix[i])
        best_rank = float('inf')
        for rank, doc_idx in enumerate(ranking.tolist(), 1):
            did = d_order_ids[doc_idx]
            if did in relevant:
                best_rank = rank
                break

        if best_rank < float('inf'):
            mrr_sum += 1.0 / best_rank

    return mrr_sum / n_evaluated if n_evaluated > 0 else 0.0


def compute_metrics_incremental(scores, q_order_ids, d_order_ids, relevant_docs):
    """Compute NDCG@10 and MRR in a single pass."""
    ndcg_sum = 0.0
    mrr_sum = 0.0
    n_evaluated = 0

    for i, qid in enumerate(q_order_ids):
        if qid not in relevant_docs:
            continue
        relevant = relevant_docs[qid]
        if not relevant:
            continue
        n_evaluated += 1

        # Get ranking
        ranking = np.argsort(-scores[i])
        sorted_rel = sorted(relevant.values(), reverse=True)

        # MRR
        best_rank = float('inf')
        for rank, doc_idx in enumerate(ranking.tolist(), 1):
            did = d_order_ids[doc_idx]
            if did in relevant:
                best_rank = rank
                break
        if best_rank < float('inf'):
            mrr_sum += 1.0 / best_rank

        # NDCG@10
        dcg = 0.0
        for rank_idx, doc_idx in enumerate(ranking[:10], 1):
            did = d_order_ids[doc_idx]
            if did in relevant:
                rel = relevant[did]
                dcg += rel / np.log2(rank_idx + 1)

        idcg = 0.0
        for rank_idx, rel in enumerate(sorted_rel[:10], 1):
            idcg += rel / np.log2(rank_idx + 1)

        if idcg > 0:
            ndcg_sum += dcg / idcg

    return {
        "ndcg10": ndcg_sum / n_evaluated if n_evaluated > 0 else 0.0,
        "mrr": mrr_sum / n_evaluated if n_evaluated > 0 else 0.0,
    }


def quantize_int8(t):
    scale = t.abs().max() / 127.0
    data = (t / scale).round().clamp(-127, 127).to(torch.int8)
    return data, scale


def load_task_data(task_name):
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


def encode_queries_chunked(model, queries_dict, task_name, cache_dir, batch_size=16, flush_every=500):
    """Encode queries in chunks to avoid OOM, save to single pt file."""
    import shutil

    query_ids = list(queries_dict.keys())
    tmp_dir = cache_dir / ".tmp" / task_name / "queries"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    chunk_files = []
    chunk_embeds = []
    chunk_ids = []
    chunk_idx = 0

    with torch.no_grad():
        for i in range(0, len(query_ids), batch_size):
            batch_ids = query_ids[i:i+batch_size]
            batch_texts = [queries_dict[qid].get("text", "") or "" for qid in batch_ids]
            inputs = model.processor.process_queries(texts=batch_texts)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outs = model._encode_inputs(inputs)
            for j in range(outs.shape[0]):
                chunk_embeds.append(outs[j].cpu().float())
                chunk_ids.append(batch_ids[j])
            del inputs, outs
            cleanup_memory()

            if len(chunk_embeds) >= flush_every:
                chunk_file = tmp_dir / f"chunk_{chunk_idx}.pt"
                chunk_tensor = rnn_utils.pad_sequence(chunk_embeds, batch_first=True, padding_value=0)
                torch.save(chunk_tensor, chunk_file)
                chunk_files.append(chunk_file)
                chunk_embeds = []
                chunk_ids = []
                chunk_idx += 1
                cleanup_memory()

    # Save remaining
    if chunk_embeds:
        chunk_file = tmp_dir / f"chunk_{chunk_idx}.pt"
        chunk_tensor = rnn_utils.pad_sequence(chunk_embeds, batch_first=True, padding_value=0)
        torch.save(chunk_tensor, chunk_file)
        chunk_files.append(chunk_file)

    # Merge all chunks
    all_tensors = [torch.load(f, weights_only=False) for f in chunk_files]
    merged = torch.cat(all_tensors, dim=0)
    del all_tensors
    for f in chunk_files:
        f.unlink()
    shutil.rmtree(tmp_dir.parent / task_name, ignore_errors=True)
    # query_ids is already in correct order from list(queries_dict.keys())
    return merged, query_ids


def encode_corpus_chunked(model, corpus_dict, task_name, cache_dir, batch_size=4, flush_every=500):
    """Encode corpus in chunks to avoid OOM, save to single pt file."""
    import torchvision.transforms.functional as F
    from PIL import Image
    import shutil

    doc_ids = list(corpus_dict.keys())
    tmp_dir = cache_dir / ".tmp" / task_name / "docs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    chunk_files = []
    chunk_embeds = []
    chunk_ids = []
    chunk_idx = 0

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
            for j in range(outs.shape[0]):
                chunk_embeds.append(outs[j].cpu().float())
                chunk_ids.append(batch_ids[j])
            del inputs, outs, imgs
            cleanup_memory()

            if len(chunk_embeds) >= flush_every:
                chunk_file = tmp_dir / f"chunk_{chunk_idx}.pt"
                chunk_tensor = rnn_utils.pad_sequence(chunk_embeds, batch_first=True, padding_value=0)
                torch.save(chunk_tensor, chunk_file)
                chunk_files.append(chunk_file)
                chunk_embeds = []
                chunk_ids = []
                chunk_idx += 1
                cleanup_memory()

    # Save remaining
    if chunk_embeds:
        chunk_file = tmp_dir / f"chunk_{chunk_idx}.pt"
        chunk_tensor = rnn_utils.pad_sequence(chunk_embeds, batch_first=True, padding_value=0)
        torch.save(chunk_tensor, chunk_file)
        chunk_files.append(chunk_file)

    # Merge all chunks
    all_tensors = [torch.load(f, weights_only=False) for f in chunk_files]
    merged = torch.cat(all_tensors, dim=0)
    del all_tensors
    for f in chunk_files:
        f.unlink()
    shutil.rmtree(tmp_dir.parent / task_name, ignore_errors=True)
    return merged, doc_ids


def main():
    accelerator = Accelerator()
    if accelerator.num_processes != 1:
        raise RuntimeError("Only single process")

    cache_dir = Path("./embeddings_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load model for encoding
    from mteb.models.model_implementations.colqwen_models import ColQwen3VLEmbeddingWrapper

    model = ColQwen3VLEmbeddingWrapper(
        model_name="Qwen/Qwen3-VL-Embedding-2B-colpali",
        hub_model_id="Qwen/Qwen3-VL-Embedding-2B",
        peft_adapter_path="/home/moxu/MMRAG/otherExp/colpali/merged_qwen3_vl_stage1",
        max_num_visual_tokens=768,
        device=str(accelerator.device),
        similarity_use_max_sim=True,
        attn_implementation="sdpa",
        use_cache=False,
    )

    all_results = []

    for task_name in TASKS:
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"{'='*60}")
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        q_cache = cache_dir / f"{task_name}_queries.pt"
        d_cache = cache_dir / f"{task_name}_docs.pt"

        # Load or encode
        if q_cache.exists() and d_cache.exists():
            print("Loading cached embeddings...")
            q_loaded = torch.load(q_cache)
            d_loaded = torch.load(d_cache)
            query_embeds = q_loaded["data"].to(accelerator.device).float() * q_loaded["scale"].to(accelerator.device)
            doc_embeds = d_loaded["data"].to(accelerator.device).float() * d_loaded["scale"].to(accelerator.device)
            q_order_ids = q_loaded["ids"]
            d_order_ids = d_loaded["ids"]
            del q_loaded, d_loaded
            cleanup_memory()
        else:
            print("Encoding (chunked)...")
            queries_dict, corpus_dict, relevant_docs = load_task_data(task_name)
            query_embeds, q_order_ids = encode_queries_chunked(
                model, queries_dict, task_name, cache_dir, batch_size=16, flush_every=500
            )
            doc_embeds, d_order_ids = encode_corpus_chunked(
                model, corpus_dict, task_name, cache_dir, batch_size=4, flush_every=500
            )
            del queries_dict, corpus_dict
            cleanup_memory()

            q_data, q_scale = quantize_int8(query_embeds.cpu())
            d_data, d_scale = quantize_int8(doc_embeds.cpu())
            torch.save({"data": q_data, "scale": q_scale, "ids": q_order_ids}, q_cache)
            torch.save({"data": d_data, "scale": d_scale, "ids": d_order_ids}, d_cache)
            del q_data, d_data, q_scale, d_scale
            cleanup_memory()
            print(f"Saved to cache: {cache_dir}")

        _, _, relevant_docs = load_task_data(task_name)

        print(f"Query: {query_embeds.shape}, Doc: {doc_embeds.shape}")
        seq_d = doc_embeds.size(1)

        # Baseline
        print("Computing baseline...")
        scores_base = compute_similarity_base(query_embeds, doc_embeds)
        metrics_base = compute_metrics_incremental(
            scores_base.cpu().numpy(), q_order_ids, d_order_ids, relevant_docs
        )
        scores_base = None
        cleanup_memory()
        print(f"  Baseline: NDCG@10={metrics_base['ndcg10']:.4f}, MRR={metrics_base['mrr']:.4f}")

        # Top-k weighted
        print("Computing top-k=4 weighted...")
        scores_topk = topk_position_weighted_max_sim(query_embeds, doc_embeds, k=4, alpha=1.0)
        metrics_topk = compute_metrics_incremental(
            scores_topk.cpu().numpy(), q_order_ids, d_order_ids, relevant_docs
        )
        scores_topk = None
        cleanup_memory()
        print(f"  TopK-4:   NDCG@10={metrics_topk['ndcg10']:.4f}, MRR={metrics_topk['mrr']:.4f}")

        delta_ndcg = metrics_topk['ndcg10'] - metrics_base['ndcg10']
        delta_mrr = metrics_topk['mrr'] - metrics_base['mrr']

        print(f"  Delta:    NDCG@10={delta_ndcg:+.4f}, MRR={delta_mrr:+.4f}")

        # Free embeddings
        query_embeds = None
        doc_embeds = None
        relevant_docs = None
        scores_base = None
        scores_topk = None
        cleanup_memory()

        all_results.append({
            "task": task_name,
            "n_queries": len(q_order_ids),
            "n_docs": len(d_order_ids),
            "seq_d": seq_d,
            "baseline": {"ndcg10": metrics_base['ndcg10'], "mrr": metrics_base['mrr']},
            "topk4": {"ndcg10": metrics_topk['ndcg10'], "mrr": metrics_topk['mrr']},
            "delta": {"ndcg10": delta_ndcg, "mrr": delta_mrr},
        })

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Task':<40} {'Base NDCG':>12} {'TopK NDCG':>12} {'Delta':>10}")
    print("-" * 80)

    for r in all_results:
        task_short = r["task"].replace("Vidore3", "V3").replace(".v2", "")
        print(f"{task_short:<40} {r['baseline']['ndcg10']:>12.4f} {r['topk4']['ndcg10']:>12.4f} {r['delta']['ndcg10']:>+10.4f}")

    # Average
    avg_base_ndcg = np.mean([r["baseline"]["ndcg10"] for r in all_results])
    avg_topk_ndcg = np.mean([r["topk4"]["ndcg10"] for r in all_results])
    avg_delta_ndcg = np.mean([r["delta"]["ndcg10"] for r in all_results])

    avg_base_mrr = np.mean([r["baseline"]["mrr"] for r in all_results])
    avg_topk_mrr = np.mean([r["topk4"]["mrr"] for r in all_results])
    avg_delta_mrr = np.mean([r["delta"]["mrr"] for r in all_results])

    print("-" * 80)
    print(f"{'AVERAGE':<40} {avg_base_ndcg:>12.4f} {avg_topk_ndcg:>12.4f} {avg_delta_ndcg:>+10.4f}")
    print(f"\nMRR: Base={avg_base_mrr:.4f}, TopK={avg_topk_mrr:.4f}, Delta={avg_delta_mrr:+.4f}")

    # Save
    output_path = "./k_sweep_results/vidore3_full_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
