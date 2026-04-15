"""
MTEB 检索评测：多 GPU 时对各进程上的 query / corpus **编码分片**，向量在 CPU 合并后做 **分块 MaxSim**（避免单卡同时驻留全量 Q_emb+D_emb）。

单进程：行为与 eval.py 一致，直接调用 mteb.evaluate。

多进程：accelerate launch 启动，例如:
  accelerate launch --num_processes 2 eval_retrieval_accelerate.py --model-path ...
"""
from __future__ import annotations

import argparse
import heapq
import sys
import time
from pathlib import Path

import torch
from accelerate import Accelerator

_REPO = Path(__file__).resolve().parent
if str(_REPO / "mteb") not in sys.path:
    sys.path.insert(0, str(_REPO / "mteb"))

import mteb  # noqa: E402
from mteb.abstasks.retrieval import (  # noqa: E402
    AbsTaskRetrieval,
    _filter_queries_without_positives,
)
from mteb._create_dataloaders import create_dataloader  # noqa: E402
from mteb._evaluators import RetrievalEvaluator  # noqa: E402
from mteb._evaluators.retrieval_metrics import make_score_dict  # noqa: E402
from mteb.models.model_implementations.colqwen_models import (  # noqa: E402
    ColQwen3VLEmbeddingWrapper,
)
from mteb.models.model_meta import ModelMeta, ScoringFunction  # noqa: E402
from mteb.results.task_result import TaskResult  # noqa: E402
from mteb.similarity_functions import max_sim  # noqa: E402
from mteb.types import PromptType  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MTEB 检索评测；多卡用 accelerate launch --num_processes N；单进程同 eval.py。",
    )
    p.add_argument(
        "--model-path",
        type=str,
        default="output/colqwen3_vl_2B_lr5e-6_dim2048_stage3/checkpoint-462",
    )
    p.add_argument("--hub-model-id", type=str, default="Qwen/Qwen3-VL-Embedding-2B")
    p.add_argument("--max-num-visual-tokens", type=int, default=1280)
    p.add_argument("--encode-batch-size", type=int, default=16)
    p.add_argument(
        "--shard-cache-dir",
        type=str,
        default="mteb_embedding_shards",
        help="多进程时各 rank 写入 q/d 分片，主进程合并；可复跑时删此目录重算",
    )
    p.add_argument(
        "--corpus-chunk-sim",
        type=int,
        default=200,
        help="MaxSim 时 corpus 方向分块大小（与 wrapper 默认一致）",
    )
    p.add_argument(
        "--query-chunk-sim",
        type=int,
        default=4,
        help="MaxSim 时 query 方向分块大小",
    )
    p.add_argument(
        "--sim-device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="分块 MaxSim 计算设备；cuda 时在合并后释放模型再算",
    )
    return p.parse_args()


def _contiguous_shard(
    dataset, rank: int, world_size: int
):
    """返回 [start, end) 与对应子集；world_size 路均匀切分。"""
    n = len(dataset)
    if n == 0:
        return 0, 0, None
    chunk = (n + world_size - 1) // world_size
    start = rank * chunk
    end = min(start + chunk, n)
    if start >= end:
        return start, end, None
    return start, end, dataset.select(range(start, end))


def _merge_shard_tensors(
    cache_dir: Path, prefix: str, world_size: int, feat_dim_hint: int | None = None
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    meta: list[tuple[int, int]] = []
    for r in range(world_size):
        path = cache_dir / f"{prefix}_rank{r}.pt"
        if not path.exists():
            continue
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(path, map_location="cpu")
        start, end = int(blob["start"]), int(blob["end"])
        emb = blob["embeddings"]
        if isinstance(emb, torch.Tensor) and emb.shape[0] > 0:
            parts.append(emb)
            meta.append((start, end))
    if not parts:
        if feat_dim_hint is None:
            raise RuntimeError(f"无有效分片: {cache_dir} / {prefix}")
        return torch.empty(0, 0, 0)
    meta.sort(key=lambda x: x[0])
    # 校验连续性
    for i in range(len(meta) - 1):
        if meta[i][1] != meta[i + 1][0]:
            raise RuntimeError(f"分片区间不连续: {meta}")
    return torch.cat(parts, dim=0)


def _maxsim_topk_heaps(
    query_emb: torch.Tensor,
    doc_emb: torch.Tensor,
    corpus_ids: list[str],
    query_idx_to_id: dict[int, str],
    top_k: int,
    corpus_chunk: int,
    query_chunk: int,
    sim_device: torch.device,
) -> dict[str, list[tuple[float, str]]]:
    """与 SearchEncoderWrapper._full_corpus_search 等价：doc 分块，每块对 query 子批算 max_sim，维护每 query 的 top_k 堆。"""
    nq, nd = query_emb.size(0), doc_emb.size(0)
    result_heaps: dict[str, list[tuple[float, str]]] = {
        qid: [] for qid in query_idx_to_id.values()
    }

    for c_start in range(0, nd, corpus_chunk):
        c_end = min(c_start + corpus_chunk, nd)
        d_chunk = doc_emb[c_start:c_end].to(sim_device, dtype=torch.float32)
        sub_corpus_ids = corpus_ids[c_start:c_end]

        chunk_scores_list: list[torch.Tensor] = []
        for q_start in range(0, nq, query_chunk):
            q_end = min(q_start + query_chunk, nq)
            q_batch = query_emb[q_start:q_end].to(sim_device, dtype=torch.float32)
            chunk_scores_list.append(max_sim(q_batch, d_chunk))
        scores = torch.cat(chunk_scores_list, dim=0)

        k_take = min(top_k + 1, scores.size(1))
        vals, idxs = torch.topk(scores, k_take, dim=1, largest=True)
        cos_scores_top_k_idx = idxs.cpu().tolist()
        cos_scores_top_k_values = vals.cpu().tolist()

        for query_itr in range(nq):
            query_id = query_idx_to_id[query_itr]
            for sub_idx, score in zip(
                cos_scores_top_k_idx[query_itr],
                cos_scores_top_k_values[query_itr],
            ):
                corpus_id = sub_corpus_ids[sub_idx]
                if len(result_heaps[query_id]) < top_k:
                    heapq.heappush(result_heaps[query_id], (score, corpus_id))
                else:
                    heapq.heappushpop(result_heaps[query_id], (score, corpus_id))

    return result_heaps


def _heaps_to_results(
    result_heaps: dict[str, list[tuple[float, str]]],
    query_idx_to_id: dict[int, str],
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {qid: {} for qid in query_idx_to_id.values()}
    for qid in result_heaps:
        for score, corpus_id in result_heaps[qid]:
            results[qid][corpus_id] = float(score)
    return results


def _build_model(args, device: str) -> ColQwen3VLEmbeddingWrapper:
    return ColQwen3VLEmbeddingWrapper(
        model_name="Qwen/Qwen3-VL-Embedding-2B-colpali",
        hub_model_id=args.hub_model_id,
        peft_adapter_path=args.model_path,
        max_num_visual_tokens=args.max_num_visual_tokens,
        device=device,
        similarity_use_max_sim=True,
        attn_implementation="sdpa",
        use_cache=False,
    )


def _evaluate_task_sharded(
    accelerator: Accelerator,
    task: AbsTaskRetrieval,
    encode_kwargs: dict,
    args: argparse.Namespace,
    split: str,
    hf_subset: str,
) -> dict:
    model_wrapper = _build_model(args, str(accelerator.device))

    data_split = task.dataset[hf_subset][split]
    data_split["relevant_docs"], data_split["queries"] = _filter_queries_without_positives(
        data_split["relevant_docs"],
        data_split["queries"],
    )
    queries = data_split["queries"]
    corpus = data_split["corpus"]
    relevant_docs = data_split["relevant_docs"]

    rank = accelerator.process_index
    world = accelerator.num_processes
    cache_root = Path(args.shard_cache_dir)
    safe_name = f"{task.metadata.name}_{hf_subset}_{split}".replace("/", "_")
    cache_dir = cache_root / safe_name
    if accelerator.is_main_process:
        cache_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    # ---------- encode queries (sharded) ----------
    qs, qe, q_sub = _contiguous_shard(queries, rank, world)
    if q_sub is not None:
        q_loader = create_dataloader(
            q_sub,
            task.metadata,
            prompt_type=PromptType.query,
            **encode_kwargs,
        )
        q_emb = model_wrapper.encode(
            q_loader,
            task_metadata=task.metadata,
            hf_split=split,
            hf_subset=hf_subset,
            prompt_type=PromptType.query,
            **encode_kwargs,
        )
        if isinstance(q_emb, torch.Tensor):
            q_cpu = q_emb.detach().cpu().float()
        else:
            q_cpu = torch.as_tensor(q_emb).cpu().float()
        torch.save(
            {"embeddings": q_cpu, "start": qs, "end": qe},
            cache_dir / f"q_rank{rank}.pt",
        )
        del q_emb, q_cpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    accelerator.wait_for_everyone()

    # ---------- encode corpus (sharded) ----------
    cs, ce, c_sub = _contiguous_shard(corpus, rank, world)
    if c_sub is not None:
        c_loader = create_dataloader(
            c_sub,
            task.metadata,
            prompt_type=PromptType.document,
            **encode_kwargs,
        )
        d_emb = model_wrapper.encode(
            c_loader,
            task_metadata=task.metadata,
            hf_split=split,
            hf_subset=hf_subset,
            prompt_type=PromptType.document,
            **encode_kwargs,
        )
        if isinstance(d_emb, torch.Tensor):
            d_cpu = d_emb.detach().cpu().float()
        else:
            d_cpu = torch.as_tensor(d_emb).cpu().float()
        torch.save(
            {"embeddings": d_cpu, "start": cs, "end": ce},
            cache_dir / f"d_rank{rank}.pt",
        )
        del d_emb, d_cpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    accelerator.wait_for_everyone()

    del model_wrapper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    accelerator.wait_for_everyone()

    score_dict: dict = {}
    if accelerator.is_main_process:
        Q = _merge_shard_tensors(cache_dir, "q", world)
        D = _merge_shard_tensors(cache_dir, "d", world)
        if Q.size(0) != len(queries) or D.size(0) != len(corpus):
            raise RuntimeError(
                f"合并后形状与数据不一致: Q={Q.shape} vs n_queries={len(queries)}, "
                f"D={D.shape} vs n_corpus={len(corpus)}"
            )

        corpus_ids = [str(x) for x in corpus["id"]]
        query_idx_to_id = {i: str(row["id"]) for i, row in enumerate(queries)}

        top_k = task._top_k
        sim_device = torch.device(
            "cuda:0" if args.sim_device == "cuda" and torch.cuda.is_available() else "cpu"
        )

        t0 = time.perf_counter()
        heaps = _maxsim_topk_heaps(
            Q,
            D,
            corpus_ids,
            query_idx_to_id,
            top_k,
            args.corpus_chunk_sim,
            args.query_chunk_sim,
            sim_device,
        )
        retrieval_out = _heaps_to_results(heaps, query_idx_to_id)

        retriever = RetrievalEvaluator(
            corpus=corpus,
            queries=queries,
            task_metadata=task.metadata,
            hf_split=split,
            hf_subset=hf_subset,
            top_k=top_k,
            top_ranked=data_split["top_ranked"]
            if "top_ranked" in data_split and data_split["top_ranked"] is not None
            else None,
        )
        (
            all_scores,
            ndcg,
            _map,
            recall,
            precision,
            naucs,
            mrr,
            naucs_mrr,
            hit_rate,
        ) = retriever.evaluate(
            relevant_docs,
            retrieval_out,
            task.k_values,
            ignore_identical_ids=task.ignore_identical_ids,
            skip_first_result=task.skip_first_result,
        )
        task_specific = task.task_specific_scores(
            all_scores,
            relevant_docs,
            retrieval_out,
            hf_split=split,
            hf_subset=hf_subset,
        )
        score_dict = make_score_dict(
            ndcg,
            _map,
            recall,
            precision,
            mrr,
            naucs,
            naucs_mrr,
            hit_rate,
            task_specific,
            task._previous_results_model_meta,
        )
        _ = time.perf_counter() - t0

    accelerator.wait_for_everyone()
    return score_dict


def main() -> None:
    args = parse_args()
    accelerator = Accelerator()

    meta = ModelMeta(
        name="moxu/colqwen3_vl_2B_lr5e-6_dim2048_stage3",
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

    vidore3_tasks = [
        "Vidore3ComputerScienceRetrieval.v2",
        "Vidore3EnergyRetrieval.v2",
        "Vidore3FinanceEnRetrieval.v2",
        "Vidore3FinanceFrRetrieval.v2",
        "Vidore3HrRetrieval.v2",
        "Vidore3PharmaceuticalsRetrieval.v2",
        "Vidore3PhysicsRetrieval.v2",
    ]
    tasks = mteb.get_tasks(tasks=vidore3_tasks, languages=["eng-Latn"])
    cache = mteb.ResultCache(cache_path="./mteb_results")
    encode_kwargs = {"batch_size": args.encode_batch_size}

    # ----- 单进程：沿用 mteb.evaluate -----
    if accelerator.num_processes == 1:
        model_wrapper = _build_model(args, str(accelerator.device))
        model_wrapper.mteb_model_meta = meta
        model_result = mteb.evaluate(
            model=model_wrapper,
            tasks=tasks,
            cache=cache,
            overwrite_strategy="only-missing",
            encode_kwargs=encode_kwargs,
        )
        for tr in model_result.task_results:
            path = cache.get_task_result_path(task_name=tr.task_name, model_name=meta)
            print(f"\n「{tr.task_name}」: {path.resolve()}")
        print("\n结果目录: ./mteb_results/results/")
        return

    # ----- 多进程：分片 encode + 主进程 MaxSim -----
    print(
        f"Accelerate rank={accelerator.process_index}/{accelerator.num_processes} "
        f"device={accelerator.device}"
    )

    task_results: list = []
    for task in tasks:
        if not isinstance(task, AbsTaskRetrieval):
            if accelerator.is_main_process:
                print(f"跳过非检索任务: {task.metadata.name}")
            continue
        task.load_data()
        task.convert_v1_dataset_format_to_v2(num_proc=None)

        split_scores: dict = {}
        task_t0 = time.perf_counter()
        for split in task.eval_splits:
            hf_subsets = task.hf_subsets or list(task.dataset.keys())
            for hf_subset in hf_subsets:
                t0 = time.perf_counter()
                scores = _evaluate_task_sharded(
                    accelerator,
                    task,
                    encode_kwargs,
                    args,
                    split,
                    hf_subset,
                )
                if accelerator.is_main_process and scores:
                    split_scores.setdefault(split, {})[hf_subset] = scores
                elapsed = time.perf_counter() - t0
                if accelerator.is_main_process and scores:
                    print(
                        f"完成 {task.metadata.name} {split=} {hf_subset=} "
                        f"eval_time≈{elapsed:.1f}s"
                    )

        if accelerator.is_main_process and split_scores:
            tr = TaskResult.from_task_results(
                task,
                {k: v for k, v in split_scores.items()},
                evaluation_time=time.perf_counter() - task_t0,
            )
            task_results.append(tr)
            cache.save_to_cache(tr, meta)

    if accelerator.is_main_process:
        for tr in task_results:
            path = cache.get_task_result_path(task_name=tr.task_name, model_name=meta)
            print(f"\n「{tr.task_name}」: {path.resolve()}")
        print("\n多进程结果已写入 ./mteb_results/results/（与单进程格式一致）")


if __name__ == "__main__":
    main()
