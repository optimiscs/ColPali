"""
官方 Qwen3-VL-Embedding-2B 基座评测（MTEB 内置 qwen3_vl_embedding_2b）。

使用 mteb.models.model_implementations.qwen3_vl_embedding_models 中的
Qwen3VLEmbeddingWrapper：last-token 池化 + L2 归一化 + 余弦相似度（dense），
与 colpali_engine 的 ColBERT/LoRA 路径无关。

LoRA 微调对比请使用 eval.py。

依赖: pip install 'mteb[qwen-vl]'

用法:
  python eval_ablation_base.py
  python eval_ablation_base.py --tasks Vidore3EnergyRetrieval.v2 --cache-dir ./mteb_results_ablation_base
"""

from __future__ import annotations

import argparse

import mteb
from mteb.models.model_implementations.qwen3_vl_embedding_models import qwen3_vl_embedding_2b


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MTEB：官方 Qwen3-VL-Embedding-2B（qwen3_vl_embedding_2b）")
    p.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=[
            "Vidore3ComputerScienceRetrieval.v2",
            "Vidore3EnergyRetrieval.v2",
            "Vidore3FinanceEnRetrieval.v2",
            "Vidore3FinanceFrRetrieval.v2",
            "Vidore3HrRetrieval.v2",
            "Vidore3IndustrialRetrieval.v2",
            "Vidore3PharmaceuticalsRetrieval.v2",
            "Vidore3PhysicsRetrieval.v2",
        ],
        help="MTEB 任务名",
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default="./mteb_results_ablation_base",
        help="ResultCache 根目录",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda:1", help="cuda / cpu，默认自动")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    meta = qwen3_vl_embedding_2b
    model = meta.load_model(device=args.device)

    tasks = mteb.get_tasks(tasks=list(args.tasks),languages=["eng-Latn"])
    cache = mteb.ResultCache(cache_path=args.cache_dir)
    overwrite = "always" if args.overwrite else "only-missing"

    model_result = mteb.evaluate(
        model=model,
        tasks=tasks,
        cache=cache,
        overwrite_strategy=overwrite,
        show_progress_bar=True,
    )

    for tr in model_result.task_results:
        path = cache.get_task_result_path(task_name=tr.task_name, model_name=meta)
        print(f"\n📁 任务「{tr.task_name}」结果: {path.resolve()}")

    print("\n模型: 官方 qwen3_vl_embedding_2b（单向量 + cosine）。LoRA 结果见 eval.py。")


if __name__ == "__main__":
    main()
