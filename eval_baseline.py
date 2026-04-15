from __future__ import annotations

import gc
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "mteb"))

import argparse

import torch
from accelerate import Accelerator


_ORIGINAL_FIND_SPEC = importlib.util.find_spec


def _patched_find_spec(name: str, package: str | None = None):
    if name == "torchvision" or name.startswith("torchvision."):
        return None
    return _ORIGINAL_FIND_SPEC(name, package)


importlib.util.find_spec = _patched_find_spec
try:
    import mteb
finally:
    importlib.util.find_spec = _ORIGINAL_FIND_SPEC


DEFAULT_VIDORE3_V2_TASKS = [
    "Vidore3ComputerScienceRetrieval.v2",
    "Vidore3EnergyRetrieval.v2",
    "Vidore3FinanceEnRetrieval.v2",
    "Vidore3HrRetrieval.v2",
    "Vidore3IndustrialRetrieval.v2",
    "Vidore3PharmaceuticalsRetrieval.v2",
    "Vidore3PhysicsRetrieval.v2",
]

DEFAULT_BASELINES = [
    "Qwen/Qwen3-VL-Embedding-2B",
    "Qwen/Qwen3-VL-Embedding-2B-colpali",
    "vidore/colqwen2.5-v0.2",
    "vidore/colpali-v1.3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用 MTEB 内置实现批量评测 baseline，结果直接写入 ./mteb_results/results/",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=DEFAULT_BASELINES,
        help="MTEB registry 里的模型名列表。",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_VIDORE3_V2_TASKS,
        help="要跑的任务列表。",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default="./mteb_results",
        help="MTEB 结果缓存目录。",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=8,
        help="编码 batch size。",
    )
    parser.add_argument(
        "--overwrite-strategy",
        choices=["always", "never", "only-missing", "only-cache"],
        default="only-missing",
        help="结果覆盖策略。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="手动指定 device；默认跟随 accelerate。",
    )
    parser.add_argument(
        "--show-progress-bar",
        action="store_true",
        help="显示 MTEB 进度条。",
    )
    return parser.parse_args()


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    accelerator = Accelerator()

    if accelerator.num_processes != 1:
        raise RuntimeError(
            "仅支持单进程评测。请使用: accelerate launch --num_processes 1 eval_baseline.py\n"
            f"当前 num_processes={accelerator.num_processes}。"
        )

    device = args.device or str(accelerator.device)
    if device == "cuda":
        device = "cuda:0"
    cache = mteb.ResultCache(cache_path=args.cache_path)
    tasks = mteb.get_tasks(tasks=args.tasks, languages=["eng-Latn"])

    print(
        f"Accelerate device={accelerator.device} "
        f"num_processes={accelerator.num_processes}"
    , flush=True)
    print(f"Running baselines: {args.baselines}", flush=True)
    print(f"Tasks: {args.tasks}", flush=True)

    for model_name in args.baselines:
        print(f"\n=== {model_name} ===", flush=True)
        cleanup_memory()

        meta = mteb.get_model_meta(model_name).model_copy(deep=True)
        model = None
        try:
            model = meta.load_model(device=device)
            model_result = mteb.evaluate(
                model=model,
                tasks=tasks,
                cache=cache,
                raise_error=False,
                overwrite_strategy=args.overwrite_strategy,
                encode_kwargs={"batch_size": args.encode_batch_size},
                show_progress_bar=args.show_progress_bar,
            )
        finally:
            del model
            cleanup_memory()

        for task_result in model_result.task_results:
            result_path = cache.get_task_result_path(
                task_name=task_result.task_name,
                model_name=meta,
            )
            print(f"{task_result.task_name}: {result_path.resolve()}", flush=True)

        if model_result.exceptions:
            for exc in model_result.exceptions:
                print(f"FAILED {exc.task_name}: {exc.exception}", flush=True)
            print(
                "继续保留已完成任务；重新运行同一命令即可按 only-missing 续跑失败任务。",
                flush=True,
            )

    print(
        "\n结果都在 ./mteb_results/results/ 下，直接看各任务 json 里的 main_score 即可。",
        flush=True,
    )


if __name__ == "__main__":
    main()
