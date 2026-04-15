import sys
sys.path.insert(0, "/home/moxu/MMRAG/otherExp/colpali/mteb")

import argparse
import torch
import mteb
from accelerate import Accelerator
from mteb.models.model_implementations.colqwen_models import ColQwen3VLEmbeddingWrapper
from mteb.models.model_meta import ModelMeta, ScoringFunction


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MTEB 评测；accelerate 请使用 --num_processes 1。",
    )
    p.add_argument(
        "--model-path",
        type=str,
        default="/home/moxu/MMRAG/otherExp/colpali/output/checkpoint-50",
        help="训练 checkpoint：含 adapter_config.json 时底座 + LoRA merge；否则视为已 merge 的完整权重目录",
    )
    p.add_argument(
        "--hub-model-id",
        type=str,
        default="Qwen/Qwen3-VL-Embedding-2B",
        help="底座与 processor 的 Hugging Face id（PEFT 时从该 id 加载底座）",
    )
    p.add_argument("--max-num-visual-tokens", type=int, default=768)
    p.add_argument("--encode-batch-size", type=int, default=16)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    accelerator = Accelerator()

    if accelerator.num_processes != 1:
        raise RuntimeError(
            "仅支持单进程评测。请: accelerate launch --num_processes 1 eval.py\n"
            f"当前 num_processes={accelerator.num_processes}。"
        )

    print(
        f"Accelerate device={accelerator.device} "
        f"num_processes={accelerator.num_processes}"
    )

    model_wrapper = ColQwen3VLEmbeddingWrapper(
        model_name="Qwen/Qwen3-VL-Embedding-2B-colpali",
        hub_model_id=args.hub_model_id,
        peft_adapter_path=args.model_path,
        max_num_visual_tokens=args.max_num_visual_tokens,
        device=str(accelerator.device),
        similarity_use_max_sim=True,
        attn_implementation="sdpa",
        use_cache=False,
    )

    model_wrapper.mteb_model_meta = ModelMeta(
        name="moxu/colqwen3vl_2B_lr5e-6_dim768_token768_merged_stage0",
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
        embed_dim=768,
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
        "Vidore3IndustrialRetrieval.v2",
        "Vidore3PharmaceuticalsRetrieval.v2",
        "Vidore3PhysicsRetrieval.v2",
    ]
    tasks = mteb.get_tasks(tasks=vidore3_tasks, languages=["eng-Latn"])
    cache = mteb.ResultCache(cache_path="./mteb_results")

    model_result = mteb.evaluate(
        model=model_wrapper,
        tasks=tasks,
        cache=cache,
        overwrite_strategy="only-missing",
        encode_kwargs={"batch_size": args.encode_batch_size},
    )

    meta = model_wrapper.mteb_model_meta
    for tr in model_result.task_results:
        path = cache.get_task_result_path(task_name=tr.task_name, model_name=meta)
        print(f"\n📁 「{tr.task_name}」: {path.resolve()}")

    print("\n结果目录: ./mteb_results/results/ 下对应模型名与 revision。")


if __name__ == "__main__":
    main()
