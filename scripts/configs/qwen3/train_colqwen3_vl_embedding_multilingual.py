#!/usr/bin/env python3
"""
多语言训练脚本 - 使用 vdr-multilingual-train 数据集
基于已 merge 的 Qwen3-VL-Embedding-2B 模型继续训练

使用示例:
accelerate launch scripts/configs/qwen3/train_colqwen3_vl_embedding_multilingual.py \
  --output-dir /home/moxu/MMRAG/otherExp/colpali/output \
  --base-model /home/moxu/MMRAG/otherExp/colpali/merged_qwen3_vl_stage1 \
  --lr 5e-5 \
  --loss negative \
  --peft \
  --languages "en"
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments

from colpali_engine.data.dataset import ColPaliEngineDataset
from colpali_engine.loss.late_interaction_losses import (
    ColbertLoss,
    ColbertNegativeCELoss,
    ColbertPairwiseCELoss,
)
from colpali_engine.models import ColQwen3VLEmbedding, ColQwen3VLEmbeddingProcessor
from colpali_engine.trainer.colmodel_training import (
    ColModelTraining,
    ColModelTrainingConfig,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--base-model", type=str, default="/home/moxu/MMRAG/otherExp/colpali/merged_qwen3_vl_stage1")
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--tau", type=float, default=0.02)
    p.add_argument("--loss", type=str, default="negative", choices=["ce", "pairwise", "negative"])
    p.add_argument("--peft", action="store_true", help="是否开启 PEFT (LoRA) 模式")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--target-modules", type=str,
                   default='["q_proj"]',
                   help='LoRA 目标层')
    p.add_argument("--resume-from-checkpoint", type=str, default=None)
    p.add_argument("--max-num-visual-tokens", type=int, default=768)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--num-negatives", type=int, default=15)
    p.add_argument("--in-batch-weight", type=float, default=0.0)
    p.add_argument("--languages", type=str, default="en,it,fr,de,es")
    p.add_argument("--dataloader-num-workers", type=int, default=4,
                   help="Number of DataLoader workers for parallel data loading")
    return p.parse_args()


def _parse_target_modules(s: str):
    if s.startswith("^") or s.startswith("(?"):
        return s
    try:
        v = json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"--target-modules 不是合法 JSON: {e}") from e
    return v


def load_multilingual_dataset(languages: list, max_samples: int = None):
    """加载多语言数据集并混合

    Returns:
        tuple: (dataset, id_to_idx) where id_to_idx is a mapping from sample ID to index
               built from the FULL dataset (before truncation) to support negative sampling.
    """
    lang_datasets = []
    full_id_to_idx = {}  # Map from full dataset, used for negative resolution

    for lang in languages:
        print(f"Loading {lang} dataset...")
        ds = load_dataset("llamaindex/vdr-multilingual-train", lang, split="train")
        original_size = len(ds)
    
        # Build id_to_idx from FULL dataset for negative resolution
        # This allows negatives to reference any sample in the full dataset
        # Use ds["id"] to only load the id column, not the entire sample (avoids loading images)
        ids = ds["id"]
        for idx, sample_id in enumerate(ids):
            if sample_id is not None:
                full_id_to_idx[sample_id] = idx

        # Truncate data AFTER building the full id_to_idx
        if max_samples is not None and original_size > max_samples:
            ds = ds.select(range(max_samples))
        print(f"  {lang}: {len(ds)} samples (full id_to_idx has {original_size} entries)")
        lang_datasets.append(ds)

    combined = concatenate_datasets(lang_datasets)
    combined = combined.shuffle(seed=42)
    print(f"Total combined: {len(combined)} samples, id_to_idx entries: {len(full_id_to_idx)}")
   
    return combined, full_id_to_idx


def check_dataset_columns(dataset):
    columns = dataset.column_names
    print(f"Dataset columns: {columns}")
    query_col = "query" if "query" in columns else ("question" if "question" in columns else None)
    pos_col = "image" if "image" in columns else ("pos_image" if "pos_image" in columns else None)
    neg_col = "negatives" if "negatives" in columns else None
    return query_col, pos_col, neg_col


if __name__ == "__main__":
    args = parse_args()
    target_modules = _parse_target_modules(args.target_modules)
    languages = [l.strip() for l in args.languages.split(",")]

    # 1. 损失函数配置
    if args.loss == "ce":
        loss_func = ColbertLoss(temperature=args.tau, normalize_scores=True)
    elif args.loss == "pairwise":
        loss_func = ColbertPairwiseCELoss(normalize_scores=False)
    elif args.loss == "negative":
        loss_func = ColbertNegativeCELoss(
            temperature=args.tau,
            normalize_scores=True,
            in_batch_term_weight=args.in_batch_weight,
        )
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    # 2. 模型加载
    print(f"Loading base model: {args.base_model}")
    model = ColQwen3VLEmbedding.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        attn_implementation="sdpa",
    )

    # 3. 梯度与精度设置
    model.enable_input_require_grads()
    if hasattr(model, "visual"):
        model.visual.to(torch.bfloat16)

    # 4. 数据处理 (修复 AssertionError 的核心区域)
    print(f"\nLoading multilingual datasets: {languages}")
    dataset, id_to_idx = load_multilingual_dataset(languages, args.max_samples)

    # --- 核心修复：移除 map/cast 操作，在 Dataset 内部处理 None query ---
    print(">>> Skipping feature casting and sanitize map to avoid disk pressure...")

    query_col, pos_col, neg_col = check_dataset_columns(dataset)
    print(f"Using columns: query={query_col}, pos={pos_col}, neg={neg_col}")

    # 5. 封装为训练集
    # IMPORTANT: Pass id_to_idx from FULL dataset to support negative sampling
    # even when data is truncated with max_samples
    train_dataset = ColPaliEngineDataset(
        dataset,
        query_column_name=query_col,
        pos_target_column_name=pos_col,
        neg_target_column_name=neg_col,
        id_to_idx=id_to_idx,
    )

    # 6. PEFT (LoRA) 配置 - 由 ColModelTrainingConfig.__post_init__ 处理
    peft_config = None
    if args.peft:
        peft_config = LoraConfig(
            r=32, lora_alpha=32, lora_dropout=0.1,
            init_lora_weights="gaussian", bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules=target_modules,
        )

    # 7. 训练配置
    config = ColModelTrainingConfig(
        output_dir=args.output_dir,
        processor=ColQwen3VLEmbeddingProcessor.from_pretrained(
            args.base_model,
            max_num_visual_tokens=args.max_num_visual_tokens,
        ),
        model=model,
        train_dataset=train_dataset,
        eval_dataset=None, 
        loss_func=loss_func,
        tr_args=TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=args.batch_size,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            dataloader_num_workers=args.dataloader_num_workers,
            dataloader_pin_memory=True,
            dataloader_prefetch_factor=2 if args.dataloader_num_workers > 0 else None,
            logging_steps=5,
            save_steps=50,
            warmup_steps=20,
            eval_steps=50,
            learning_rate=args.lr,
            bf16=True,
            remove_unused_columns=False, # 重要：防止误删图像列
            resume_from_checkpoint=args.resume_from_checkpoint,
        ),
        peft_config=peft_config,
    )

    # 8. 启动训练
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    trainer = ColModelTraining(config)
    trainer.train()
    trainer.save()