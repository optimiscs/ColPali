"""
ColQwen2 训练脚本 - 适配双3090
"""
import argparse
import os
import shutil
from pathlib import Path

# 设置使用HuggingFace远程数据集
os.environ["USE_LOCAL_DATASET"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import TrainingArguments

from colpali_engine.data.dataset import ColPaliEngineDataset
from colpali_engine.loss.late_interaction_losses import ColbertLoss
from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.utils.dataset_transformation import load_train_set


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, required=True, help="output directory")
    p.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    p.add_argument("--tau", type=float, default=0.02, help="temperature")
    p.add_argument("--epochs", type=int, default=3, help="number of epochs")
    p.add_argument("--batch-size", type=int, default=8, help="per-device batch size")
    p.add_argument("--grad-accum", type=int, default=16, help="gradient accumulation steps")
    p.add_argument("--use-peft", action="store_true", default=True, help="use PEFT/LoRA (default: True)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    loss_func = ColbertLoss(
        temperature=args.tau,
        normalize_scores=True,
        use_smooth_max=False,
        pos_aware_negative_filtering=False,
    )

    # 模型配置 - 使用双卡
    config = ColModelTrainingConfig(
        output_dir=args.output_dir,
        processor=ColQwen2Processor.from_pretrained(
            "vidore/colqwen2-v1.0",
            max_num_visual_tokens=768,
        ),
        model=ColQwen2.from_pretrained(
            "vidore/colqwen2-v1.0",
            torch_dtype=torch.bfloat16,
        ),
        train_dataset=load_train_set(),
        eval_dataset=None,  # 简化评估
        run_eval=False,
        loss_func=loss_func,
        tr_args=TrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            per_device_eval_batch_size=32,
            eval_strategy="no",
            dataloader_num_workers=4,
            save_steps=100,
            logging_steps=10,
            warmup_steps=100,
            learning_rate=args.lr,
            save_total_limit=2,
            report_to="none",
            # 多卡配置
            local_rank=-1,  # 使用DataParallel
            # 优化配置
            bf16=True,
            # 禁用deepspeed
            fsdp="",
            deepspeed=None,
            logging_first_step=True,
        ),
        peft_config=LoraConfig(
            r=32,
            lora_alpha=32,
            lora_dropout=0.1,
            init_lora_weights="gaussian",
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules="(.*(model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)",
        ) if args.use_peft else None,
    )

    # 确保输出目录存在
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__), Path(config.output_dir) / Path(__file__).name)

    print(f"Starting training with:")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Gradient accumulation: {args.grad_accum}")
    print(f"  - Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - PEFT: {args.use_peft}")

    trainer = ColModelTraining(config)
    trainer.train()
    trainer.save()
    print("Training complete!")
