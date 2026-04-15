#stage1: accelerate launch scripts/configs/qwen3/train_colqwen3_vl_embedding_model.py --output-dir /home/moxu/MMRAG/otherExp/colpali/output/colqwen3_vl_2B_lr2e-4_dim128_stage1_token --peft --only-train-projection --max-num-visual-tokens 1280 lr 2e-4
#stage2: accelerate launch scripts/configs/qwen3/train_colqwen3_vl_embedding_model.py --output-dir /home/moxu/MMRAG/otherExp/colpali/output/colqwen3_vl_2B_lr5e-6_dim2048_stage3 --peft --load-projection-only /home/moxu/MMRAG/otherExp/colpali/output/colqwen3_vl_2B_lr5e-6_dim2048_stage1/checkpoint-462 --target-modules '"^(?!.*(embed_tokens|patch_embed|merger)).*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|proj).*$"' --lr 5e-5
import argparse
import json
import shutil
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from colpali_engine.data.dataset import ColPaliEngineDataset
from colpali_engine.loss.late_interaction_losses import ColbertLoss, ColbertPairwiseCELoss
from colpali_engine.models import ColQwen3VLEmbedding, ColQwen3VLEmbeddingProcessor
from colpali_engine.trainer.colmodel_torch_training import ColModelTorchTraining
from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.utils.dataset_transformation import load_train_set
#accelerate launch
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--base-model", type=str, default="Qwen/Qwen3-VL-Embedding-2B",
                   help="基础模型路径或 HuggingFace model id")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--tau", type=float, default=0.02)
    p.add_argument("--trainer", type=str, default="hf", choices=["torch", "hf"])
    p.add_argument("--loss", type=str, default="ce", choices=["ce", "pairwise"])
    p.add_argument("--peft", action="store_true", help="是否开启 PEFT (LoRA) 模式")
    p.add_argument("--only-train-projection", action="store_true", help="Stage 1 特有：冻结所有参数包括 LoRA，仅练投影层")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--target-modules",
        type=str,
        default='["q_proj"]',
        help='LoRA 目标层，例如 \'["q_proj", "v_proj"]\'',
    )
    p.add_argument(
        "--projection-path",
        type=str,
        default=None,
        help="手动加载投影层权重 (.pt/.bin)",
    )
    p.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="从断点继续训练，可设为路径或 'true'",
    )
    p.add_argument(
        "--load-projection-only",
        type=str,
        default=None,
        help="Stage 2 专用：只加载投影层权重，不加载 optimizer 状态。例如指向 checkpoint-80 目录",
    )
    p.add_argument(
        "--max-num-visual-tokens",
        type=int,
        default=1280,
        help="ColQwen3VLEmbeddingProcessor 的 max_num_visual_tokens（默认 768，与此前脚本一致）",
    )
    return p.parse_args()

def _parse_target_modules(s: str) -> list[str]:
    try:
        v = json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"--target-modules 不是合法 JSON: {e}") from e
    return v

if __name__ == "__main__":
    args = parse_args()
    target_modules = _parse_target_modules(args.target_modules)
    loss_func = ColbertLoss(temperature=args.tau, normalize_scores=True) if args.loss == "ce" else ColbertPairwiseCELoss(normalize_scores=False)

    # 1. 处理投影层路径逻辑
    projection_path = args.projection_path

    # Stage 2 专用：只加载投影层权重，不加载 optimizer 状态
    if args.load_projection_only:
        proj_file = Path(args.load_projection_only) / "last_projection.pt"
        if proj_file.exists():
            projection_path = str(proj_file)
            print(f">>> [Stage 2] Loading projection from: {projection_path}")
        else:
            # 尝试从 merged model 加载
            merged_proj = Path(args.load_projection_only) / "pytorch_model.bin"
            if merged_proj.exists():
                projection_path = str(merged_proj)
                print(f">>> [Stage 2] Loading from merged model: {projection_path}")

    # resume_from_checkpoint 只用于 Stage 1 断点续跑
    if args.resume_from_checkpoint and args.resume_from_checkpoint.lower() != "none" and not args.load_projection_only:
        res_path = args.resume_from_checkpoint
        if res_path.lower() == "true":
            checkpoints = [d for d in Path(args.output_dir).iterdir() if d.name.startswith("checkpoint-")]
            res_path = str(max(checkpoints, key=lambda p: int(p.name.split("-")[1]))) if checkpoints else args.output_dir

        potential_proj = Path(res_path) / "last_projection.pt"
        if potential_proj.exists():
            projection_path = str(potential_proj)
            print(f">>> Found checkpoint projection: {projection_path}")

    # 2. 加载基础模型
    model = ColQwen3VLEmbedding.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        attn_implementation="sdpa",
        projection_path=projection_path,
    )

    # 3. 精度修复与梯度准备
    model.enable_input_require_grads()
    if hasattr(model, "visual"):
        model.visual.to(torch.bfloat16)
        type(model.visual).dtype = property(lambda self: torch.bfloat16)

    # 4. 构造训练配置
    peft_config = None
    if args.peft:
        peft_config = LoraConfig(
            r=32,
            lora_alpha=32,
            lora_dropout=0.1,
            init_lora_weights="gaussian",
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules=target_modules,
            modules_to_save=["custom_text_proj"], # 核心：投影层作为全量保存对象
        )

    config = ColModelTrainingConfig(
        output_dir=args.output_dir,
        processor=ColQwen3VLEmbeddingProcessor.from_pretrained(
            args.base_model,
            max_num_visual_tokens=args.max_num_visual_tokens,
        ),
        model=model,
        train_dataset=load_train_set() if not args.max_samples else ColPaliEngineDataset(
            load_dataset("vidore/colpali_train_set", split="train").shuffle(seed=42).select(range(args.max_samples)), pos_target_column_name="image"
        ),
        eval_dataset=ColPaliEngineDataset(load_dataset("vidore/colpali_train_set", split="test"), pos_target_column_name="image"),
        run_eval=True,
        loss_func=loss_func,
        tr_args=TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=32,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            per_device_eval_batch_size=32,
            gradient_accumulation_steps=4,
            eval_strategy="steps",
            save_steps=25,
            logging_steps=10,
            eval_steps=25,
            warmup_steps=25,
            learning_rate=args.lr,
            save_total_limit=3,
            bf16=True,
            # Stage 2 时明确不加载 optimizer 状态，只加载模型权重
            resume_from_checkpoint=args.resume_from_checkpoint if args.resume_from_checkpoint and args.resume_from_checkpoint.lower() != "none" and not args.load_projection_only else None,
        ),
        peft_config=peft_config,
    )

    # 5. 实例化 Trainer
    trainer = ColModelTraining(config) if args.trainer == "hf" else ColModelTorchTraining(config)

    # 6. Stage 1 强制冻结逻辑（实现真正的无损向量对齐）
    if args.peft and args.only_train_projection:
        print(">>> [Stage 1 Mode] Freezing all LoRA and Backbone parameters. Training ONLY custom_text_proj.")
        for name, param in trainer.model.named_parameters():
            if "custom_text_proj" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    # 7. 保存脚本副本并启动
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__), Path(config.output_dir) / Path(__file__).name)

    trainer.train()
    trainer.save()