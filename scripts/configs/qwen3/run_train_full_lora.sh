#!/bin/bash

# 1. 强制激活环境
source /home/miniconda3/etc/profile.d/conda.sh
conda activate colpali

# 2. 定义具体的编译器路径（用绝对路径最稳）
export CC=/home/moxu/miniconda3/envs/colpali/bin/x86_64-conda-linux-gnu-cc
export CXX=/home/moxu/miniconda3/envs/colpali/bin/x86_64-conda-linux-gnu-c++

# 3. 验证版本
echo "正在验证 Conda 编译器..."
$CC --version | head -n 1

# 4. 清理之前的失败残留
rm -rf /home/moxu/.cache/torch_extensions/py311_cu128/cpu_adam

# 5. 全面LoRA微调：基于 checkpoint-80，扩大 target_modules
# 全面 LoRA target_modules 覆盖:
# - language_model: q,k,v,o_proj + gate,up,down_proj (FFN)
# - visual: vision_model 的 attention 相关模块 (blocks.*.attn.qkv, blocks.*.attn.proj)
# - custom_text_proj: 投影层

accelerate launch \
    /home/moxu/MMRAG/otherExp/colpali/scripts/configs/qwen3/train_colqwen3_vl_embedding_model.py \
    --output-dir /home/moxu/MMRAG/otherExp/colpali/output/colqwen3_vl_embedding_2B_full_lora \
    --peft \
    --lr 2e-5 \
    --target-modules '["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "custom_text_proj", "blocks\\..*\\.attn\\.qkv", "blocks\\..*\\.attn\\.proj"]' \
    --resume-from-checkpoint /home/moxu/MMRAG/otherExp/colpali/output/colqwen3_vl_embedding_2B_5e-6_dim768_v2/checkpoint-80
