#!/bin/bash

# 1. 强制激活环境
source /home/moxu/miniconda3/etc/profile.d/conda.sh
conda activate colpali

# 2. 定义具体的编译器路径（用绝对路径最稳）
export CC=/home/moxu/miniconda3/envs/colpali/bin/x86_64-conda-linux-gnu-cc
export CXX=/home/moxu/miniconda3/envs/colpali/bin/x86_64-conda-linux-gnu-c++

# 3. 验证版本（注意这里要查 $CC 的版本，而不是 gcc 的版本）
echo "✅ 正在验证 Conda 编译器..."
$CC --version | head -n 1


# 5. 清理之前的失败残留
rm -rf /home/moxu/.cache/torch_extensions/py311_cu128/cpu_adam

# 6. 启动训练
# 请确保 --dataset_name 和 --model_name_or_path 指向你服务器上的【本地绝对路径】
accelerate launch \
    /home/moxu/MMRAG/otherExp/colpali/scripts/configs/qwen3/train_colqwen3_vl_embedding_model.py \
    --output-dir /home/moxu/MMRAG/otherExp/colpali/output/colqwen3_vl_embedding_2B_5e-6 \
    --peft
target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "custom_text_proj"],
target_modules="(.*(model)(?!.*visual).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)",