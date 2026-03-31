import torch
from peft import PeftModel
from colpali_engine.models import ColQwen3VLEmbedding

# 1. 必须用 float32 加载，防止相加时精度溢出
base_model = ColQwen3VLEmbedding.from_pretrained(
    "Qwen/Qwen3-VL-Embedding-2B", 
    torch_dtype=torch.float32, 
    device_map="cpu" # 内存够大建议在内存操作，更稳
)

# 2. 加载 Adapter
model = PeftModel.from_pretrained(base_model, "/home/moxu/MMRAG/otherExp/colpali/output/colqwen3_vl_embedding_2B_5e-6_dim768_v2/checkpoint-80")

# 3. 官方合并（它会自动处理 Key 的命名，不需要你手动 replace）
merged_model = model.merge_and_unload()

# 4. 转回 bfloat16 准备保存
merged_model.to(torch.bfloat16)

# 5. 使用 .bin 格式保存（避开你之前的 header too large 报错）
merged_model.save_pretrained(
        "./merged_model_final", 
        max_shard_size="128MB",     # 你提到的 128MB 限制
        safe_serialization=True,    # 坚持使用 safetensors
    )