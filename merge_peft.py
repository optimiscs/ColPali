"""
合并 PEFT adapter 和 base model，生成完整的 merged model。

Usage:
    python merge_peft.py \
        --adapter-path /home/moxu/MMRAG/otherExp/colpali/output/colqwen3_vl_embedding_2B_5e-6_dim768_v2/checkpoint-80 \
        --base-model Qwen/Qwen3-VL-Embedding-2B \
        --output-dir ./merged_model
"""
import argparse
import torch
from peft import PeftModel
from colpali_engine.models import ColQwen3VLEmbedding, ColQwen3VLEmbeddingProcessor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-path", type=str, required=True, help="PEFT adapter 目录路径")
    p.add_argument("--base-model", type=str, default="Qwen/Qwen3-VL-Embedding-2B", help="Base model 名称或路径")
    p.add_argument("--output-dir", type=str, required=True, help="Merged model 输出目录")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading base model: {args.base_model}")
    base_model = ColQwen3VLEmbedding.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
    )

    print(f"Loading PEFT adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)

    print("Merging adapter with base model...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to: {args.output_dir}")
    merged_model.save_pretrained(args.output_dir, max_shard_size="128MB")

    print("Saving processor...")
    processor = ColQwen3VLEmbeddingProcessor.from_pretrained(args.base_model, max_num_visual_tokens=768)
    processor.save_pretrained(args.output_dir)

    print("Done!")


if __name__ == "__main__":
    main()
