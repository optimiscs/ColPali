import torch
from peft import PeftModel
from colpali_engine.models import ColQwen3VLEmbedding, ColQwen3VLEmbeddingProcessor


def main():
    # ========== 配置区 ==========
    base_model_name = "Qwen/Qwen3-VL-Embedding-2B"
    lora_checkpoint = "output/colqwen3_vl_embedding_2B_5e-6/checkpoint-250"
    output_dir = "output/colqwen3_vl_embedding_2B_5e-6/merged_model"
    max_num_visual_tokens = 768
    # ============================

    print(f"Loading base model: {base_model_name}")
    base_model = ColQwen3VLEmbedding.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
    )

    print(f"Loading LoRA adapter from: {lora_checkpoint}")
    model = PeftModel.from_pretrained(base_model, lora_checkpoint)

    print("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to: {output_dir}")
    merged_model.save_pretrained(output_dir)

    print("Saving processor...")
    processor = ColQwen3VLEmbeddingProcessor.from_pretrained(
        base_model_name,
        max_num_visual_tokens=max_num_visual_tokens,
    )
    processor.save_pretrained(output_dir)

    print("Done! Merged model saved to:", output_dir)


if __name__ == "__main__":
    main()
