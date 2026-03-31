"""
ColQwen2 简单训练脚本 - 直接使用 PyTorch 训练
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["USE_LOCAL_DATASET"] = "0"

import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm

from colpali_engine.models import ColQwen2, ColQwen2Processor
from colpali_engine.loss.late_interaction_losses import ColbertLoss
from colpali_engine.data.dataset import ColPaliEngineDataset
from datasets import load_dataset


def main():
    # 配置
    batch_size = 64  # 每GPU
    grad_accum = 2   # 梯度累积
    lr = 2e-4
    epochs = 1
    output_dir = "/home/moxu/MMRAG/outputs/colqwen2_simple"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 加载模型
    print("Loading model...")
    model = ColQwen2.from_pretrained(
        "vidore/colqwen2-v1.0",
        torch_dtype=torch.bfloat16,
    )
    model = model.to(device)
    model.train()

    processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")

    # 加载数据集
    print("Loading dataset...")
    dataset = load_dataset("vidore/colpali_train_set", split="train")
    # 取前1000条样本测试
    dataset = dataset.select(range(min(1000, len(dataset))))

    # 创建数据加载器
    def collate_fn(batch):
        queries = [item["query"] for item in batch]
        images = [item["image"] for item in batch]

        queries_enc = processor.process_queries(queries)
        images_enc = processor.process_images(images)

        return {
            "queries": queries_enc,
            "images": images_enc
        }

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )

    # 优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = ColbertLoss(temperature=0.02)

    print(f"Training with {len(dataloader)} batches per epoch")
    print(f"Effective batch size: {batch_size * grad_accum}")

    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(pbar):
            queries = {k: v.to(device) for k, v in batch["queries"].items()}
            images = {k: v.to(device) for k, v in batch["images"].items()}

            # Forward
            query_emb = model(**queries)
            image_emb = model(**images)

            # Loss
            loss = criterion(query_emb, image_emb)
            loss = loss / grad_accum

            # Backward
            loss.backward()

            if (step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum
            pbar.set_postfix({"loss": f"{loss.item() * grad_accum:.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

    # 保存模型
    print(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
