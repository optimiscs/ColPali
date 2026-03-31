"""
ColQwen3VLEmbedding Late Interaction 评估脚本
专门处理 ColBERT/ColPali 风格的多向量 embeddings
"""
import torch
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

from peft import PeftModel
from colpali_engine.models.qwen3.colqwen3_vl_embedding import ColQwen3VLEmbedding
from colpali_engine.models.qwen3.colqwen3_vl_embedding import ColQwen3VLEmbeddingProcessor


def load_model(model_path: str, base_model_name: str = "Qwen/Qwen3-VL-Embedding-2B"):
    """加载微调后的模型"""
    print(f"📦 加载模型: {model_path}")

    base_model = ColQwen3VLEmbedding.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
    )

    model = PeftModel.from_pretrained(base_model, model_path, torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    processor = ColQwen3VLEmbeddingProcessor.from_pretrained(model_path)

    return model, processor


def extract_embeddings(model, processor, images, batch_size: int = 4):
    """批量提取图像 embeddings"""
    model.eval()
    all_embeddings = []

    for i in tqdm(range(0, len(images), batch_size), desc="提取图像 embeddings"):
        batch_images = images[i:i+batch_size]
        batch_images = [img.convert("RGB") if isinstance(img, Image.Image) else img for img in batch_images]

        with torch.no_grad():
            batch = processor.process_images(batch_images).to(model.device)
            embeddings = model(**batch)
            all_embeddings.append(embeddings.cpu())

    # 合并所有 embeddings
    if len(all_embeddings) > 0 and hasattr(all_embeddings[0], '__iter__'):
        all_embeddings = [e for batch_emb in all_embeddings for e in torch.unbind(batch_emb)]

    return all_embeddings


def compute_metrics(scores: torch.Tensor, k_values: list = [1, 5, 10]):
    """
    计算检索指标
    scores: (n_queries, n_docs) 相似度分数矩阵
    """
    n_queries, n_docs = scores.shape
    metrics = {}

    for k in k_values:
        recall_sum = 0.0
        mrr_sum = 0.0

        for i in range(n_queries):
            # 获取 top-k
            top_indices = torch.topk(scores[i], min(k, n_docs)).indices.tolist()

            # 假设第 i 个查询对应的正例文档就是第 i 个 (简化假设)
            # 实际评估需要 ground truth
            if i < n_docs:
                # 理想情况：第 i 个查询应该匹配第 i 个文档
                if i in top_indices:
                    rank = top_indices.index(i) + 1
                    recall_sum += 1.0 / min(k, n_docs)  # Recall@K
                    mrr_sum += 1.0 / rank  # MRR

        metrics[f'Recall@{k}'] = recall_sum / n_queries
        metrics[f'MRR@{k}'] = mrr_sum / n_queries

    return metrics


def main():
    MODEL_PATH = "/home/moxu/MMRAG/otherExp/colpali/output/colqwen3_vl_embedding_test"

    # 加载模型
    model, processor = load_model(MODEL_PATH)
    model.to("cuda")

    # 加载测试数据
    print("\n📂 加载 Vidore 测试集...")
    dataset = load_dataset("vidore/colpali_train_set", split="test", streaming=False)

    # 取一部分测试 (资源限制)
    max_samples = 50
    dataset = dataset.select(range(min(max_samples, len(dataset))))
    print(f"测试样本数: {len(dataset)}")

    # 分离图像和查询
    images = []
    queries = []
    for sample in dataset:
        if "image" in sample:
            images.append(sample["image"])
        if "query" in sample:
            queries.append(sample["query"])
        elif "question" in sample:
            queries.append(sample["question"])

    print(f"图像数: {len(images)}, 查询数: {len(queries)}")

    if len(images) == 0 or len(queries) == 0:
        print("❌ 数据集中没有图像或查询")
        return

    # 提取 embeddings
    print("\n🔢 提取 embeddings...")

    # 图像 embeddings
    image_embeddings = []
    batch_size = 2  # 小 batch 防止 OOM
    for i in tqdm(range(0, len(images), batch_size), desc="图像 embeddings"):
        batch_images = images[i:i+batch_size]
        batch_images_pil = []
        for img in batch_images:
            if isinstance(img, Image.Image):
                batch_images_pil.append(img.convert("RGB"))
            else:
                batch_images_pil.append(Image.new("RGB", (224, 224), color="white"))

        with torch.no_grad():
            batch = processor.process_images(batch_images_pil).to(model.device)
            embs = model(**batch)
            # embs shape: (batch, seq_len, dim)
            for j in range(embs.shape[0]):
                image_embeddings.append(embs[j])

    # 查询 embeddings
    query_embeddings = []
    for i in tqdm(range(0, len(queries), batch_size), desc="查询 embeddings"):
        batch_queries = queries[i:i+batch_size]
        with torch.no_grad():
            batch = processor.process_queries(batch_queries).to(model.device)
            embs = model(**batch)
            for j in range(embs.shape[0]):
                query_embeddings.append(embs[j])

    print(f"\n图像 embeddings: {len(image_embeddings)}")
    print(f"查询 embeddings: {len(query_embeddings)}")

    # 计算相似度分数 (late interaction: MaxSim)
    print("\n📊 计算相似度分数 (MaxSim)...")
    n_queries = min(len(query_embeddings), 20)  # 限制查询数
    n_docs = min(len(image_embeddings), 100)   # 限制文档数

    scores_matrix = []
    for i in tqdm(range(n_queries), desc="计算分数"):
        query_emb = query_embeddings[i]
        row_scores = []
        for j in range(n_docs):
            doc_emb = image_embeddings[j]
            # MaxSim: 对查询的每个 token，找最相似的文档 token，然后求平均
            score = processor.score_multi_vector(
                query_emb.unsqueeze(0),
                doc_emb.unsqueeze(0)
            )
            row_scores.append(score.item())
        scores_matrix.append(row_scores)

    scores_matrix = torch.tensor(scores_matrix)
    print(f"分数矩阵形状: {scores_matrix.shape}")

    # 计算指标
    print("\n📈 评估结果:")
    metrics = compute_metrics(scores_matrix, k_values=[1, 5, 10])
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    # 示例: 展示 top-3 最相关的文档
    print("\n🔍 示例 - 查询 0 的 top-3 匹配:")
    top3 = torch.topk(scores_matrix[0], 3).indices.tolist()
    for rank, doc_idx in enumerate(top3, 1):
        print(f"  Rank {rank}: 文档 {doc_idx}, 分数: {scores_matrix[0][doc_idx]:.4f}")

    print("\n✅ 评估完成!")


if __name__ == "__main__":
    main()
