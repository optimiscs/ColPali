"""
计算微调后模型的 NDCG@10
使用 Vidore3IndustrialOCRRetrieval english subset
"""
import torch
from PIL import Image
from tqdm import tqdm
import pytrec_eval

from peft import PeftModel
from colpali_engine.models.qwen3.colqwen3_vl_embedding import ColQwen3VLEmbedding
from colpali_engine.models.qwen3.colqwen3_vl_embedding import ColQwen3VLEmbeddingProcessor
from mteb.abstasks.retrieval_dataset_loaders import RetrievalDatasetLoader


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


def extract_all_embeddings(model, processor, corpus, queries, batch_size: int = 4):
    """提取所有文档和查询的 embeddings"""
    model.eval()

    # 提取文档 embeddings
    print(f"\n🔢 提取文档 embeddings ({len(corpus)} docs)...")
    doc_embeddings = []
    for i in tqdm(range(0, len(corpus), batch_size), desc="文档 embeddings"):
        batch_docs = [corpus[idx] for idx in range(i, min(i+batch_size, len(corpus)))]
        batch_images = []
        for doc in batch_docs:
            img = doc.get('image')
            if img is not None and isinstance(img, Image.Image):
                batch_images.append(img.convert("RGB"))
            else:
                batch_images.append(Image.new("RGB", (224, 224), color="white"))

        with torch.no_grad():
            batch = processor.process_images(batch_images).to(model.device)
            embs = model(**batch)
            for j in range(embs.shape[0]):
                doc_embeddings.append(embs[j].cpu())

    print(f"文档 embeddings: {len(doc_embeddings)}")

    # 提取查询 embeddings
    print(f"\n🔢 提取查询 embeddings ({len(queries)} queries)...")
    query_embeddings = []
    valid_query_indices = []
    for i in tqdm(range(0, len(queries), batch_size), desc="查询 embeddings"):
        batch_queries = []
        batch_indices = []
        for idx in range(i, min(i+batch_size, len(queries))):
            text = queries[idx].get('text', '')
            if text and len(text.strip()) > 0:
                batch_queries.append(text)
                batch_indices.append(idx)

        if len(batch_queries) == 0:
            continue

        try:
            with torch.no_grad():
                batch = processor.process_queries(batch_queries).to(model.device)
                embs = model(**batch)
                for j in range(embs.shape[0]):
                    query_embeddings.append(embs[j].cpu())
                    valid_query_indices.append(batch_indices[j])
        except Exception as e:
            print(f"\n警告: 批次 {i} 处理失败: {e}")
            # 用零向量替代
            dummy_emb = torch.zeros(1, 128, dtype=torch.float32)
            for j in range(len(batch_queries)):
                query_embeddings.append(dummy_emb.squeeze(0))
                valid_query_indices.append(batch_indices[j])

    print(f"查询 embeddings: {len(query_embeddings)}")

    return doc_embeddings, query_embeddings


def compute_scores_matrix(doc_embeddings, query_embeddings, processor):
    """计算查询-文档相似度分数矩阵"""
    print("\n📊 计算相似度分数矩阵...")
    n_queries = len(query_embeddings)
    n_docs = len(doc_embeddings)

    scores_matrix = []
    for i in tqdm(range(n_queries), desc="计算分数"):
        query_emb = query_embeddings[i]
        row_scores = []
        for j in range(n_docs):
            doc_emb = doc_embeddings[j]
            # MaxSim: 对查询的每个 token，找最相似的文档 token，然后求平均
            score = processor.score_multi_vector(
                query_emb.unsqueeze(0).to(query_emb.device),
                doc_emb.unsqueeze(0).to(doc_emb.device)
            )
            row_scores.append(score.item())
        scores_matrix.append(row_scores)

    return torch.tensor(scores_matrix)


def compute_ndcg(scores_matrix, qrels, corpus_ids, query_ids, k_values=[10]):
    """使用 pytrec_eval 计算 NDCG@K"""
    print("\n📈 计算 NDCG@K...")

    # 构建 pytrec_eval 需要的格式
    # qrels: {query_id: {doc_id: relevance}}
    # scores: {query_id: {doc_id: score}}

    # 创建 id 到索引的映射
    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(corpus_ids)}
    query_id_to_idx = {qid: idx for idx, qid in enumerate(query_ids)}

    # 构建 qrels 字典 (pytrec_eval 格式)
    qrels_dict = {}
    for qid, rel_docs in qrels.items():
        qrels_dict[qid] = {}
        for doc_id, rel in rel_docs.items():
            qrels_dict[qid][doc_id] = float(rel)

    # 构建 run 字典 (pytrec_eval 格式)
    run_dict = {}
    for i, qid in enumerate(query_ids):
        run_dict[qid] = {}
        for j, doc_id in enumerate(corpus_ids):
            run_dict[qid][doc_id] = scores_matrix[i][j].item()

    # 使用 pytrec_eval 计算
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, {'ndcg'})
    results = evaluator.evaluate(run_dict)

    # 汇总结果
    metrics = {}
    for k in k_values:
        ndcg_key = f'ndcg_cut_{k}'
        ndcg_sum = 0.0
        count = 0
        for qid in results:
            if ndcg_key in results[qid]:
                ndcg_sum += results[qid][ndcg_key]
                count += 1
        if count > 0:
            metrics[f'NDCG@{k}'] = ndcg_sum / count

    return metrics


def main():
    MODEL_PATH = "/home/moxu/MMRAG/otherExp/colpali/output/colqwen3_vl_embedding_test"

    # 加载模型
    model, processor = load_model(MODEL_PATH)
    model.to("cuda")

    # 加载 Vidore3IndustrialRetrieval 数据集
    print("\n📂 加载 Vidore3IndustrialRetrieval english subset...")
    loader = RetrievalDatasetLoader(
        hf_repo='mteb/Vidore3IndustrialOCRRetrieval',
        revision='ff40e351f82d26dc8b406edf13b6471f00e378d0',
        split='test',
        config='english',
    )

    data = loader.load()
    corpus = data['corpus']
    queries = data['queries']
    qrels = data['relevant_docs']

    print(f"文档数: {len(corpus)}")
    print(f"查询数: {len(queries)}")
    print(f"相关文档条目: {len(qrels)}")

    # 获取 ID 列表
    corpus_ids = [doc['id'] for doc in corpus]
    query_ids = [q['id'] for q in queries]

    # 提取 embeddings
    doc_embeddings, query_embeddings = extract_all_embeddings(
        model, processor, corpus, queries, batch_size=4
    )

    # 计算分数矩阵
    scores_matrix = compute_scores_matrix(doc_embeddings, query_embeddings, processor)
    print(f"分数矩阵形状: {scores_matrix.shape}")

    # 计算 NDCG@10
    metrics = compute_ndcg(scores_matrix, qrels, corpus_ids, [query_ids[i] for i in valid_query_indices], k_values=[1, 5, 10])

    print("\n📈 评估结果:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    print("\n✅ 评估完成!")


if __name__ == "__main__":
    main()