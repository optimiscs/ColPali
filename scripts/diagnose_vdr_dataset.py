#!/usr/bin/env python3
"""
诊断脚本：检查 llamaindex/vdr-multilingual-train 数据集 (en)
"""
from datasets import load_dataset

def check_dataset_structure():
    """检查数据集的基本结构"""
    print("=" * 60)
    print("1. 检查数据集基本结构")
    print("=" * 60)

    lang = "en"
    ds = load_dataset("llamaindex/vdr-multilingual-train", lang, split="train")

    print(f"  总样本数: {len(ds)}")
    print(f"  列名: {ds.column_names}")
    print(f"  特征类型: {ds.features}")

    # 检查前几个样本
    print(f"\n  前3个样本:")
    for i in range(min(3, len(ds))):
        sample = ds[i]
        print(f"    Sample {i}:")
        print(f"      query: type={type(sample['query']).__name__}, value={str(sample['query'])[:50] if sample['query'] else None}...")
        print(f"      image: type={type(sample['image']).__name__}")
        print(f"      negatives: type={type(sample['negatives']).__name__}, len={len(sample['negatives'])}")


def check_none_queries():
    """检查 None query 出现的位置"""
    print("\n" + "=" * 60)
    print("2. 检查 None Query 出现的位置")
    print("=" * 60)

    lang = "en"
    ds = load_dataset("llamaindex/vdr-multilingual-train", lang, split="train")

    # 找到第一个 None
    first_none = None
    for i in range(len(ds)):
        if ds[i]['query'] is None:
            first_none = i
            break

    if first_none:
        print(f"  第一个 None query 出现在索引: {first_none}")
        print(f"  None query 总数: {len(ds) - first_none}")
        # 显示 None 前后几个样本
        start = max(0, first_none - 3)
        end = min(len(ds), first_none + 3)
        for i in range(start, end):
            q = ds[i]['query']
            print(f"    Sample {i}: {type(q).__name__}")
    else:
        print(f"  没有找到 None query")


def check_negatives_structure():
    """检查 negatives 列的结构"""
    print("\n" + "=" * 60)
    print("3. 检查 Negatives 列结构")
    print("=" * 60)

    ds = load_dataset("llamaindex/vdr-multilingual-train", "en", split="train")

    print(f"  negatives 类型: {type(ds[0]['negatives'])}")
    print(f"  negatives 长度: {len(ds[0]['negatives'])}")
    print(f"  negatives[0] 类型: {type(ds[0]['negatives'][0])}")
    print(f"  negatives[0] 内容示例: {str(ds[0]['negatives'][0])[:100]}...")

    # 检查是否有空列表
    empty_count = 0
    for i in range(min(1000, len(ds))):
        if len(ds[i]['negatives']) == 0:
            empty_count += 1
    print(f"  前1000个样本中空 negatives 列表数量: {empty_count}")


def check_colpali_dataset():
    """检查 ColPaliEngineDataset 处理后的结构"""
    print("\n" + "=" * 60)
    print("4. 检查 ColPaliEngineDataset 处理后的结构")
    print("=" * 60)

    from colpali_engine.data.dataset import ColPaliEngineDataset

    ds = load_dataset("llamaindex/vdr-multilingual-train", "en", split="train")
    ds_small = ds.select(range(10))

    train_dataset = ColPaliEngineDataset(
        ds_small,
        query_column_name='query',
        pos_target_column_name='image',
        neg_target_column_name='negatives',
    )

    for i in range(3):
        sample = train_dataset[i]
        print(f"\n  Sample {i}:")
        print(f"    query: type={type(sample['query']).__name__}")
        print(f"    pos_target: type={type(sample['pos_target']).__name__}")
        print(f"    neg_target: type={type(sample['neg_target']).__name__}")

        if isinstance(sample['pos_target'], list):
            print(f"      pos_target 是列表，长度={len(sample['pos_target'])}")
            print(f"      pos_target[0]: type={type(sample['pos_target'][0]).__name__}")

        if isinstance(sample['neg_target'], list):
            print(f"      neg_target 是列表，长度={len(sample['neg_target'])}")


if __name__ == "__main__":
    check_dataset_structure()
    check_none_queries()
    check_negatives_structure()
    check_colpali_dataset()

    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)
