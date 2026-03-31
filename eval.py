import torch
import mteb
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType, Array, BatchedInput
from peft import PeftModel
from tqdm.auto import tqdm
from typing import Any, TYPE_CHECKING
import torchvision.transforms.functional as F
from PIL import Image

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from mteb.abstasks.task_metadata import TaskMetadata

class ColQwen3IndustrialWrapper(AbsEncoder):
    """
    遵循 MTEB 官方 ColQwen3 标准格式封装的工业微调版模型。
    集成 LoRA 权重加载与多模态 RAG 优化。
    """

    def __init__(
        self, 
        model_path: str, 
        base_model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        device: str | None = None,
        **kwargs
    ):
        from colpali_engine.models.qwen3.colqwen3_vl_embedding import ColQwen3VLEmbedding
        from colpali_engine.models.qwen3.colqwen3_vl_embedding import ColQwen3VLEmbeddingProcessor
        import os

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 检测是否为 merged 完整模型（merged checkpoint 有 adapter_config.json 则为 PEFT）
        adapter_config_path = os.path.join(model_path, "adapter_config.json")

        if os.path.exists(adapter_config_path):
            # PEFT adapter - 先加载 base 再应用 adapter
            print(f"📦 正在加载底座模型: {base_model_name}")
            base = ColQwen3VLEmbedding.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                **kwargs
            )
            print(f"🔗 LoRA 适配器加载中: {model_path}")
            self.model = PeftModel.from_pretrained(base, model_path)
            self.model = self.model.merge_and_unload()  # 合并权重提升推理效率
            self.model = self.model.to(self.device)  # 确保模型在同一设备
        else:
            # Merged 完整模型 - 直接加载
            print(f"📦 直接加载 Merged 模型: {model_path}")
            self.model = ColQwen3VLEmbedding.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                **kwargs
            )

        self.model.eval()

        # 2. 初始化 Processor 并限制视觉 Token (关键：防内存爆炸)
        self.processor = ColQwen3VLEmbeddingProcessor.from_pretrained(
            base_model_name,
            max_num_visual_tokens=1280  # 官方推荐阈值，平衡精度与内存
        )

    def _encode_inputs(self, encoded_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """核心修正：清理 RoPE 缓存，防止 Batch 间维度冲突"""
        if hasattr(self.model, "rope_deltas"):
            self.model.rope_deltas = None
        return self.model(**encoded_inputs)

    @torch.no_grad()
    def encode(
        self,
        inputs: "DataLoader[BatchedInput]",
        *,
        task_metadata: "TaskMetadata",
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> "Array":
        """遵循 AbsEncoder 标准接口"""
        return self.get_fused_embeddings(inputs, prompt_type=prompt_type, **kwargs)

    def get_fused_embeddings(
        self,
        dataloader: "DataLoader[BatchedInput]",
        prompt_type: PromptType | None = None,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> "Array":
        all_embeds: list[torch.Tensor] = []
        
        for batch in tqdm(dataloader, disable=not show_progress_bar, desc="Encoding"):
            # 自动识别输入类型
            is_query = (prompt_type == PromptType.query) or ("image" not in batch)
            
            if is_query:
                texts = batch["text"]
                if not isinstance(texts, (list, tuple)): texts = [texts]
                # 处理文本查询
                feat = self.processor.process_queries(texts=list(texts))
            else:
                # 处理图像文档
                imgs = [
                    F.to_pil_image(b.to(self.device)) if not isinstance(b, Image.Image) else b 
                    for b in batch["image"]
                ]
                imgs = [img.convert("RGB") for img in imgs]
                feat = self.processor.process_images(imgs)

            # 搬运到设备并推理
            feat = {k: v.to(self.device) for k, v in feat.items()}
            outs = self._encode_inputs(feat)
            
            # 统一转为 float32 存入 CPU
            all_embeds.extend(outs.detach().cpu().to(torch.float32))

        # 变长序列 Padding
        return torch.nn.utils.rnn.pad_sequence(all_embeds, batch_first=True, padding_value=0)

    def similarity(self, q_embeds: torch.Tensor, d_embeds: torch.Tensor) -> torch.Tensor:
        from mteb.similarity_functions import max_sim

        # 将 query 移到 GPU，corpus embeddings 保留在 CPU（避免 21GB+ 的 GPU OOM）
        q_embeds = q_embeds.to(self.device)

        # 分批处理 corpus embeddings（全部在 CPU 上计算）
        # ColBERT max_sim 是内存密集型操作，CPU 虽然慢但能处理大 tensor
        corpus_chunk_size = 200  # 每批处理的 doc 数量
        num_docs = d_embeds.size(0)

        # query 分批，减少每次计算量
        query_step = 4

        final_scores = None

        for chunk_start in range(0, num_docs, corpus_chunk_size):
            chunk_end = min(chunk_start + corpus_chunk_size, num_docs)
            d_chunk = d_embeds[chunk_start:chunk_end]  # 仍在 CPU 上

            # 对当前 corpus chunk 计算与所有 query 的相似度
            chunk_scores_list = []
            for q_start in range(0, q_embeds.size(0), query_step):
                q_end = min(q_start + query_step, q_embeds.size(0))
                q_batch = q_embeds[q_start:q_end].cpu()  # query 移到 CPU 做计算

                s = max_sim(q_batch, d_chunk)  # (query_batch_size, chunk_size)
                chunk_scores_list.append(s)

            chunk_scores = torch.cat(chunk_scores_list, dim=0)  # (num_queries, chunk_size)

            # 累积所有 chunk 的最大分数
            if final_scores is None:
                final_scores = chunk_scores
            else:
                final_scores = torch.cat([final_scores, chunk_scores], dim=1)  # (num_queries, accumulated_docs)

        return final_scores

# --- 评测入口 ---

def main():
    model_path = "/home/moxu/MMRAG/otherExp/colpali/output/colqwen3_vl_2B_lr5e-6_dim2048_stage1/checkpoint-462"
    
    # 实例包装器
    model_wrapper = ColQwen3IndustrialWrapper(model_path)

    # 定义元数据（用于 ResultCache 归档）
    model_wrapper.mteb_model_meta = ModelMeta(
        name="moxu/colqwen3_vl_2B_lr5e-6_dim2048_stage1",
        revision="v1",
        release_date="2026-03-24",
        languages=["eng-Latn"],
        framework=["PyTorch", "ColPali"],
        similarity_fn_name=ScoringFunction.MAX_SIM,
        modalities=["text", "image"],
        model_type=["late-interaction"],
        loader=None,              # 自定义 Wrapper 设为 None 即可
        n_parameters=2_000_000_000, # 2B 模型
        memory_usage_mb=8000.0,
        max_tokens=768,
        embed_dim=2048,
        license="apache-2.0",
        open_weights=True,
        public_training_code=None,
        public_training_data=None,
        use_instructions=False,
        training_datasets=None,
    )

    # Vidore3 v2 完整评估 (8个子任务)
    vidore3_tasks = [
        "Vidore3ComputerScienceRetrieval.v2",
        "Vidore3EnergyRetrieval.v2",
        "Vidore3FinanceEnRetrieval.v2",
        "Vidore3FinanceFrRetrieval.v2",
        "Vidore3HrRetrieval.v2",
        
        "Vidore3PharmaceuticalsRetrieval.v2",
        "Vidore3PhysicsRetrieval.v2",
    ]
    tasks = mteb.get_tasks(tasks=vidore3_tasks, languages=["eng-Latn"])

    # 未指定 cache 时，MTEB 默认写入 ~/.cache/mteb/results/...，容易找不到。
    out_dir = "./mteb_results"
    cache = mteb.ResultCache(cache_path=out_dir)

    model_result = mteb.evaluate(
        model=model_wrapper,
        tasks=tasks,
        cache=cache,
        overwrite_strategy="only-missing",
        
    )

    meta = model_wrapper.mteb_model_meta
    for tr in model_result.task_results:
        p = cache.get_task_result_path(task_name=tr.task_name, model_name=meta)
        print(f"\n📁 任务「{tr.task_name}」结果 JSON: {p.resolve()}")

    print(
        "\n说明：若未改 cache，默认路径为 ~/.cache/mteb/results/<模型名>/<revision>/ 。"
        "本脚本已改为写入项目目录下的 mteb_results/results/。"
    )

if __name__ == "__main__":
    main()