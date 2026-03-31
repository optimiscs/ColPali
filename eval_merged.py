"""
评估 merged 模型的专用脚本
"""
import logging
import os

os.environ["TQDM_DISABLE"] = "1"

for logger_name in ["transformers", "mteb", "huggingface_hub", "accelerate", "tokenizers"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

import torch
import mteb
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType
from mteb.similarity_functions import max_sim
from tqdm.auto import tqdm
from PIL import Image
import torchvision.transforms.functional as F


class SimpleColQwen3Wrapper(AbsEncoder):

    def __init__(self, model_path: str, device: str | None = None, **kwargs):
        from colpali_engine.models.qwen3.colqwen3_vl_embedding import (
            ColQwen3VLEmbedding,
            ColQwen3VLEmbeddingProcessor,
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading: {model_path}")

        self.model = ColQwen3VLEmbedding.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=self.device, **kwargs
        )
        self.model.eval()

        self.processor = ColQwen3VLEmbeddingProcessor.from_pretrained(
            model_path, max_num_visual_tokens=1280
        )

    @torch.no_grad()
    def encode(self, inputs, *, task_metadata, hf_split, hf_subset, prompt_type=None, **kwargs):
        return self.get_fused_embeddings(inputs, prompt_type=prompt_type, **kwargs)

    def get_fused_embeddings(self, dataloader, prompt_type=None, show_progress_bar=True, **kwargs):
        all_embeds = []

        for batch in tqdm(dataloader, disable=not show_progress_bar, desc="Encoding"):
            is_query = (prompt_type == PromptType.query) or ("image" not in batch)

            if is_query:
                texts = batch["text"]
                if not isinstance(texts, (list, tuple)):
                    texts = [texts]
                feat = self.processor.process_queries(texts=list(texts))
            else:
                imgs = [b if isinstance(b, Image.Image) else F.to_pil_image(b.to(self.device)) for b in batch["image"]]
                feat = self.processor.process_images([img.convert("RGB") for img in imgs])

            feat = {k: v.to(self.device) for k, v in feat.items()}
            if hasattr(self.model, "rope_deltas"):
                self.model.rope_deltas = None
            outs = self.model(**feat)
            all_embeds.extend(outs.detach().cpu().float())

        return torch.nn.utils.rnn.pad_sequence(all_embeds, batch_first=True, padding_value=0)

    def similarity(self, q_embeds: torch.Tensor, d_embeds: torch.Tensor) -> torch.Tensor:
        q_embeds = q_embeds.to(self.device)
        d_embeds = d_embeds.to(self.device)
        results = []
        for i in range(0, q_embeds.size(0), 8):
            results.append(max_sim(q_embeds[i : i + 8], d_embeds).cpu())
        return torch.cat(results, dim=0)


def main():
    model_path = "/home/moxu/MMRAG/otherExp/colpali/merged_model"
    model_wrapper = SimpleColQwen3Wrapper(model_path)

    model_wrapper.mteb_model_meta = ModelMeta(
        name="moxu/ColQwen3-VL-Embedding-2B-merged",
        revision="v1",
        release_date="2026-03-24",
        languages=["eng-Latn"],
        framework=["PyTorch", "ColPali"],
        similarity_fn_name=ScoringFunction.MAX_SIM,
        modalities=["text", "image"],
        model_type=["late-interaction"],
        loader=None,
        n_parameters=2_000_000_000,
        memory_usage_mb=8000.0,
        max_tokens=768,
        embed_dim=768,
        license="apache-2.0",
        open_weights=True,
    )

    tasks = mteb.get_tasks(tasks=["Vidore3EnergyRetrieval.v2"], languages=["eng-Latn"])
    cache = mteb.ResultCache(cache_path="./mteb_results")

    model_result = mteb.evaluate(
        model=model_wrapper,
        tasks=tasks,
        cache=cache,
        overwrite_strategy="only-missing",
    )

    for tr in model_result.task_results:
        p = cache.get_task_result_path(task_name=tr.task_name, model_name=model_wrapper.mteb_model_meta)
        print(f"\nResult: {p.resolve()}")


if __name__ == "__main__":
    main()
