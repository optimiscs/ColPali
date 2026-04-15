from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from mteb._requires_package import (
    requires_image_dependencies,
    requires_package,
)
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

from .colpali_models import (
    COLPALI_CITATION,
    COLPALI_TRAINING_DATA,
    ColPaliEngineWrapper,
)
from .qwen3_vl_embedding_models import QWEN3_VL_EMBEDDING_CITATION

logger = logging.getLogger(__name__)


class ColQwen2Wrapper(ColPaliEngineWrapper):
    """Wrapper for ColQwen2 model."""

    def __init__(
        self,
        model_name: str = "vidore/colqwen2-v1.0",
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import ColQwen2, ColQwen2Processor

        super().__init__(
            model_name=model_name,
            model_class=ColQwen2,
            processor_class=ColQwen2Processor,
            revision=revision,
            device=device,
            **kwargs,
        )


class ColQwen2_5Wrapper(ColPaliEngineWrapper):  # noqa: N801
    """Wrapper for ColQwen2.5 model."""

    def __init__(
        self,
        model_name: str = "vidore/colqwen2.5-v0.2",
        revision: str | None = None,
        device: str | None = None,
        attn_implementation: str | None = None,
        **kwargs,
    ):
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
        from transformers.utils.import_utils import is_flash_attn_2_available

        if attn_implementation is None:
            attn_implementation = (
                "flash_attention_2" if is_flash_attn_2_available() else None
            )

        super().__init__(
            model_name=model_name,
            model_class=ColQwen2_5,
            processor_class=ColQwen2_5_Processor,
            revision=revision,
            device=device,
            **kwargs,
        )


class ColQwen3_5Wrapper(AbsEncoder):  # noqa: N801
    """Wrapper for ColQwen3.5 models (colpali_engine)."""

    def __init__(
        self,
        model_name: str = "athrael-soju/colqwen3.5-4.5B-v3",
        revision: str | None = None,
        device: str | None = None,
        **kwargs,
    ):
        requires_image_dependencies()
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ColQwen3_5.from_pretrained(
            model_name,
            device_map=self.device,
            adapter_kwargs={"revision": revision},
            **kwargs,
        )
        self.model.eval()

        self.processor = ColQwen3_5Processor.from_pretrained(model_name)

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        if (
            "text" not in inputs.dataset.features
            and "image" not in inputs.dataset.features
        ):
            raise ValueError("No text or image features found in inputs.")
        return self.get_fused_embeddings(inputs, **kwargs)

    def _encode_inputs(self, encoded_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        # Clear stale rope_deltas cache to avoid shape mismatches across batches
        if hasattr(self.model, "rope_deltas"):
            self.model.rope_deltas = None
        # ColQwen3_5.forward returns the projection tensor directly (not a named tuple)
        return self.model(**encoded_inputs)

    def get_fused_embeddings(
        self,
        image_texts_pairs: DataLoader[BatchedInput] | None = None,
        batch_size: int = 32,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ):
        import torchvision.transforms.functional as F
        from PIL import Image

        contains_image = "image" in image_texts_pairs.dataset.features
        contains_text = "text" in image_texts_pairs.dataset.features
        contains_both = contains_image and contains_text

        if contains_both:
            progress_desc = "Encoding images+texts"
        elif contains_image:
            progress_desc = "Encoding images"
        elif contains_text:
            progress_desc = "Encoding texts"
        else:
            raise ValueError("No text or image features found in inputs.")

        all_embeds: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in tqdm(
                image_texts_pairs,
                disable=not show_progress_bar,
                desc=progress_desc,
            ):
                if contains_image:
                    imgs = [
                        F.to_pil_image(b.to(self.device))
                        if not isinstance(b, Image.Image)
                        else b
                        for b in batch["image"]
                    ]
                else:
                    imgs = None
                if contains_text:
                    texts = batch["text"]
                else:
                    texts = None
                if contains_both:
                    assert len(imgs) == len(texts), (
                        f"The number of texts and images must have the same length, got {len(imgs)} and {len(texts)}"
                    )

                if contains_image:
                    imgs = [img.convert("RGB") for img in imgs]
                    inputs = self.processor.process_images(imgs)
                else:
                    inputs = self.processor.process_queries(texts)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outs = self._encode_inputs(inputs)
                all_embeds.extend(outs.cpu().to(torch.float32))

        padded = torch.nn.utils.rnn.pad_sequence(
            all_embeds, batch_first=True, padding_value=0
        )
        return padded

    def similarity(self, a, b):
        a = [torch.as_tensor(x) for x in a]
        b = [torch.as_tensor(x) for x in b]
        return self.processor.score_multi_vector(a, b, device=self.device)


class ColQwen3VLEmbeddingWrapper(AbsEncoder):
    """Wrapper for Qwen3-VL-Embedding via colpali_engine (late-interaction / MaxSim).

    Registry `ModelMeta.name` is suffixed with ``-colpali`` to avoid clashing with the
    dense ``qwen3_vl_embedding_2b`` entry; weights load from ``hub_model_id`` (default
    ``Qwen/Qwen3-VL-Embedding-2B``).

    Optional ``peft_adapter_path`` loads LoRA from a training checkpoint (directory with
    ``adapter_config.json``) or a merged full model (no adapter config).

    Loading matches ``ColQwen3_5Wrapper``: single ``device`` string and ``device_map`` thereof.
    """

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        max_num_visual_tokens: int | None = None,
        hub_model_id: str | None = None,
        peft_adapter_path: str | None = None,
        similarity_use_max_sim: bool = False,
        **kwargs: Any,
    ):
        requires_image_dependencies()
        requires_package(
            self, "colpali_engine", model_name, "pip install mteb[colpali_engine]"
        )
        from colpali_engine.models import ColQwen3VLEmbedding, ColQwen3VLEmbeddingProcessor

        self.similarity_use_max_sim = similarity_use_max_sim

        primary = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = primary
        weights_id = hub_model_id or model_name

        fk = {
            k: v
            for k, v in kwargs.items()
            if k not in ("device_map", "max_memory", "torch_dtype")
        }
        load_kw: dict[str, Any] = {
            **fk,
            "torch_dtype": kwargs.get("torch_dtype", torch.bfloat16),
            "device_map": primary,
        }

        if peft_adapter_path is None:
            self.model = ColQwen3VLEmbedding.from_pretrained(
                weights_id,
                adapter_kwargs={"revision": revision},
                **load_kw,
            )
            proc_source = weights_id
        elif os.path.isfile(os.path.join(peft_adapter_path, "adapter_config.json")):
            requires_package(self, "peft", peft_adapter_path, "pip install peft")
            from peft import PeftModel

            print(f"📦 底座: {weights_id}")
            base = ColQwen3VLEmbedding.from_pretrained(
                weights_id,
                adapter_kwargs={"revision": revision},
                **load_kw,
            )
            print(f"🔗 LoRA: {peft_adapter_path}")
            self.model = PeftModel.from_pretrained(base, peft_adapter_path).merge_and_unload()
            self.model = self.model.to(primary)
            proc_source = weights_id
        else:
            print(f"📦 Merged 权重: {peft_adapter_path}")
            self.model = ColQwen3VLEmbedding.from_pretrained(
                peft_adapter_path,
                adapter_kwargs={"revision": revision},
                **load_kw,
            )
            proc_source = peft_adapter_path

        self.model.eval()

        proc_kw: dict[str, Any] = {}
        if revision is not None:
            proc_kw["revision"] = revision
        if max_num_visual_tokens is not None:
            proc_kw["max_num_visual_tokens"] = max_num_visual_tokens
        self.processor = ColQwen3VLEmbeddingProcessor.from_pretrained(proc_source, **proc_kw)

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        if (
            "text" not in inputs.dataset.features
            and "image" not in inputs.dataset.features
        ):
            raise ValueError("No text or image features found in inputs.")
        # prompt_type 由 MTEB 传入；路由与 ColQwen3_5 一致，按 dataset features（检索：query 常仅 text、corpus 仅 image）
        return self.get_fused_embeddings(inputs, **kwargs)

    def _encode_inputs(self, encoded_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if hasattr(self.model, "rope_deltas"):
            self.model.rope_deltas = None
        return self.model(**encoded_inputs)

    def get_fused_embeddings(
        self,
        image_texts_pairs: DataLoader[BatchedInput] | None = None,
        batch_size: int = 32,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ):
        import torchvision.transforms.functional as F
        from PIL import Image

        kwargs.pop("prompt_type", None)

        contains_image = "image" in image_texts_pairs.dataset.features
        contains_text = "text" in image_texts_pairs.dataset.features
        contains_both = contains_image and contains_text

        if contains_both:
            progress_desc = "Encoding images+texts"
        elif contains_image:
            progress_desc = "Encoding images"
        elif contains_text:
            progress_desc = "Encoding texts"
        else:
            raise ValueError("No text or image features found in inputs.")

        all_embeds: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in tqdm(
                image_texts_pairs,
                disable=not show_progress_bar,
                desc=progress_desc,
            ):
                if contains_image:
                    imgs = [
                        F.to_pil_image(b.to(self.device))
                        if not isinstance(b, Image.Image)
                        else b
                        for b in batch["image"]
                    ]
                    imgs = [img.convert("RGB") for img in imgs]
                else:
                    imgs = None
                if contains_text:
                    texts = batch["text"]
                else:
                    texts = None
                if contains_both:
                    assert imgs is not None and texts is not None
                    assert len(imgs) == len(texts), (
                        f"The number of texts and images must have the same length, got {len(imgs)} and {len(texts)}"
                    )

                if contains_image:
                    inputs = self.processor.process_images(imgs)
                else:
                    if not isinstance(texts, (list, tuple)):
                        texts = [texts]
                    inputs = self.processor.process_queries(texts=texts)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outs = self._encode_inputs(inputs)
                all_embeds.extend(outs.cpu().to(torch.float32))

        return torch.nn.utils.rnn.pad_sequence(
            all_embeds, batch_first=True, padding_value=0
        )

    def similarity(self, a, b):
        if self.similarity_use_max_sim:
            from mteb.similarity_functions import max_sim

            q_embeds = torch.as_tensor(a) if not isinstance(a, torch.Tensor) else a
            d_embeds = torch.as_tensor(b) if not isinstance(b, torch.Tensor) else b
            corpus_chunk_size = 200
            query_step = 4
            num_docs = d_embeds.size(0)
            final_scores = None
            for chunk_start in range(0, num_docs, corpus_chunk_size):
                chunk_end = min(chunk_start + corpus_chunk_size, num_docs)
                d_chunk = d_embeds[chunk_start:chunk_end]
                chunk_scores_list = []
                for q_start in range(0, q_embeds.size(0), query_step):
                    q_end = min(q_start + query_step, q_embeds.size(0))
                    q_batch = q_embeds[q_start:q_end]
                    chunk_scores_list.append(max_sim(q_batch, d_chunk))
                chunk_scores = torch.cat(chunk_scores_list, dim=0)
                if final_scores is None:
                    final_scores = chunk_scores
                else:
                    final_scores = torch.cat([final_scores, chunk_scores], dim=1)
            return final_scores

        a = [torch.as_tensor(x) for x in a]
        b = [torch.as_tensor(x) for x in b]
        return self.processor.score_multi_vector(a, b, device=self.device)

    def similarity_with_attribution(self, query_embeddings, doc_embeddings, top_k=None):
        """Compute similarity with token-level attribution.

        Args:
            query_embeddings: List of query embedding tensors or a single tensor.
            doc_embeddings: List of doc embedding tensors or a single tensor.
            top_k: If specified, only use top_k query tokens by contribution.

        Returns:
            Tuple of (final_scores, token_contributions, max_doc_indices).
            - final_scores: (num_queries, num_docs)
            - token_contributions: (num_queries, num_docs, num_query_tokens)
            - max_doc_indices: (num_queries, num_docs, num_query_tokens)
        """
        from mteb.similarity_functions import max_sim_attribution

        q_embeds = torch.as_tensor(query_embeddings) if not isinstance(query_embeddings, torch.Tensor) else query_embeddings
        d_embeds = torch.as_tensor(doc_embeddings) if not isinstance(doc_embeddings, torch.Tensor) else doc_embeddings

        corpus_chunk_size = 200
        query_step = 4

        num_docs = d_embeds.size(0)
        num_queries = q_embeds.size(0)

        final_scores = None
        all_token_contribs = None
        all_max_idxs = None

        for chunk_start in range(0, num_docs, corpus_chunk_size):
            chunk_end = min(chunk_start + corpus_chunk_size, num_docs)
            d_chunk = d_embeds[chunk_start:chunk_end]

            chunk_scores_list = []
            chunk_contribs_list = []
            chunk_idxs_list = []

            for q_start in range(0, num_queries, query_step):
                q_end = min(q_start + query_step, num_queries)
                q_batch = q_embeds[q_start:q_end]

                scores, contribs, max_idxs = max_sim_attribution(q_batch, d_chunk)

                chunk_scores_list.append(scores)
                chunk_contribs_list.append(contribs)
                chunk_idxs_list.append(max_idxs)

            chunk_scores = torch.cat(chunk_scores_list, dim=0)
            chunk_contribs = torch.cat(chunk_contribs_list, dim=0)
            chunk_idxs = torch.cat(chunk_idxs_list, dim=0)

            if final_scores is None:
                final_scores = chunk_scores
                all_token_contribs = chunk_contribs
                all_max_idxs = chunk_idxs
            else:
                final_scores = torch.cat([final_scores, chunk_scores], dim=1)
                all_token_contribs = torch.cat([all_token_contribs, chunk_contribs], dim=1)
                all_max_idxs = torch.cat([all_max_idxs, chunk_idxs], dim=1)

        if top_k is not None:
            top_k = min(top_k, all_token_contribs.shape[-1])
            top_contribs, top_indices = torch.topk(all_token_contribs, k=top_k, dim=-1)
            final_scores = top_contribs.sum(dim=-1)

            batch_size, num_docs, num_query_tokens = all_token_contribs.shape
            new_contribs = torch.zeros_like(all_token_contribs)
            new_idxs = torch.zeros_like(all_max_idxs, dtype=torch.long)

            for i in range(batch_size):
                for j in range(num_docs):
                    for t, idx in enumerate(top_indices[i, j]):
                        new_contribs[i, j, t] = all_token_contribs[i, j, idx]
                        new_idxs[i, j, t] = all_max_idxs[i, j, idx]

            all_token_contribs = new_contribs
            all_max_idxs = new_idxs

        return final_scores, all_token_contribs, all_max_idxs


class ColQwen3Wrapper(AbsEncoder):
    """Wrapper for the ColQwen3 vision-language retrieval model."""

    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None = None,
        device: str | None = None,
        dtype: torch.dtype | str | None = torch.bfloat16,
        **kwargs: Any,
    ):
        requires_image_dependencies()
        requires_package(self, "transformers", model_name, "pip install mteb[colqwen3]")
        from transformers import AutoModel, AutoProcessor

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            dtype=dtype,
            trust_remote_code=True,
            **kwargs,
        ).to(self.device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
            max_num_visual_tokens=1280,
        )

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        if (
            "text" not in inputs.dataset.features
            and "image" not in inputs.dataset.features
        ):
            raise ValueError("No text or image features found in inputs.")
        return self.get_fused_embeddings(inputs, **kwargs)

    def _encode_inputs(self, encoded_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.model(**encoded_inputs)
        # Avoid boolean casting of tensors when checking for custom attributes.
        embeddings = getattr(outputs, "embeddings", None)
        if embeddings is None:
            embeddings = outputs[0]
        return embeddings

    def get_fused_embeddings(
        self,
        image_texts_pairs: DataLoader[BatchedInput] | None = None,
        batch_size: int = 32,
        show_progress_bar: bool = True,
        fusion_mode="concat",
        **kwargs: Any,
    ):
        import torchvision.transforms.functional as F
        from PIL import Image

        contains_image = "image" in image_texts_pairs.dataset.features
        contains_text = "text" in image_texts_pairs.dataset.features
        contains_both = contains_image and contains_text

        if contains_both:
            progress_desc = "Encoding images+texts"
        elif contains_image:
            progress_desc = "Encoding images"
        elif contains_text:
            progress_desc = "Encoding texts"
        else:
            raise ValueError("No text or image features found in inputs.")

        all_embeds: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in tqdm(
                image_texts_pairs,
                disable=not show_progress_bar,
                desc=progress_desc,
            ):
                if contains_image:
                    imgs = [
                        F.to_pil_image(b.to(self.device))
                        if not isinstance(b, Image.Image)
                        else b
                        for b in batch["image"]
                    ]
                else:
                    imgs = None
                if contains_text:
                    texts = batch["text"]
                else:
                    texts = None
                if contains_both:
                    assert len(imgs) == len(texts), (
                        f"The number of texts and images must have the same length, got {len(imgs)} and {len(texts)}"
                    )

                inputs = self.processor(images=imgs, text=texts)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outs = self._encode_inputs(inputs)
                all_embeds.extend(outs.cpu().to(torch.float32))

        padded = torch.nn.utils.rnn.pad_sequence(
            all_embeds, batch_first=True, padding_value=0
        )
        return padded

    def similarity(self, a, b):
        return self.processor.score_multi_vector(a, b, device=self.device)


colqwen2 = ModelMeta(
    loader=ColQwen2Wrapper,
    loader_kwargs=dict(
        torch_dtype=torch.float16,
    ),
    name="vidore/colqwen2-v1.0",
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    revision="530094e83a40ca4edcb5c9e5ddfa61a4b5ea0d2f",
    release_date="2025-11-03",
    modalities=["image", "text"],
    n_parameters=2_210_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=7200,
    max_tokens=32768,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali", "safetensors"],
    reference="https://huggingface.co/vidore/colqwen2-v1.0",
    similarity_fn_name="MaxSim",
    use_instructions=True,
    training_datasets=COLPALI_TRAINING_DATA,
    citation=COLPALI_CITATION,
)

colqwen2_5 = ModelMeta(
    loader=ColQwen2_5Wrapper,
    loader_kwargs=dict(
        torch_dtype=torch.float16,
    ),
    name="vidore/colqwen2.5-v0.2",
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    revision="6f6fcdfd1a114dfe365f529701b33d66b9349014",
    release_date="2025-01-31",
    modalities=["image", "text"],
    n_parameters=3_000_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=7200,
    max_tokens=128000,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali", "safetensors"],
    reference="https://huggingface.co/vidore/colqwen2.5-v0.2",
    similarity_fn_name="MaxSim",
    use_instructions=True,
    training_datasets=COLPALI_TRAINING_DATA,
    citation=COLPALI_CITATION,
)

TOMORO_TRAINING_DATA = {
    "VDRMultilingualRetrieval",
    # from https://huggingface.co/datasets/vidore/colpali_train_set
    "VidoreDocVQARetrieval",
    "VidoreInfoVQARetrieval",
    "VidoreTatdqaRetrieval",
    "VidoreArxivQARetrieval",
    "VisRAG-Ret-Train-Synthetic-data",
    "VisRAG-Ret-Train-In-domain-data",
}

TOMORO_CITATION = """
@misc{huang2025tomoro_colqwen3_embed,
  title={TomoroAI/tomoro-colqwen3-embed},
  author={Xin Huang and Kye Min Tan and Albert Phelps},
  year={2025},
  url={https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-8b}
}
"""

colqwen3_8b = ModelMeta(
    loader=ColQwen3Wrapper,
    name="TomoroAI/tomoro-colqwen3-embed-8b",
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    revision="0b9fe28142910e209bbac15b1efe85507c27644f",
    release_date="2025-11-26",
    modalities=["image", "text"],
    n_parameters=8_000_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=16724,
    max_tokens=262144,
    embed_dim=320,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-8b",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=True,
    training_datasets=TOMORO_TRAINING_DATA,
    citation=TOMORO_CITATION,
)

colqwen3_4b = ModelMeta(
    loader=ColQwen3Wrapper,
    name="TomoroAI/tomoro-colqwen3-embed-4b",
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    revision="6a32fb68598730bf5620fbf18d832c784235c59c",
    release_date="2025-11-26",
    modalities=["image", "text"],
    n_parameters=4_000_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=8466,
    max_tokens=262144,
    embed_dim=320,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-4b",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=True,
    training_datasets=TOMORO_TRAINING_DATA,
    citation=TOMORO_CITATION,
)


COLNOMIC_CITATION = """
@misc{nomicembedmultimodal2025,
  title={Nomic Embed Multimodal: Interleaved Text, Image, and Screenshots for Visual Document Retrieval},
  author={Nomic Team},
  year={2025},
  publisher={Nomic AI},
  url={https://nomic.ai/blog/posts/nomic-embed-multimodal}
}"""

COLNOMIC_TRAINING_DATA = {"VDRMultilingual"} | COLPALI_TRAINING_DATA
COLNOMIC_LANGUAGES = [
    "deu-Latn",  # German
    "spa-Latn",  # Spanish
    "eng-Latn",  # English
    "fra-Latn",  # French
    "ita-Latn",  # Italian
]

colnomic_3b = ModelMeta(
    loader=ColQwen2_5Wrapper,
    loader_kwargs=dict(
        torch_dtype=torch.float16, attn_implementation="flash_attention_2"
    ),
    name="nomic-ai/colnomic-embed-multimodal-3b",
    model_type=["late-interaction"],
    languages=COLNOMIC_LANGUAGES,
    revision="86627b4a9b0cade577851a70afa469084f9863a4",
    release_date="2025-03-31",
    modalities=["image", "text"],
    n_parameters=3_000_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=7200,
    max_tokens=128000,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/nomic-ai/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali", "safetensors"],
    reference="https://huggingface.co/nomic-ai/colnomic-embed-multimodal-3b",
    similarity_fn_name="MaxSim",
    use_instructions=True,
    training_datasets=COLNOMIC_TRAINING_DATA,
    citation=COLNOMIC_CITATION,
)

colnomic_7b = ModelMeta(
    loader=ColQwen2_5Wrapper,
    loader_kwargs=dict(
        torch_dtype=torch.float16,
    ),
    name="nomic-ai/colnomic-embed-multimodal-7b",
    model_type=["late-interaction"],
    languages=COLNOMIC_LANGUAGES,
    revision="09dbc9502b66605d5be56d2226019b49c9fd3293",
    release_date="2025-03-31",
    modalities=["image", "text"],
    n_parameters=7_000_000_000,
    memory_usage_mb=14400,
    max_tokens=128000,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/nomic-ai/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali"],
    reference="https://huggingface.co/nomic-ai/colnomic-embed-multimodal-7b",
    similarity_fn_name="MaxSim",
    use_instructions=True,
    training_datasets=COLNOMIC_TRAINING_DATA,
    citation=COLNOMIC_CITATION,
)


EVOQWEN_TRAINING_DATA = {
    # "colpali_train_set",
    "VidoreDocVQARetrieval",
    "VidoreInfoVQARetrieval",
    "VidoreTatdqaRetrieval",
    "VidoreArxivQARetrieval",
    "VisRAG-Ret-Train-Synthetic-data",
    "VisRAG-Ret-Train-In-domain-data",
}

evoqwen25_vl_retriever_3b_v1 = ModelMeta(
    loader=ColQwen2_5Wrapper,
    loader_kwargs=dict(
        torch_dtype=torch.float16, attn_implementation="flash_attention_2"
    ),
    name="ApsaraStackMaaS/EvoQwen2.5-VL-Retriever-3B-v1",
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    revision="aeacaa2775f2758d82721eb1cf2f5daf1a392da9",
    release_date="2025-11-04",
    modalities=["image", "text"],
    n_parameters=3_000_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=7200,
    max_tokens=128000,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali", "safetensors"],
    reference="https://huggingface.co/ApsaraStackMaaS/EvoQwen2.5-VL-Retriever-3B-v1",
    similarity_fn_name="MaxSim",
    use_instructions=True,
    training_datasets=EVOQWEN_TRAINING_DATA,
)

evoqwen25_vl_retriever_7b_v1 = ModelMeta(
    loader=ColQwen2_5Wrapper,
    loader_kwargs=dict(
        torch_dtype=torch.float16, attn_implementation="flash_attention_2"
    ),
    name="ApsaraStackMaaS/EvoQwen2.5-VL-Retriever-7B-v1",
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    revision="8952ac6ee0e7de2e9211b165921518caf9202110",
    release_date="2025-11-04",
    modalities=["image", "text"],
    n_parameters=7_000_000_000,
    n_embedding_parameters=None,
    memory_usage_mb=14400,
    max_tokens=128000,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/illuin-tech/colpali",
    public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
    framework=["ColPali", "safetensors"],
    reference="https://huggingface.co/ApsaraStackMaaS/EvoQwen2.5-VL-Retriever-7B-v1",
    similarity_fn_name="MaxSim",
    use_instructions=True,
    training_datasets=EVOQWEN_TRAINING_DATA,
)


COLQWEN35_V3_TRAINING_DATA = {
    # from https://huggingface.co/datasets/vidore/colpali_train_set
    "VidoreDocVQARetrieval",
    "VidoreInfoVQARetrieval",
    "VidoreTatdqaRetrieval",
    "VidoreArxivQARetrieval",
    # from https://huggingface.co/datasets/openbmb/VisRAG-Ret-Train-Synthetic-data
    "VisRAG-Ret-Train-Synthetic-data",
    # from https://huggingface.co/datasets/openbmb/VisRAG-Ret-Train-In-domain-data
    "VisRAG-Ret-Train-In-domain-data",
    # from https://huggingface.co/datasets/llamaindex/vdr-multilingual-train
    "VDRMultilingualRetrieval",
    # from https://huggingface.co/datasets/Metric-AI/tabfquad_train_set
    "VidoreTabfquadRetrieval",
}

colqwen3_5_v3 = ModelMeta(
    loader=ColQwen3_5Wrapper,
    loader_kwargs=dict(
        torch_dtype=torch.bfloat16,
    ),
    name="athrael-soju/colqwen3.5-4.5B-v3",
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    revision="4ad8f151e39bce3adcf88e0bdd72e724c7606638",
    release_date="2026-03-15",
    modalities=["image", "text"],
    n_parameters=4_600_000_000,
    n_embedding_parameters=635_699_200,
    memory_usage_mb=8660,
    max_tokens=262144,
    embed_dim=128,
    license="apache-2.0",
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "ColPali", "safetensors"],
    reference="https://huggingface.co/athrael-soju/colqwen3.5-4.5B-v3",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    use_instructions=False,
    training_datasets=COLQWEN35_V3_TRAINING_DATA,
)

# Late-interaction (colpali_engine) weights from Qwen/Qwen3-VL-Embedding-2B; distinct name from dense qwen3_vl_embedding_2b.
colqwen3_vl_embedding_2b_colpali = ModelMeta(
    loader=ColQwen3VLEmbeddingWrapper,
    loader_kwargs=dict(
        hub_model_id="Qwen/Qwen3-VL-Embedding-2B",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        use_cache=False,
        similarity_use_max_sim=True,
    ),
    name="Qwen/Qwen3-VL-Embedding-2B-colpali",
    model_type=["late-interaction"],
    languages=["eng-Latn"],
    open_weights=True,
    revision="2a50926d213628c727f38025982a76f655673f54",
    release_date="2026-01-08",
    modalities=["image", "text"],
    n_parameters=2_127_532_032,
    n_embedding_parameters=311_164_928,
    memory_usage_mb=7629,
    max_tokens=32768,
    embed_dim=2048,
    license="apache-2.0",
    reference="https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B",
    similarity_fn_name=ScoringFunction.MAX_SIM,
    framework=["PyTorch", "ColPali", "Transformers", "safetensors"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=COLPALI_TRAINING_DATA,
    citation=QWEN3_VL_EMBEDDING_CITATION,
)
