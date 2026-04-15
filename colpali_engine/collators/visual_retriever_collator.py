import random
from typing import Any, Dict, List, Union

import torch
from PIL.Image import Image

from colpali_engine.data.dataset import ColPaliEngineDataset
from colpali_engine.models.paligemma import ColPaliProcessor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor

N_AUGMENTATION_TOKENS = 10


def prefix_keys(data: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """
    Prefix all keys in a dictionary with the given prefix.
    """
    return {f"{prefix}{k}": v for k, v in data.items()}


class VisualRetrieverCollator:
    """
    Collator for training vision retrieval models.
    """

    # Prefixes
    query_prefix = "query_"
    pos_doc_prefix = "doc_"
    neg_doc_prefix = "neg_doc_"

    def __init__(
        self,
        processor: BaseVisualRetrieverProcessor,
        max_length: int = 2048,
        dataset=None,  # For lazy loading of negative images
    ):
        self.processor = processor
        self.max_length = max_length
        self.image_token_id = None
        self.dataset = dataset  # Must have id_to_idx and data attributes

        # If processor is one of the supported types, extract the <image> token id.
        if isinstance(self.processor, (ColPaliProcessor,)):
            if hasattr(self.processor, "image_token_id"):
                token_id = self.processor.image_token_id
            else:
                token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
            self.image_token_id = token_id if token_id is not None and token_id >= 0 else None

        # Force padding to be on the right for ColPaliProcessor.
        if isinstance(self.processor, ColPaliProcessor) and self.processor.tokenizer.padding_side != "right":
            print("Setting padding side to right")
            self.processor.tokenizer.padding_side = "right"

    def _resolve_negative_ids(self, neg_target_ids):
        """Resolve negative IDs to images using dataset's id_to_idx mapping."""
        if not neg_target_ids:
            return None

        resolved_images = []
        skipped = 0
        for neg_id in neg_target_ids:
            if neg_id in self.dataset.id_to_idx:
                neg_idx = self.dataset.id_to_idx[neg_id]
                if neg_idx < len(self.dataset.data):
                    neg_sample = self.dataset.data[neg_idx]
                    img = neg_sample.get(self.dataset.pos_target_column_name)
                    if img is not None:
                        resolved_images.append(img)
                    else:
                        skipped += 1
                else:
                    skipped += 1
            else:
                skipped += 1

        if skipped > 0 and len(resolved_images) == 0:
            print(f"【tiaoshi】WARNING: all {len(neg_target_ids)} negatives skipped! resolved={len(resolved_images)}")
        return resolved_images if resolved_images else None

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        queries: List[Union[None, str, Image]] = []
        pos_targets: List[Union[str, Image]] = []
        neg_targets: List[Union[str, Image]] = []
        neg_ids_list: List[List[str]] = []  # Store negative IDs for resolution

        # Parse the examples, filtering out None (None query samples)
        for example in examples:
            if example is None:
                continue
            assert ColPaliEngineDataset.QUERY_KEY in example, f"Missing {ColPaliEngineDataset.QUERY_KEY} in example."
            query = example[ColPaliEngineDataset.QUERY_KEY]
            if isinstance(query, list):
                query = query[0]  # Take first query if list
            queries.append(query)

            assert ColPaliEngineDataset.POS_TARGET_KEY in example, (
                f"Missing {ColPaliEngineDataset.POS_TARGET_KEY} in example."
            )
            pos_tgt = example[ColPaliEngineDataset.POS_TARGET_KEY]
            if isinstance(pos_tgt, list):
                pos_tgt = pos_tgt[0]  # Take first pos target
            pos_targets.append(pos_tgt)

            # Collect negative IDs for later resolution
            neg_tgt_ids = example.get(ColPaliEngineDataset.NEG_TARGET_IDS_KEY, None)
            neg_ids_list.append(neg_tgt_ids if neg_tgt_ids else [])

        # Resolve negative IDs to images (if dataset is available)
        # Only include samples that have actual negatives (matching original behavior)
        neg_targets = []
        for neg_ids in neg_ids_list:
            if neg_ids and self.dataset is not None:
                resolved = self._resolve_negative_ids(neg_ids)
                if resolved:
                    neg_targets.append(resolved)

        # Ensure all queries are strings or images.
        assert all(isinstance(q, str) for q in queries), (
            "All queries must be strings, this collator does not support images in queries."
        )

        # Process queries.
        queries = [
            self.processor.query_prefix + q + self.processor.query_augmentation_token * N_AUGMENTATION_TOKENS
            for q in queries
        ]
        batch_query = self.auto_collate(queries, key_prefix=self.query_prefix)

        # Process targets.
        batch_pos_target = self.auto_collate(pos_targets, key_prefix=self.pos_doc_prefix)
        batch_neg_target = self.auto_collate(neg_targets, key_prefix=self.neg_doc_prefix) if neg_targets else {}

        return {
            **batch_query,
            **batch_pos_target,
            **batch_neg_target,
        }

    def auto_collate(self, batch: List[Union[str, Image]], key_prefix: str = "") -> Dict[str, Any]:
        """Automatically collate a batch of documents."""
        # Convert Document objects to their underlying data.
        # if type is mixed across the batch, raise an error.
        all_types = set(type(item) for item in batch)
        if str in all_types and Image in all_types:
            raise ValueError(f"Batch contains mixed types: {all_types}. Expected all items to be of the same type.")
        if isinstance(batch[0], str):
            proc_batch = self.processor.process_texts(texts=batch)
        elif isinstance(batch[0], Image):
            proc_batch = self.processor.process_images(images=batch)
        elif isinstance(batch[0], list):
            if isinstance(batch[0][0], str):
                batch_size = len(batch)
                all_texts = [text for texts in batch for text in texts]
                num_negatives = len(all_texts) // batch_size
                proc_batch = self.processor.process_texts(texts=all_texts)
            elif isinstance(batch[0][0], Image):
                batch_size = len(batch)
                all_imgs = [img for imgs in batch for img in imgs]
                num_negatives = len(all_imgs) // batch_size
                proc_batch = self.processor.process_images(images=all_imgs)
            else:
                raise ValueError(f"Unsupported batch type: {type(batch[0][0])}. Expected str or Image.")
            for k, v in proc_batch.items():
                if isinstance(v, torch.Tensor):
                    proc_batch[k] = v.view(batch_size, num_negatives, *v.shape[1:])
        else:
            raise ValueError(f"Unsupported batch type: {type(batch[0])}. Expected str or Image.")
        return prefix_keys(proc_batch, key_prefix)
