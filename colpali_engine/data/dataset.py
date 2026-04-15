from typing import Any, Dict, List, Optional, Union

from datasets import Dataset as HFDataset
from PIL import Image
from torch.utils.data import Dataset

Document = Union[str, Image.Image]


class Corpus:
    """
    Corpus class for handling retrieving with simple mapping.
    This class is meant to be overridden by the user to handle their own corpus.

    Args:
        corpus_data (List[Dict[str, Any]]): List of dictionaries containing doc data.
        docid_to_idx_mapping (Optional[Dict[str, int]]): Optional mapping from doc IDs to indices.
    """

    def __init__(
        self,
        corpus_data: List[Dict[str, Any]],
        docid_to_idx_mapping: Optional[Dict[str, int]] = None,
        doc_column_name: str = "doc",
    ):
        """
        Initialize the corpus with the provided data.
        """
        self.corpus_data = corpus_data
        self.docid_to_idx_mapping = docid_to_idx_mapping
        self.doc_column_name = doc_column_name

        assert isinstance(
            self.corpus_data,
            (list, Dataset, HFDataset),
        ), "Corpus data must be a map-style dataset"

        assert self.doc_column_name in self.corpus_data[0], f"Corpus data must contain a column {self.doc_column_name}."

    def __len__(self) -> int:
        """
        Return the number of docs in the corpus.

        Returns:
            int: The number of docs in the corpus.
        """
        return len(self.corpus_data)

    def retrieve(self, docid: Any) -> Document:
        """
        Get the corpus row from the given Doc ID.

        Args:
            docid (str): The id of the document.

        Returns:
            Document: The document retrieved from the corpus.
        """
        if self.docid_to_idx_mapping is not None:
            doc_idx = self.docid_to_idx_mapping[docid]
        else:
            doc_idx = docid
        return self.corpus_data[doc_idx][self.doc_column_name]


class ColPaliEngineDataset(Dataset):
    # Output keys
    QUERY_KEY = "query"
    POS_TARGET_KEY = "pos_target"
    NEG_TARGET_KEY = "neg_target"
    NEG_TARGET_IDS_KEY = "neg_target_ids"  # For lazy loading in collator

    def __init__(
        self,
        data: List[Dict[str, Any]],
        corpus: Optional[Corpus] = None,
        query_column_name: str = "query",
        pos_target_column_name: str = "pos_target",
        neg_target_column_name: str = None,
        num_negatives: int = 3,
        id_column_name: str = "id",
        id_to_idx: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize the dataset with the provided data and external document corpus.

        Args:
            data (Dict[str, List[Any]]): A dictionary containing the dataset samples.
            corpus (Optional[Corpus]): An optional external document corpus to retrieve
            documents (images) from.
            id_column_name (str): Column name for sample IDs, used to build id->image mapping.
            id_to_idx (Optional[Dict[str, int]]): Pre-built mapping from sample ID to index.
                If provided, uses this instead of building from data. This is useful when
                the data is a subset but negatives reference samples outside the subset.
        """
        self.data = data
        self.corpus = corpus

        # Column args
        self.query_column_name = query_column_name
        self.pos_target_column_name = pos_target_column_name
        self.neg_target_column_name = neg_target_column_name
        self.id_column_name = id_column_name

        self.num_negatives = num_negatives

        assert isinstance(
            self.data,
            (list, Dataset, HFDataset),
        ), "Data must be a map-style dataset"

        assert self.query_column_name in self.data[0], f"Data must contain the {self.query_column_name} column"
        assert self.pos_target_column_name in self.data[0], f"Data must contain a {self.pos_target_column_name} column"
        if self.neg_target_column_name is not None:
            assert self.neg_target_column_name in self.data[0], (
                f"Data must contain a {self.neg_target_column_name} column"
            )

        # Build id -> index mapping for negative sample resolution (lazy loading)
        # If id_to_idx is provided externally (e.g., from full dataset), use it directly.
        # Otherwise, build from current data.
        # This enables negatives to reference samples outside the current data split.
        if id_to_idx is not None:
            self.id_to_idx = id_to_idx
        else:
            self.id_to_idx = {}
            if self.id_column_name in self.data[0]:
                for idx, sample in enumerate(self.data):
                    sample_id = sample.get(self.id_column_name)
                    if sample_id is not None:
                        self.id_to_idx[sample_id] = idx

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        sample = self.data[idx]
        query = sample[self.query_column_name]
        # Skip None query samples - these are only useful as negative sample sources
        if query is None:
            return None

        if isinstance(query, str):
            query = query.strip()
        else:
            query = str(query).strip() if query else ""

        pos_targets = sample[self.pos_target_column_name]
        if not isinstance(pos_targets, list):
            pos_targets = [pos_targets]

        # Get sample's own negatives - return IDs only (images resolved in collator for efficiency)
        # This avoids repeated image loading when num_workers > 1
        neg_target_ids = None
        if self.neg_target_column_name is not None:
            neg_ids = sample[self.neg_target_column_name]
            if isinstance(neg_ids, list) and len(neg_ids) > 0:
                neg_target_ids = neg_ids

        # If an external document corpus is provided, retrieve the documents from it.
        if self.corpus is not None:
            pos_targets = [self.corpus.retrieve(doc_id) for doc_id in pos_targets]

        return {
            self.QUERY_KEY: query,
            self.POS_TARGET_KEY: pos_targets,
            self.NEG_TARGET_KEY: None,  # No longer loading images here
            self.NEG_TARGET_IDS_KEY: neg_target_ids,  # Pass IDs for collator to resolve
        }

    def take(self, n: int) -> "ColPaliEngineDataset":
        """
        Take the first n samples from the dataset.

        Args:
            n (int): The number of samples to take.

        Returns:
            ColPaliEngineDataset: A new dataset containing the first n samples.
        """
        return self.__class__(
            self.data.take(n),
            self.corpus,
            self.query_column_name,
            self.pos_target_column_name,
            self.neg_target_column_name,
        )
