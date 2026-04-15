from typing import Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import BatchSampler, Dataset


class SingleDatasetBatchSampler(BatchSampler):
    """
    A batch sampler that samples from a single dataset per batch and handles distribution across GPUs.
    Supports filtering to only sample valid indices (e.g., non-None samples).

    Args:
        datasets (List[Dataset]): List of datasets to sample from
        batch_size (int): Global batch size (will be divided across GPUs)
        drop_last (bool): Whether to drop the last incomplete batch
        generator (Optional[torch.Generator]): Random number generator
        valid_indices (Optional[List[List[int]]]): Pre-computed list of valid indices per dataset.
            If provided, only these indices will be sampled.
    """

    def __init__(
        self,
        datasets: List[Dataset],
        global_batch_size: int,
        drop_last: bool = True,
        generator: Optional[torch.Generator] = None,
        valid_indices: Optional[List[List[int]]] = None,
    ):
        self.datasets = datasets
        self.global_batch_size = global_batch_size
        self.drop_last = drop_last
        self.generator = generator or torch.Generator()
        self.initial_seed = self.generator.initial_seed()

        # Calculate dataset sizes and create index mappings
        self.dataset_sizes = [len(dataset) for dataset in datasets]
        #### get start of each dataset #####
        self.cumsum_sizes = np.cumsum([0] + self.dataset_sizes).tolist()
        self.total_size = sum(self.dataset_sizes)

        # Use pre-computed valid indices if provided
        if valid_indices is not None:
            self.valid_indices_per_dataset = valid_indices
            self.valid_sizes = [len(v) for v in valid_indices]
        else:
            self.valid_indices_per_dataset = None
            self.valid_sizes = self.dataset_sizes

        # Create shuffled indices for each dataset (using valid indices if filter provided)
        if self.valid_indices_per_dataset is not None:
            self.indices_per_dataset = [
                self._shuffle_valid_indices(valid_idxs)
                for valid_idxs in self.valid_indices_per_dataset
            ]
            self.max_positions = [(size // self.global_batch_size) * self.global_batch_size for size in self.valid_sizes]
        else:
            self.indices_per_dataset = [
                torch.randperm(size, generator=self.generator).tolist() for size in self.dataset_sizes
            ]
            self.max_positions = [(size // self.global_batch_size) * self.global_batch_size for size in self.dataset_sizes]

        self.current_positions = [0] * len(datasets)

        self.available_datasets = list(range(len(datasets)))
        self._update_available_datasets()

    def _shuffle_valid_indices(self, valid_idxs: List[int]) -> List[int]:
        """Shuffle valid indices using the generator."""
        if len(valid_idxs) == 0:
            return []
        perm = torch.randperm(len(valid_idxs), generator=self.generator).tolist()
        return [valid_idxs[i] for i in perm]

    def _update_available_datasets(self):
        """Update available datasets based on remaining valid samples."""
        self.available_datasets = [
            i for i in range(len(self.datasets))
            if self.current_positions[i] < len(self.indices_per_dataset[i])
        ]

    def __iter__(self) -> Iterator[List[int]]:
        # Reset state
        self.current_positions = [0] * len(self.datasets)
        self._update_available_datasets()

        # Recompute valid indices for fresh epoch (respects new seed from set_epoch)
        if self.valid_indices_per_dataset is not None:
            self.indices_per_dataset = [
                self._shuffle_valid_indices(valid_idxs)
                for valid_idxs in self.valid_indices_per_dataset
            ]

        while self.available_datasets:
            # Build probabilities for available datasets only
            lengths = []
            for i in self.available_datasets:
                remaining = len(self.indices_per_dataset[i]) - self.current_positions[i]
                lengths.append(remaining)

            total_length = sum(lengths)
            if total_length <= 0:
                break  # nothing left to sample

            probs = torch.tensor(lengths, dtype=torch.float) / total_length

            # Pick dataset
            dataset_idx_in_available = torch.multinomial(probs, num_samples=1, generator=self.generator).item()
            dataset_idx = self.available_datasets[dataset_idx_in_available]

            # Fetch batch
            dataset_indices = self.indices_per_dataset[dataset_idx]
            current_pos = self.current_positions[dataset_idx]
            end_pos = current_pos + self.global_batch_size

            if end_pos <= len(dataset_indices):  # Check against actual shuffled list length
                batch_indices = [idx + self.cumsum_sizes[dataset_idx] for idx in dataset_indices[current_pos:end_pos]]
                self.current_positions[dataset_idx] = end_pos

                # Remove if exhausted
                if current_pos >= len(dataset_indices):
                    self.available_datasets.remove(dataset_idx)

                yield batch_indices
            else:
                # Not enough for a full batch
                self.available_datasets.remove(dataset_idx)

    def set_epoch(self, epoch):
        """
        Sets the epoch for this sampler.

        Args:
            epoch (int): Epoch number
        """
        torch_gen = torch.Generator()

        # Set seed based on epoch to ensure different shuffling each epoch
        new_seed = self.initial_seed + epoch
        torch_gen.manual_seed(new_seed)
        self.generator.manual_seed(new_seed)

        # Reshuffle indices for each dataset
        if self.valid_indices_per_dataset is not None:
            self.indices_per_dataset = [
                self._shuffle_valid_indices(valid_idxs)
                for valid_idxs in self.valid_indices_per_dataset
            ]
        else:
            self.indices_per_dataset = [
                torch.randperm(size, generator=torch_gen).tolist() for size in self.dataset_sizes
            ]

    @property
    def batch_size(self) -> int:
        return self.global_batch_size

    def __len__(self) -> int:
        if self.valid_sizes is not None:
            return sum(size // self.global_batch_size for size in self.valid_sizes)
        return sum(size // self.global_batch_size for size in self.dataset_sizes)
