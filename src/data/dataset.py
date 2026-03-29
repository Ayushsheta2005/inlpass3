"""
PyTorch Dataset classes for all three tasks.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple


class CipherDataset(Dataset):
    """
    Character-level dataset for cipher ↔ plain mapping (Part 1).

    Each sample is a pair of equal-length integer sequences:
        cipher_ids  – encoded cipher characters
        plain_ids   – encoded plain characters (labels)
    """

    def __init__(self, samples: List[Tuple[List[int], List[int]]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cipher_ids, plain_ids = self.samples[idx]
        return (
            torch.tensor(cipher_ids, dtype=torch.long),
            torch.tensor(plain_ids, dtype=torch.long),
        )


class NWPDataset(Dataset):
    """
    Word-level dataset for Next Word Prediction (Part 2 – SSM).

    Each sample:
        input_ids  – token indices [0 … seq_len-1]
        target_ids – token indices [1 … seq_len] (shifted by 1)
    """

    def __init__(self, samples: List[Tuple[List[int], List[int]]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids, target_ids = self.samples[idx]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )


class MLMDataset(Dataset):
    """
    Word-level dataset for Masked Language Modeling (Part 2 – Bi-LSTM).

    Each sample:
        masked_ids  – token indices with some replaced by <MASK>
        original_ids – ground-truth token indices
        mask_flags   – binary flag: 1 where a mask was applied
    """

    def __init__(
        self,
        samples: List[Tuple[List[int], List[int], List[int]]],
    ):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        masked_ids, original_ids, mask_flags = self.samples[idx]
        return (
            torch.tensor(masked_ids, dtype=torch.long),
            torch.tensor(original_ids, dtype=torch.long),
            torch.tensor(mask_flags, dtype=torch.long),
        )


class NoisyCipherDataset(Dataset):
    """
    Noisy cipher dataset for Part 3.

    Identical structure to CipherDataset but sourced from noisy files.
    """

    def __init__(self, samples: List[Tuple[List[int], List[int]]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cipher_ids, plain_ids = self.samples[idx]
        return (
            torch.tensor(cipher_ids, dtype=torch.long),
            torch.tensor(plain_ids, dtype=torch.long),
        )
