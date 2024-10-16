import json
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import read_fasta
from scipy.ndimage import gaussian_filter1d
import os
from collections import Counter
from dataclasses import dataclass

@dataclass
class EnzymeData:
    id: str
    ph_label: float
    sequence: str
    ec_label: str = None
    cls: int = None
    ph_bias: float = None

    def __post_init__(self):
        self.cls = int(self.ph_label) - 1
        self.pH_bias = self.ph_label - int(self.ph_label)

    @classmethod
    def from_header(cls, header, sequence):
        id_, ph_label, *ec_label = map(str.strip, header.split('|'))
        return cls(id=id_, ph_label=float(ph_label), sequence=sequence, ec_label=ec_label[0] if ec_label else None)


class DatasetBase(Dataset):
    def __init__(self):
        super().__init__()
        self.data = None

    def precompute_bin_inverse_weights(self, ph_labels):
        counts = Counter(int(ph) - 1 for ph in ph_labels)
        mean_weight = sum(1 / v for v in counts.values()) / len(counts)
        return {k: (1 / v) / mean_weight for k, v in counts.items()}

    def precompute_cbw_weight(self, labels, beta=0.9):
        counts = Counter(int(label) - 1 for label in labels)
        effective_num = {k: (1 - beta ** v) / (1 - beta) for k, v in counts.items()}
        total_sum = sum(1 / v for v in effective_num.values())
        return {k: (1 / v) * len(effective_num) / total_sum for k, v in effective_num.items()}

    def precompute_lds_weight(self, labels, kernel_size=5, sigma=1.0):
        labels = [int(label) - 1 for label in labels]
        label_counts = np.bincount(labels, minlength=max(labels) + 1)
        smoothed_counts = gaussian_filter1d(label_counts, sigma=sigma, truncate=(kernel_size - 1) / (2 * sigma))
        effective_density = smoothed_counts / smoothed_counts.sum()
        total_sum = sum(1 / d for d in effective_density)
        return {i: (1 / d) * len(effective_density) / total_sum for i, d in enumerate(effective_density)}

    def precompute_root_csw_weight(self, labels):
        counts = np.bincount([int(label) - 1 for label in labels])
        total_sum = sum(1 / np.sqrt(c) for c in counts)
        return {i: (1 / np.sqrt(c)) * len(counts) / total_sum for i, c in enumerate(counts)}

    def precompute_dmw_weight(self, labels, edge_weight=10, mid_weight=1):
        labels = np.array([int(label) - 1 for label in labels])
        num_classes = 12
        class_weights = np.array([edge_weight / (1 + min(i, num_classes - 1 - i)) for i in range(num_classes)])
        total_sum = class_weights.sum()
        return {i: w * num_classes / total_sum for i, w in enumerate(class_weights)}

    def precompute_weight(self, method):
        ph_labels = [data.ph_label for data in self.data]
        method_map = {
            'bin_inverse': self.precompute_bin_inverse_weights,
            'cbw': self.precompute_cbw_weight,
            'lds': self.precompute_lds_weight,
            'root_csw': self.precompute_root_csw_weight,
            'dmw': self.precompute_dmw_weight
        }
        if method not in method_map:
            raise ValueError(f'Unknown precompute_weight: {method}')
        weights = method_map[method](ph_labels)
        return torch.tensor(list(dict(sorted(weights.items())).values()), dtype=torch.float32)


class DatasetPh(DatasetBase):
    def __init__(self, fasta_file):
        super().__init__()
        headers, sequences = read_fasta(fasta_file)
        self.data = [EnzymeData.from_header(header, seq) for header, seq in zip(headers, sequences)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]




