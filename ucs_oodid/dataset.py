from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .windowing import WindowedData


class WindowTorchDataset(Dataset):
    def __init__(self, data: WindowedData):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {
            "x": torch.tensor(self.data.x[idx], dtype=torch.float32),
            "y": torch.tensor(self.data.y[idx], dtype=torch.float32),
            "record_indices": torch.tensor(self.data.record_indices[idx], dtype=torch.long),
            "mask": torch.tensor(self.data.valid_mask[idx], dtype=torch.bool),
        }
        if self.data.group_index is not None:
            item["group_index"] = torch.tensor(self.data.group_index[idx], dtype=torch.long)
        if self.data.record_y is not None:
            item["record_y"] = torch.tensor(self.data.record_y[idx], dtype=torch.float32)
        return item
