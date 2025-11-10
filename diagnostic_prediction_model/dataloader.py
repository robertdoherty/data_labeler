"""Dataset utilities for error prediction inputs."""

import json, numpy as np, torch
from torch.utils.data import Dataset, DataLoader


def soft_target(
    diag2id: dict[str, int],
    y_diag: list[tuple[str, float]] | None = None,
    add_other: bool = True,
) -> torch.Tensor:
    """Convert diagnostic labels/weights to a normalized soft target tensor."""

    y = np.zeros(len(diag2id), np.float32)
    tot = 0.0

    for d, w in (y_diag or []):
        i = diag2id.get(d)
        if i is None:
            continue
        w = float(w)
        y[i] += w
        tot += w

    if add_other:
        other_idx = diag2id.get("dx.other_or_unclear")
        if other_idx is not None:
            y[other_idx] += max(0.0, 1.0 - tot)

    if y.sum() > 0:
        y /= y.sum()

    return torch.from_numpy(y)


class HVACDataset(Dataset):
    
    def __init__(self, path: str, v: dict[str, dict[str, int]]):
        with open(path) as fh:
            self.rows = [json.loads(l) for l in fh]
        self.v = v
    
    
    def _multi_hot(self, toks: list[str], map_: dict[str, int]) -> np.ndarray:
        x = np.zeros(len(map_), np.float32)
        for t in toks: 
            i = map_.get(t); 
            if i is not None: x[i] = 1.0
        return x
    
    
    def _one_hot(self, key: str, map_: dict[str, int], unk: str) -> np.ndarray:
        k = key if key in map_ else unk
        x = np.zeros(len(map_), np.float32); x[map_[k]] = 1.0; return x
    
    
    def __getitem__(self, i: int):
        ex = self.rows[i]; eq = ex["equip"]
        x = np.concatenate([
            self._multi_hot(ex["symptoms_canon"], self.v["symptom2id"]),
            self._one_hot(eq.get("family","<unk_family>"),  self.v["family2id"],  "<unk_family>"),
            self._one_hot(eq.get("subtype","<unk_subtype>"), self.v["subtype2id"], "<unk_subtype>"),
            self._one_hot(eq.get("brand","<unk_brand>"),     self.v["brand2id"],   "<unk_brand>")
        ], 0)
        y = soft_target(self.v["diag2id"], ex.get("y_diag"), add_other=True)
        return torch.from_numpy(x), y
    
    def __len__(self) -> int: return len(self.rows)

