"""Smoke test for training data loading and batching."""

import json
import torch
from torch.utils.data import DataLoader
from diagnostic_prediction_model.dataloader import HVACDataset


def train_smoke_test():
    """Load vocabs and datasets, verify batching works correctly."""
    VOCAB_PATH = "diagnostic_prediction_model/etl/data/vocabs.json"
    TRAIN_PATH = "diagnostic_prediction_model/etl/data/train.jsonl"
    VAL_PATH   = "diagnostic_prediction_model/etl/data/val.jsonl"
    TEST_PATH  = "diagnostic_prediction_model/etl/data/test.jsonl"

    # 1) load vocabs
    v = json.load(open(VOCAB_PATH))
    SYM_DIM = len(v["symptom2id"])
    FAM_DIM = len(v["family2id"])
    SUB_DIM = len(v["subtype2id"])
    BR_DIM  = len(v["brand2id"])
    INPUT_DIM = SYM_DIM + FAM_DIM + SUB_DIM + BR_DIM
    NUM_CLASSES = len(v["diag2id"])

    print(f"INPUT_DIM={INPUT_DIM}  NUM_CLASSES={NUM_CLASSES}")

    # 2) datasets
    train_ds = HVACDataset(TRAIN_PATH, v)
    val_ds   = HVACDataset(VAL_PATH,   v)
    test_ds  = HVACDataset(TEST_PATH,  v)

    print(f"rows: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    # 3) dataloaders
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0, pin_memory=False)

    # 4) fetch a batch
    xb, yb = next(iter(train_dl))
    print("xb shape:", xb.shape)  # [B, INPUT_DIM]
    print("yb shape:", yb.shape)  # [B, NUM_CLASSES]
    print("xb dtype:", xb.dtype, "yb dtype:", yb.dtype)


if __name__ == "__main__":
    train_smoke_test()

