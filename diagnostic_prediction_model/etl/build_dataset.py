import json
v = json.load(open("diagnostic_prediction_model/data/vocabs.json"))
sym2id, fam2id, sub2id, br2id, diag2id = v["symptom2id"], v["family2id"], v["subtype2id"], v["brand2id"], v["diag2id"]

SYM_DIM = len(sym2id)
FAM_DIM, SUB_DIM, BR_DIM = len(fam2id), len(sub2id), len(br2id)
NUM_CLASSES = len(diag2id)

def build_dataset(data_dir: str) -> list[tuple[list[int], dict[str, int], int, str]]:
    """Build a dataset from a directory of JSONL files."""
    dataset = []
    with open(data_dir, "r") as f:
        for line in f:
            issue = json.loads(line)
            symptoms_canon = issue['symptoms_canon']
            equip_canon = issue['equip']
            diag_canon = issue['diag_canon']
            split = issue['split']
            dataset.append((symptoms_canon, equip_canon, diag_canon, split))
    return dataset


def train_smoke_test():
    import json, torch
    from torch.utils.data import DataLoader
    from diagnostic_prediction_model.dataio.dataloader import HVACDataset

    VOCAB_PATH = "diagnostic_prediction_model/data/vocabs.json"
    TRAIN_PATH = "diagnostic_prediction_model/data/train.jsonl"
    VAL_PATH   = "diagnostic_prediction_model/data/val.jsonl"
    TEST_PATH  = "diagnostic_prediction_model/data/test.jsonl"

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