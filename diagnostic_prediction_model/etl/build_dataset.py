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


if __name__ == "__main__":
    print("Use train_smoke.py to test data loading and batching.")