

Diagnostic Prediction Model
===========================

This package builds a tabular dataset for HVAC diagnostic prediction and provides utilities to load and train a classifier on it.


Contents
--------
- Overview
- Architecture
- Data locations
- Setup
- Quickstart
- Training
- Evaluation
- Inference
- Repository layout
- Development notes
- Roadmap / TODO


Overview
--------
The system converts labeled HVAC troubleshooting posts into a machine-learnable dataset. Inputs are symptoms and equipment metadata; outputs are diagnostic probabilities.


Architecture
------------
The following describes how the pipeline works today, with pointers to where each step happens.

- Label the data
  - Use LLM/rules to produce `symptoms_canon` (list), `equip` (family/subtype/brand), and `y_diag=[(diag, confidence), …]`.
  - Where: `data_labeler/` (e.g., `data_labeler_orchestrator.py`, `diagnostic_agent/`, `rule_labeler/`)
  - Output (example): `output/2025-11-03/diagnostic_dataset_*.json`

- Canonicalize text
  - Lowercase, strip punctuation, split symptoms on `;`, replace spaces with `_`, dedupe; normalize equipment fields with `<unk_*>` fallbacks.
  - Where: `diagnostic_prediction_model/etl/canon.py` (`canonicalize_fields`, `canonicalize_equip`)

- Split data deterministically
  - Hash `post_id` → `train`/`val`/`test`.
  - Where: `diagnostic_prediction_model/etl/canon.py` (`split_from_id`)

- Build vocab maps (train-only)
  - Map symptoms, equipment (family/subtype/brand), and diagnosis to numeric ids; save `vocabs.json`.
  - Where: `diagnostic_prediction_model/etl/canon.py` (writes to `diagnostic_prediction_model/etl/data/vocabs.json`)

- Encode examples
  - Inputs: multi-hot symptoms + one-hot equipment → `x ∈ R^[B, INPUT_DIM]`.
  - Targets: turn `y_diag` into a soft label distribution `y ∈ R^[B, NUM_CLASSES]` (leftover mass → `dx.other_or_unclear`; normalize to sum 1).
  - Where: `diagnostic_prediction_model/dataloader.py` (`HVACDataset`, `soft_target`)
  - Data files written to: `diagnostic_prediction_model/etl/data/{train,val,test}.jsonl`

- Load with DataLoader
  - Batch into tensors: `x.float(): [B, INPUT_DIM]`, `y.float(): [B, NUM_CLASSES]`.
  - Where: `diagnostic_prediction_model/dataloader.py`, examples in `diagnostic_prediction_model/train_smoke.py` and `diagnostic_prediction_model/test_dataset.ipynb`

- Train a classifier
  - Recommended: small MLP or Wide&Deep; model outputs logits; minimize soft cross-entropy / KLDiv against soft labels.
  - Where: `diagnostic_prediction_model/classifier.py` (placeholder to implement)

- Validate & save
  - Track val loss + top-k/F1; save `model.pt` with `vocabs.json` for inference parity.
  - Where: to be implemented (see Roadmap)


Data locations
--------------
- Raw labeled dataset (example): `output/2025-11-03/diagnostic_dataset_*.json`
- Processed dataset: `diagnostic_prediction_model/etl/data/{train,val,test}.jsonl`
- Vocabularies: `diagnostic_prediction_model/etl/data/vocabs.json`


Setup
-----
Prereqs: Python 3.10+ and PyTorch.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


Quickstart
----------
1) Build processed data from the latest diagnostic dataset:
```bash
python -m diagnostic_prediction_model.etl.canon
```
Or in the notebook: `diagnostic_prediction_model/test_dataset.ipynb` (Cell: “Run ETL Pipeline”).

2) Verify loading and batching:
```bash
python -m diagnostic_prediction_model.train_smoke
```


Training
--------
- Model: fill in `diagnostic_prediction_model/classifier.py` with a small MLP (INPUT_DIM → hidden → NUM_CLASSES).
- Loss: soft cross-entropy or KLDiv with `reduction='batchmean'`.
- Optimizer: Adam, weight decay optional.
- Hyperparameters: [fill in learning rate, batch size, epochs].
- Script/entrypoint: [add `train.py` and link here].


Evaluation
----------
- Metrics: top-1/top-k accuracy, macro/micro F1, calibration (ECE).
- Validation loop: [to be implemented].
- Reporting: [tensorboard or CSV logs] at `diagnostic_prediction_model/runs/…`.


Inference
---------
- Load `model.pt` and `vocabs.json`.
- Preprocess inputs with `canonicalize_fields` / `canonicalize_equip` (reuse ETL logic or a light runtime variant).
- Encode to features (multi-hot + one-hot), run model → diagnostics probabilities.
- Serving: [FastAPI endpoint path], [CLI example], [batch scoring script].


Repository layout
-----------------
- `diagnostic_prediction_model/`
  - `dataloader.py`: `HVACDataset`, `soft_target`
  - `etl/canon.py`: data canonicalization, splitting, vocab building, JSONL writers
  - `etl/data/`: processed `{train,val,test}.jsonl`, `vocabs.json`
  - `train_smoke.py`: quick load/batch sanity check
  - `test_dataset.ipynb`: end-to-end smoke (ETL → load → batch)
  - `classifier.py`: placeholder for model definition/training loop
- `data_labeler/`: LLM/rule agents to produce raw labeled data
- `output/DATE/`: archived raw labeled datasets


Development notes
-----------------
- Features are sparse and concatenated in a single vector: `[symptoms_multi_hot | family_one_hot | subtype_one_hot | brand_one_hot]`.
- Targets are soft labels; any leftover mass goes to `dx.other_or_unclear`.
- Unknown equipment tokens map to `<unk_* >`, ensuring fixed-dim encodings.
- Deterministic splits by `post_id` for reproducibility.


Roadmap / TODO
--------------
- Implement `classifier.py` (model, train/eval loops, checkpointing).
- Add `train.py` entrypoint with arg parsing and logging.
- Add evaluation metrics and reporting.
- Add inference script and/or service.
- Add tests for ETL and dataloader encodings.
- Document hyperparameter recommendations and examples.
