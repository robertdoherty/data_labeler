## data_labeler

Simple tools to label HVAC-related Reddit posts.

### What it does
- Finds posts that are actual breaks.
- Pulls likely solutions from comments for those posts.
- Adds rule-based diagnostic labels. Uses an LLM when rules are unsure.
- (Optional) Reads images to enrich brand/model and symptoms.

### Pipeline at a glance
1. Break labeling: mark posts as BREAK or not, and extract fields from the OP.
2. Solution extraction: reviews comments for only break posts, fills in missing fields for system/symptom, and determines if solution is available
3. Diagnostics: map symptoms to a diagnostic label using rules. If unclear, calls an agent to make a best guess using finite diagnosis
4. Vision enrichment (optional): parse images to add brand/model and visual symptoms.


### Inputs and outputs
- Input: a Reddit dataset like `output/reddit_research_data_YYYY-MM-DD.json`. Note to run a new scrape use: `python "/Users/robertdoherty/Desktop/Playground/research agent/reddit_fetcher.py"`
- Outputs (written under `output/YYYY-MM-DD/`):
  - `break_labeled_posts_*.json`
  - `solutions_*.json`
  - `rule_predictions_*.json`
  - `solutions_with_diagnostics_*.json`
  - `diagnostic_dataset_*.json`
  - `data_labeler_orchestrator.log`

### Run the full pipeline
```bash
python data_labeler_orchestrator.py \
  output/reddit_research_data_2025-10-29.json \
  --output-dir output \
  --subs hvacadvice,HVAC \
  --solution-max-concurrency 3
```

Returns paths to all written files. Logs go to `output/YYYY-MM-DD/data_labeler_orchestrator.log`.

### Run components separately

Break labeler (step 1)
```bash
python break_labeler_agent/break_labeler_agent.py \
  --output-dir output \
  --subs hvacadvice,HVAC \
  --max-concurrency 3 \
  output/reddit_research_data_2025-10-29.json
```

Solution labeler (step 2)
```bash
python solution_labeler_agent/solution_labeler_agent.py \
  --raw output/reddit_research_data_2025-10-29.json \
  --labels output/2025-10-30/break_labeled_posts_2025-10-29.json \
  --output output/2025-10-30/solutions_2025-10-30_09-03-15.json \
  --max-concurrency 3
```

Vision enrichment (optional)
```bash
python vision_enricher_agent/vision_enricher_agent.py \
  --raw output/reddit_research_data_2025-10-29.json \
  --labels output/2025-10-30/break_labeled_posts_2025-10-29.json \
  --output output/2025-10-30/vision_enriched_2025-10-30_09-03-15.json \
  --max-images 3
```

Diagnostic agent (used inside step 3)
- Programmatic use: `data_labeler/diagnostic_agent/agent.py` exposes `predict_diagnostics` and `predict_diagnostics_batch`.
- The orchestrator calls this only when rules are weak or conflicting.

### Folder overview
- `break_labeler_agent/`: labels posts as BREAK vs not. Extracts OP fields. Batch-friendly.
- `solution_labeler_agent/`: builds an index of BREAK posts + comments and finds solutions. Also merges safe comment-based enrichment into labels.
- `diagnostic_agent/`: LLM that predicts up to 2 diagnostic labels given symptoms and small examples.
- `rule_labeler/`: rules and helpers for diagnostics. Contains:
  - `meta/diagnostics_v1.json`: allowed diagnostic labels.
  - `meta/rules_v1.json`: ordered rules for mapping symptoms to labels.
  - `gold/golden_examples.json`: small examples set for prompts.
  - `scripts/make_error_prediction.py`: utilities to normalize symptoms, apply rules, and build a training JSONL.
  - `scripts/make_error_prediction_config.json`: normalizer config (aliases, units, phrases).
- `vision_enricher_agent/`: multimodal prompt to extract brand/model and visual symptoms from images. Adds results under `record["vision"]`.
- `tools/`: helpers. `break_data_loader.py` is a placeholder.
- `data_labeler_orchestrator.py`: runs steps 1→3 in one call and writes outputs.

### Configuration
- See `config.py` at repo root for defaults, like:
  - `DEFAULT_BREAK_MAX_CONCURRENCY`
  - `DEFAULT_SOLUTION_MAX_CONCURRENCY`
  - `DEFAULT_DIAGNOSTIC_MAX_CONCURRENCY`
  - `COMMENT_ENRICHMENT_MIN_CONFIDENCE` (accept OP-only comment fields above this confidence)

### Notes
- LLM access is required for steps 1, 2, and the diagnostic fallback. Set your API keys as your normal environment expects (e.g., used by LangChain).
- IDs: we key records by `post_id`. The solution step joins BREAK labels with raw posts using this key.
- Per-post errors do not stop the batch. Failures are recorded on that record with an `error` field.
- Timestamps in filenames help track runs. Each run writes into `output/YYYY-MM-DD/`.


