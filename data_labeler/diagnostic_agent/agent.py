# data_labeler_agent/solution_labeler_agent/agent.py
"""
Minimal diagnostic agent orchestrator.

Loads ontology, rules, and golden examples. Builds input from a post record
and calls a constrained LLM chain to predict up to 2 diagnostic labels.
"""

import os
import json
from typing import Dict, Any

from .schema import DiagnosticInput, DiagnosticOutput, enforce_allowed_predictions
from .chains import (
    build_diagnostic_labeler_chain,
    render_allowed_labels,
    render_examples_block,
    _load_json,
    _load_golden_examples,
)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _normalize_symptoms(symptoms: list[str], title: str, body: str) -> str:
    # Simple concatenation; reuse existing make_error_prediction normalize if desired later
    clean = [s.strip() for s in (symptoms or []) if isinstance(s, str) and s.strip()]
    if clean:
        return "; ".join(clean)
    text = f"{title} {body}".strip()
    return " ".join(text.split())


def build_llm_payload(
    post: DiagnosticInput,
    ontology: Dict[str, Any],
    gold_examples: Dict[str, Any],
) -> Dict[str, Any]:
    labels_block = render_allowed_labels(ontology)
    examples_block = render_examples_block(gold_examples)
    x_symptoms = _normalize_symptoms(post.get("symptoms", []), post.get("title", ""), post.get("body", ""))

    return {
        "labels_block": labels_block,
        "examples_block": examples_block,
        "post_id": post.get("post_id", ""),
        "title": post.get("title", ""),
        "body": post.get("body", ""),
        "equip": post.get("equip", {}),
        "x_symptoms": x_symptoms,
    }


def predict_diagnostics(post: DiagnosticInput) -> DiagnosticOutput:
    root = _repo_root()
    # Align paths with rule labeler assets used elsewhere in the pipeline
    ontology_path = os.path.join(root, "data_labeler", "rule_labeler", "meta", "diagnostics_v1.json")
    gold_path = os.path.join(root, "data_labeler", "rule_labeler", "gold", "golden_examples.json")

    ontology = _load_json(ontology_path)
    gold = _load_golden_examples(gold_path, max_per_label=3)
    prompt_vars = build_llm_payload(post, ontology, gold)

    chain = build_diagnostic_labeler_chain()
    result = chain.invoke(prompt_vars)
    text = result.content if hasattr(result, "content") else str(result)
    
    try:
        parsed = json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start:end+1])
            except Exception:
                parsed = {}
        else:
            parsed = {}

    allowed = set(ontology.get("labels", []))
    preds = parsed.get("predictions", []) if isinstance(parsed, dict) else []
    
    return {
        "predictions": enforce_allowed_predictions(preds, allowed, max_labels=2),
        "ontology": "diagnostics_v1",
        "provenance": "llm_v1",
    }


__all__ = [
    "predict_diagnostics",
]



