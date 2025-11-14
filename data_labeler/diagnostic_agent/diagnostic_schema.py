"""Schemas for the diagnostic labeler minimal agent.

These are lightweight typing helpers used by the diagnostic chains/agent.
"""

from typing import TypedDict, List, Dict, Any, Sequence, Set


class Equipment(TypedDict, total=False):
    family: str
    subtype: str
    brand: str


class DiagnosticInput(TypedDict):
    post_id: str
    title: str
    body: str
    symptoms: List[str]
    equip: Equipment


class DiagnosticPrediction(TypedDict, total=False):
    label_id: str
    confidence: float
    rationale: str


class DiagnosticOutput(TypedDict, total=False):
    predictions: List[DiagnosticPrediction]
    ontology: str
    provenance: str


FALLBACK_LABEL = "dx.other_or_unclear"


def enforce_allowed_predictions(
    predictions: Sequence[DiagnosticPrediction] | None,
    allowed: Set[str],
    max_labels: int = 2
) -> List[DiagnosticPrediction]:
    """Filter predictions to allowed labels; fallback if none valid."""
    kept: List[DiagnosticPrediction] = []
    if predictions:
        for p in predictions:
            lid = (p.get("label_id") or "").strip()
            if lid in allowed:
                kept.append(p)
    if not kept:
        kept = [{"label_id": FALLBACK_LABEL, "confidence": 0.2, "rationale": "fallback"}]
    kept.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
    return kept[:max_labels]


