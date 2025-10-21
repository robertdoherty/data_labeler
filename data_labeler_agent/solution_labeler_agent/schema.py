"""Schemas for the diagnostic labeler minimal agent.

These are lightweight typing helpers used by the diagnostic chains/agent.
"""

from typing import TypedDict, List, Dict, Any


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
    spans: List[str]


class DiagnosticOutput(TypedDict, total=False):
    post_id: str
    ontology: str
    provenance: str
    x_symptoms: str
    predictions: List[DiagnosticPrediction]
    extra: Dict[str, Any]


