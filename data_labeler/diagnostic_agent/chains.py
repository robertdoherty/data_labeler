# data_labeler_agent/solution_labeler_agent/chains.py
"""
Minimal diagnostic labeler chain:
- Loads diagnostics (ontology), rules, and golden examples
- Builds a constrained prompt to assign up to 2 diagnostic labels
- Returns JSON with label(s), confidence(s), rationale, and spans
"""

import os
import sys
import json
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_golden_examples(path: str, max_per_label: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    try:
        data = _load_json(path)
        out: Dict[str, List[Dict[str, Any]]] = {}
        for label, examples in data.items():
            if isinstance(examples, list):
                out[label] = examples[:max_per_label]
        return out
    except Exception:
        return {}


def _build_llm() -> ChatGoogleGenerativeAI:
    # Resolve API key similar to existing chain setup
    try:
        from local_secrets import GEMINI_API_KEY  # type: ignore
    except Exception:
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "local_secrets", os.path.join(_repo_root(), "local_secrets.py")
            )
            local_secrets = importlib.util.module_from_spec(spec)  # type: ignore
            assert spec and spec.loader
            spec.loader.exec_module(local_secrets)  # type: ignore
            GEMINI_API_KEY = local_secrets.GEMINI_API_KEY  # type: ignore
        except Exception:
            GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not found in local_secrets.py or environment variables")

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=GEMINI_API_KEY,
    )


DIAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an HVAC diagnostic classifier.

Task: Assign 0–2 PARENT diagnostic labels from the provided ontology. If unsure, output exactly dx.other_or_unclear. Be precise and conservative.

Provide:
- Allowed labels (with descriptions)
- Normalized symptoms text + raw title/body
- Equipment fields (family, subtype, brand)
- Few-shot examples by label (from real cases)

Instructions:
- Choose 0–2 labels ONLY from the allowed list.
- If unsure, return dx.other_or_unclear as the single label.
- Rate confidence ∈ [0,{confidence_max}] based on textual evidence; include short rationale and spans (quoted substrings) from the inputs.
- Do NOT invent content; use only provided text.

OUTPUT (STRICT JSON):
{{
  "predictions": [
    {{"label_id": "string", "confidence": 0.0, "rationale": "string", "spans": ["string"]}}
  ]
}}

Validation:
- Labels MUST be in the allowed set; otherwise use dx.other_or_unclear.
- 0–2 predictions only; sort by confidence descending.
- Return VALID JSON only (no markdown).
"""
    ),
    (
        "human",
        """
ALLOWED LABELS
{labels_block}

FEW-SHOT EXAMPLES (per label, up to 3)
{examples_block}

INPUT
post_id: {post_id}
title: {title}
body: {body}
equipment: {equip}
normalized_symptoms: {x_symptoms}
"""
    ),
])


def build_diagnostic_labeler_chain() -> RunnableSequence:
    llm = _build_llm()
    return DIAG_PROMPT | llm


def render_allowed_labels(ontology: Dict[str, Any]) -> str:
    # diagnostics_v1.json currently stores only label ids; add simple descriptions
    labels = ontology.get("labels", [])
    lines = []
    for lid in labels:
        # Minimal placeholder descriptions; can be extended later
        desc = {
            "dx.electrical_low_voltage_chain": "Low-voltage chain issues (through safeties)",
            "dx.power_supply_transformer": "Transformer power supply problems",
            "dx.control_open_or_short": "Control circuit open/short faults",
            "dx.contactor_relay_fault": "Contactor/relay contact or coil faults",
            "dx.airflow_restriction_or_ice": "Airflow restriction or iced coil",
            "dx.refrigerant_leak_or_low_charge": "Refrigerant leak or low charge",
            "dx.compressor_or_valve_fault": "Compressor or reversing/expansion valve issues",
            "dx.condensate_overflow_or_switch": "Condensate overflow or float/pan switch",
            "dx.mechanical_drive_or_bearing": "Drive/bearing/seizure/misalignment",
            "dx.sensor_or_safety_fault": "Sensor or safety device fault",
            "dx.install_or_wiring_issue": "Installation or wiring problems",
            "dx.controls_board_or_comm": "Controls board or communication faults",
            "dx.fuel_delivery_or_burner_tuning": "Fuel delivery or burner tuning (oil/gas)",
            "dx.water_ingress_cabinet_or_roof": "Water ingress to cabinet/roof",
            "dx.tools_vacuum_or_gauges": "Vacuum/gauge tool behavior",
            "dx.tools_misuse_or_maintenance": "Tool misuse/maintenance",
            "dx.safety_incident_or_electrical_contact": "Safety incident or electrical contact",
            "dx.other_or_unclear": "Other/unclear",
        }.get(lid, "")
        lines.append(f"- {lid}: {desc}")
    return "\n".join(lines)


def render_examples_block(gold: Dict[str, List[Dict[str, Any]]]) -> str:
    lines: List[str] = []
    for label, examples in gold.items():
        lines.append(f"[{label}]")
        for ex in examples:
            text = ex.get("text", "")
            equip = ex.get("equip", {})
            lines.append(f"  - text: {text}\n    equip: {equip}")
    return "\n".join(lines)



