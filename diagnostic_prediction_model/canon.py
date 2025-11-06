# diagnostic_prediction_model/canon.py

import re


def canonicalize_symptoms(symptoms: str) -> str:
    """Canonicalize a symptoms string.

    Args:
        xsymptoms: The symptoms string to canonicalize.

    Returns:
        The canonicalized symptoms string.
    """
    parts = [p.strip().lower() for p in symptoms.split(";")]
    out = []
    for p in parts:
        p = re.sub(r"[.,!?]+$", "", p)
        p = re.sub(r"\s+", " ", p).strip()
        if p:
            out.append(p.replace(" ", "_"))
    seen, dedup = set(), []
    for t in out:
        if t not in seen:
            seen.add(t); dedup.append(t)
    return dedup


def canonicalize_equipment(equipment: str) -> str:
    """Canonicalize an equipment string.

    Args:
        equipment: The equipment string to canonicalize.

    Returns:
        The canonicalized equipment string.
    """
    return equipment.lower().strip()


