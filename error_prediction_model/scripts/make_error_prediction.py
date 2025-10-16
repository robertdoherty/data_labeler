"""Utilities for error prediction preprocessing.

Provides functions to normalize text, extract fields from labeled solution
data, and construct a canonical symptoms string for each post.
"""

from typing import Any, Dict, List
import json
import re

PUNCT_RE = re.compile(r"[,;:?!()<>]")
WS_RE    = re.compile(r"\s+")

def normalize(text: str, cfg: dict) -> tuple[str, list[str]]:
    """Normalize a text string using regex-based rules from config.

    The normalization pipeline lowercases, collapses whitespace, applies alias
    and unit/phrase compaction rules, strips punctuation while preserving
    placeholder tokens like ``<brand_...>``, and trims to a maximum length.

    Args:
        text: Raw input text to normalize.
        cfg: Config with keys ``aliases``, ``unit_patterns``, ``phrase_compact``,
            and optional ``max_len``.

    Returns:
        A tuple ``(normalized_text, fired_rule_tags)``.
    """
    fired = []
    s = text.lower()
    s = s.replace("\u200b", " ").replace("\n", " ")
    s = WS_RE.sub(" ", s).strip()

    # aliases
    for rule in cfg["aliases"]:
        before = s
        s = re.sub(rule["pattern"], rule["repl"], s)
        if s != before: fired.append(f"alias:{rule['repl']}")

    # units
    for rule in cfg["unit_patterns"]:
        before = s
        s = re.sub(rule["pattern"], rule["format"], s)
        if s != before: fired.append(f"unit:{rule['name']}")

    # phrases
    for rule in cfg["phrase_compact"]:
        before = s
        s = re.sub(rule["pattern"], rule["repl"], s)
        if s != before: fired.append(f"phrase:{rule['repl']}")

    # punctuation (preserve <tokens>)
    def _safe_punct(m):
        return " "
    s = PUNCT_RE.sub(_safe_punct, s)
    s = WS_RE.sub(" ", s).strip()

    # length cap
    if len(s) > cfg.get("max_len", 1500):
        s = s[:cfg["max_len"]].rsplit(" ", 1)[0]

    return s, fired


def extract_post_fields(solutions_json_path: str) -> List[Dict[str, Any]]:
    """Extract post id, symptoms, and equipment info from a solutions JSON.

    Args:
        solutions_json_path: Path to the labeled solutions JSON file.

    Returns:
        A list of dictionaries with keys:
        - ``post_id`` (str)
        - ``symptoms`` (List[str])
        - ``equip`` (dict with ``family``, ``subtype``, ``brand``)
    """

    with open(solutions_json_path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    extracted: List[Dict[str, Any]] = []

    for obj in data.values():
        labels: Dict[str, Any] = obj.get("labels", {})
        error_report: Dict[str, Any] = labels.get("error_report", {})
        system_info: Dict[str, Any] = labels.get("system_info", {})

        post_id: str = obj.get("post_id", "")
        symptoms: List[str] = error_report.get("symptoms", []) or []

        equip: Dict[str, Any] = {
            "family": system_info.get("asset_family", ""),
            "subtype": system_info.get("asset_subtype", ""),
            "brand": system_info.get("brand", ""),
        }

        extracted.append({
            "post_id": post_id,
            "symptoms": symptoms,
            "equip": equip,
        })

    return extracted



def build_x_symptoms(symptoms: List[str], title: str, body: str, cfg: dict) -> tuple[str, str]:
    """Build a normalized symptoms string and record its provenance.

    Chooses the symptoms list if present; otherwise falls back to
    ``title + " " + body``. Performs normalization and post-normalization
    de-duplication for list inputs, then truncates to the configured max length.

    Args:
        symptoms: Candidate symptom phrases (may be empty).
        title: Post title used for fallback.
        body: Post body used for fallback.
        cfg: Normalization config (``aliases``, ``unit_patterns``, ``phrase_compact``,
            optional ``max_len``).

    Returns:
        A tuple ``(x_symptoms, provenance)`` where ``provenance`` is either
        ``"symptoms_list"`` or ``"title_body_fallback"``.
    """

    clean_symptoms = [s.strip() for s in (symptoms or []) if isinstance(s, str) and s.strip()]

    if clean_symptoms:
        provenance = "symptoms_list"
        assembled = "; ".join(clean_symptoms)
        used_list = True
    else:
        provenance = "title_body_fallback"
        t = title or ""
        b = body or ""
        assembled = f"{t} {b}".strip()
        used_list = False

    # Normalize
    normalized, _ = normalize(assembled, cfg)

    # Post-normalization cleanup for list-based inputs
    if used_list:
        parts = [p.strip() for p in normalized.split(";")]
        # remove empties and de-duplicate while preserving order
        seen = set()
        unique_parts = []
        for p in parts:
            if p and p not in seen:
                seen.add(p)
                unique_parts.append(p)
        normalized = "; ".join(unique_parts)

    # Truncate to max length on a word boundary (extra safety beyond normalize)
    max_len = cfg.get("max_len", 1500)
    if len(normalized) > max_len:
        truncated = normalized[:max_len]
        normalized = truncated.rsplit(" ", 1)[0] if " " in truncated else truncated

    return normalized, provenance


def map_label(x_symptoms: str, equip: dict, rules: dict) -> tuple[str, float, list[str]]:
    """Map a normalized symptoms string to a label ID and confidence.

    Args:
        x_symptoms: Normalized symptoms string.
        equip: Equipment info dictionary.
        rules: Rules dictionary.

    Returns:
        A tuple ``(label_id, confidence, fired_rules)``.
    """
    def _to_list(value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    def _equip_matches(rule_equip: dict, post_equip: dict) -> bool:
        if not rule_equip:
            return True
        post_equip = post_equip or {}
        for key in ("family", "subtype", "brand"):
            allowed = rule_equip.get(key)
            if allowed is None:
                continue
            post_val = post_equip.get(key, "")
            if not post_val:
                return False
            if post_val not in _to_list(allowed):
                return False
        return True

    ordered = rules.get("rules") or rules.get("ordered_rules") or []
    hits: list[tuple[float, int, str]] = []  # (score, index, rule_id)

    for idx, rule in enumerate(ordered):
        rule_id = rule.get("id", f"rule_{idx}")
        phrases_all = _to_list(rule.get("phrases_all", []))
        if not all((p in x_symptoms) for p in phrases_all):
            continue
        if not _equip_matches(rule.get("equip", {}), equip):
            continue

        score = float(rule.get("score", 1.0))
        hits.append((score, idx, rule_id))

    if hits:
        hits.sort(key=lambda t: (-t[0], t[1]))
        best_score, _best_idx, best_id = hits[0]
        fired_rules = [h[2] for h in hits]
        return best_id, best_score, fired_rules

    fallback = rules.get("fallback", {})
    return (
        fallback.get("id", "dx.other_or_unclear"),
        float(fallback.get("score", 0.2)),
        [],
    )

def make_error_prediction_row(
    post_id: str,
    x_symptoms: str,
    equip: dict,
    label_id: str,
    sample_weight: float,
    ontology: str,
    provenance: str,
    fired_rules: list[str] | None = None,
    fired_norm: list[str] | None = None,
) -> dict:
    """Assemble a single TaskA row.

    Args:
        post_id: Post identifier.
        x_symptoms: Normalized symptom text.
        equip: Equipment fields with ``family``, ``subtype``, ``brand``.
        label_id: Diagnostic label id to assign.
        sample_weight: Weight in ``[0.5, 1.0]``; will be clamped.
        ontology: Ontology/version string for labels.
        provenance: Source provenance for ``x_symptoms``.
        fired_rules: Optional list of rule ids that fired.
        fired_norm: Optional list of normalization events fired.

    Returns:
        A dictionary representing one training row with keys including
        ``post_id``, ``x_symptoms``, ``equip``, ``y_diag``, ``sample_weight``,
        ``ontology``, and ``provenance``; optionally ``fired_rules`` and
        ``fired_normalizer``.
    """
    row = {
        "post_id": post_id,
        "x_symptoms": x_symptoms,
        "equip": {
            "family": equip.get("family",""),
            "subtype": equip.get("subtype",""),
            "brand":  equip.get("brand",""),
        },
        "y_diag": [[label_id, 1.0]],
        "sample_weight": float(max(0.5, min(1.0, sample_weight))),
        "ontology": ontology,
        "provenance": provenance,
    }
    # Optional audit fields
    if fired_rules: row["fired_rules"] = fired_rules
    if fired_norm:  row["fired_normalizer"] = fired_norm
    return row


def append_jsonl(path: str, obj: dict) -> None:
    """Append one JSON object as a JSONL line.

    Args:
        path: Destination file path.
        obj: JSON-serializable object to append.
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def update_golden_examples(
    store: dict[str, list[dict]],
    label_id: str,
    post_id: str,
    text: str,
    equip: dict,
    fired_rules: list[str],
    cap_per_label: int = 25,
) -> None:
    """Collect up to ``cap_per_label`` examples per label id.

    Args:
        store: Mapping of label id to list of example dicts.
        label_id: Label bucket to add the example under.
        post_id: Post identifier.
        text: Example text to store.
        equip: Equipment fields with ``family``, ``subtype``, ``brand``.
        fired_rules: Rule ids that matched for this example.
        cap_per_label: Maximum examples to retain per label.

    Returns:
        None.
    """
    bucket = store.setdefault(label_id, [])
    if len(bucket) < cap_per_label:
        bucket.append({
            "post_id": post_id,
            "text": text,
            "equip": {k: equip.get(k,"") for k in ("family","subtype","brand")},
            "hits": fired_rules,
        })


def write_json(path: str, obj: dict) -> None:
    """Write a dictionary to a JSON file.

    Args:
        path: Destination file path.
        obj: JSON-serializable object to write.

    Returns:
        None.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

