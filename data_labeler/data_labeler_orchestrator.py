"""Data Labeler Agent: Orchestrates the full HVAC labeling pipeline."""

import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Ensure imports work across direct script and package contexts
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
data_agent_dir = os.path.join(current_dir, "data_labeler_agent")
if data_agent_dir not in sys.path:
    sys.path.insert(0, data_agent_dir)

try:
    from data_labeler_agent.break_labeler_agent.break_labeler_agent import BreakLabelerAgent
except ImportError:
    from break_labeler_agent.break_labeler_agent import BreakLabelerAgent

try:
    from data_labeler_agent.solution_labeler_agent.solution_labeler_agent import process_breaks_to_solutions
except ImportError:
    from solution_labeler_agent.solution_labeler_agent import process_breaks_to_solutions

try:
    from data_labeler_agent.rule_labeler.scripts.make_error_prediction import (
        build_x_symptoms,
        map_label,
        make_error_prediction_row,
        _prepare_rules_with_normalizer,
    )
except ImportError:
    from rule_labeler.scripts.make_error_prediction import (
        build_x_symptoms,
        map_label,
        make_error_prediction_row,
        _prepare_rules_with_normalizer,
    )

try:
    from data_labeler_agent.final_diagnostic_agent.agent import predict_diagnostics
except ImportError:
    try:
        from final_diagnostic_agent.agent import predict_diagnostics
    except ImportError:
        predict_diagnostics = None  # type: ignore


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: Any) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def _format_hms(seconds: float) -> str:
    total_seconds = int(max(0, round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _extract_equipment(labels: Dict[str, Any]) -> Dict[str, str]:
    system_info = labels.get("system_info", {}) if isinstance(labels, dict) else {}
    return {
        "family": system_info.get("asset_family", "") or "",
        "subtype": system_info.get("asset_subtype", "") or "",
        "brand": system_info.get("brand", "") or "",
    }


def _clamp_conf(value: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _augment_with_diagnostics(
    records: Dict[str, Dict[str, Any]],
    rules: Dict[str, Any],
    norm_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Apply rule-based diagnostics and LLM fallback to enriched records."""

    rule_rows: List[Dict[str, Any]] = []
    final_rows: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {"llm_attempted": 0, "llm_succeeded": 0, "llm_errors": []}

    llm_available = predict_diagnostics is not None

    for post_id, rec in records.items():
        labels = rec.get("labels", {}) if isinstance(rec, dict) else {}
        post = rec.get("post", {}) if isinstance(rec, dict) else {}

        error_report = labels.get("error_report", {}) if isinstance(labels, dict) else {}
        symptoms = error_report.get("symptoms", []) or []
        title = post.get("title", "") or ""
        body = post.get("body", "") or ""

        equip = _extract_equipment(labels)

        try:
            x_symptoms, provenance = build_x_symptoms(symptoms, title, body, norm_cfg)
        except Exception:
            x_symptoms, provenance = "", "unavailable"

        label_id, rule_conf, fired_rules = map_label(x_symptoms, equip, rules)
        rule_provenance = f"rules_v1::{provenance}"

        rule_row = make_error_prediction_row(
            post_id=post_id,
            x_symptoms=x_symptoms,
            equip=equip,
            label_id=label_id,
            sample_weight=1.0,
            ontology="diagnostics_v1",
            provenance=rule_provenance,
            fired_rules=fired_rules,
        )
        rule_row["rule_confidence"] = _clamp_conf(rule_conf)
        rule_rows.append(rule_row)

        final_label = label_id
        final_conf = _clamp_conf(rule_conf)
        final_provenance = rule_provenance
        llm_payload: Optional[Dict[str, Any]] = None

        is_unclear = (label_id == "dx.other_or_unclear")
        need_llm = llm_available and (not fired_rules or is_unclear)
        if need_llm:
            stats["llm_attempted"] += 1
            try:
                llm_payload = predict_diagnostics({
                    "post_id": post_id,
                    "title": title,
                    "body": body,
                    "symptoms": symptoms,
                    "equip": equip,
                })  # type: ignore[arg-type]
                preds = llm_payload.get("predictions", []) if isinstance(llm_payload, dict) else []
                if preds:
                    top = preds[0]
                    maybe_label = (top.get("label_id") or "").strip()
                    if maybe_label:
                        final_label = maybe_label
                        final_conf = _clamp_conf(top.get("confidence", final_conf))
                        final_provenance = llm_payload.get("provenance", "llm_v1")
                        stats["llm_succeeded"] += 1
                else:
                    llm_payload.setdefault("predictions", [])  # type: ignore
            except Exception as exc:  # pragma: no cover - defensive for missing creds
                stats.setdefault("llm_errors", []).append({"post_id": post_id, "error": str(exc)})
                llm_available = False
                llm_payload = {"error": str(exc)}

        rec["x_symptoms"] = x_symptoms
        rec["diagnostics"] = {
            "rule_based": {
                "label_id": label_id,
                "confidence": _clamp_conf(rule_conf),
                "fired_rules": fired_rules,
                "provenance": rule_provenance,
            },
            "llm": llm_payload,
            "final": {
                "label_id": final_label,
                "confidence": final_conf,
                "provenance": final_provenance,
                "source": "llm" if final_provenance.startswith("llm") else "rules",
            },
        }

        final_rows.append({
            "post_id": post_id,
            "x_symptoms": x_symptoms,
            "x_post": f"{title.strip()}\n\n{body.strip()}".strip(),
            "equip": equip,
            "y_diag": [[final_label, 1.0]],
            "provenance": final_provenance,
            "rule_label": label_id,
            "rule_confidence": _clamp_conf(rule_conf),
            "rule_fired_rules": fired_rules,
        })

    return records, rule_rows, final_rows, stats


def process_reddit_data_to_solutions(
    reddit_data_file: str,
    output_dir: str = "output",
    subreddits: Optional[list] = None
) -> dict:
    """
    Full pipeline: Reddit data → Break labels → Solutions
    
    Args:
        reddit_data_file: Path to reddit_research_data_*.json
        output_dir: Output directory for intermediate and final files
        subreddits: Optional list of subreddit names to filter
        
    Returns:
        Dict with paths to intermediate and final output files
    """
    print(f"🚀 Starting data labeling pipeline...")
    print(f"📥 Input: {reddit_data_file}")
    t0 = time.time()
    total_posts_processed = 0
    
    # Step 1: Break labeling
    print(f"\n📊 Step 1: Break labeling...")
    step1_start = time.time()
    break_agent = BreakLabelerAgent(output_dir=output_dir)
    break_result = break_agent.label_from_json_file(
        reddit_data_file,
        subreddits=subreddits
    )
    
    if not break_result.get("success"):
        raise Exception(f"Break labeling failed: {break_result.get('error')}")
    
    break_labels_file = break_result["output_file"]
    print(f"✅ Break labels: {break_labels_file}")
    print(f"⏱️ Step 1 duration: {_format_hms(time.time() - step1_start)}")
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Step 2: Solution extraction
    print(f"\n💡 Step 2: Solution extraction...")
    solutions_file = os.path.join(output_dir, f"solutions_{timestamp}.json")
    step2_start = time.time()
    solutions_path = process_breaks_to_solutions(
        raw_file=reddit_data_file,
        labels_file=break_labels_file,
        out_file=solutions_file
    )
    print(f"✅ Solutions: {solutions_path}")
    print(f"⏱️ Step 2 duration: {_format_hms(time.time() - step2_start)}")

    solutions_doc = _load_json(solutions_path) if os.path.exists(solutions_path) else {}
    try:
        total_posts_processed = len(solutions_doc) if isinstance(solutions_doc, (dict, list)) else 0
    except Exception:
        total_posts_processed = 0
    print(f"🧮 Posts processed: {total_posts_processed}")

    # Step 3: Rule-based diagnostics
    print(f"\n🧠 Step 3: Rule-based diagnostics...")
    step3_start = time.time()
    rules_path = os.path.join(current_dir, "rule_labeler", "meta", "rules_v1.json")
    norm_cfg_path = os.path.join(current_dir, "rule_labeler", "scripts", "make_error_prediction_config.json")

    rules_cfg = _load_json(rules_path) if os.path.exists(rules_path) else {}
    norm_cfg = _load_json(norm_cfg_path) if os.path.exists(norm_cfg_path) else {}
    if not norm_cfg:
        norm_cfg = {"aliases": [], "unit_patterns": [], "phrase_compact": [], "max_len": 1500}
    prepared_rules = _prepare_rules_with_normalizer(rules_cfg, norm_cfg) if rules_cfg else {}

    augmented, rule_rows, final_rows, stats = _augment_with_diagnostics(solutions_doc, prepared_rules, norm_cfg)

    rule_rows_file = os.path.join(output_dir, f"rule_predictions_{timestamp}.json")
    diagnostics_file = os.path.join(output_dir, f"solutions_with_diagnostics_{timestamp}.json")
    final_dataset_file = os.path.join(output_dir, f"diagnostic_dataset_{timestamp}.json")

    _write_json(rule_rows_file, rule_rows)
    _write_json(diagnostics_file, augmented)
    _write_json(final_dataset_file, final_rows)

    print(f"✅ Rule predictions: {rule_rows_file}")
    print(f"⏱️ Step 3 duration: {_format_hms(time.time() - step3_start)}")

    # Step 4: Final diagnostic agent summary
    if stats.get("llm_attempted"):
        print(
            f"🤖 Final diagnostic agent attempted {stats['llm_attempted']} posts "
            f"(success: {stats.get('llm_succeeded', 0)})"
        )
    if stats.get("llm_errors"):
        print(f"⚠️ Final diagnostic agent errors: {len(stats['llm_errors'])}")

    print(f"✅ Solutions + diagnostics: {diagnostics_file}")
    print(f"✅ Final dataset: {final_dataset_file}")
    print(f"⏳ Total runtime: {_format_hms(time.time() - t0)}")
    print(f"🧮 Total posts processed: {total_posts_processed}")
    print(f"\n🎉 Pipeline complete!")

    return {
        "reddit_data": reddit_data_file,
        "break_labels": break_labels_file,
        "solutions": solutions_path,
        "rule_predictions": rule_rows_file,
        "solutions_with_diagnostics": diagnostics_file,
        "final_dataset": final_dataset_file,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Labeler Agent - Full pipeline")
    parser.add_argument("input", help="Path to reddit_research_data_*.json")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--subs", help="Comma-separated subreddit names to include")
    
    args = parser.parse_args()
    
    subs = [s.strip() for s in args.subs.split(",")] if args.subs else None
    
    try:
        result = process_reddit_data_to_solutions(
            reddit_data_file=args.input,
            output_dir=args.output_dir,
            subreddits=subs
        )
        print(f"\n📋 Output files:")
        for key, path in result.items():
            print(f"  {key}: {path}")
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        sys.exit(1)

