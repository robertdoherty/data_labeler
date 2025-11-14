## Script to pull latest diagnostic chart

import sys
import os
from pathlib import Path

# Add project root to Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from config import GOLDEN_DIAGNOSTIC_CHART_PATH
import json

# Path to diagnostics_v1.json
DIAGNOSTICS_META_PATH = project_root / "data_labeler/rule_labeler/meta/diagnostics_v1.json"

def make_diagnostic_list(path: str) -> list[str]:
    with open(path, 'r') as f:
        data = json.load(f)
    diagnostics = [item["diagnostic_id"] for item in data.get("diagnostics", [])]
    return diagnostics

def update_diagnostics_meta(diagnostics: list[str]) -> None:
    """Update diagnostics_v1.json with the latest diagnostic IDs from golden set."""
    # Read current diagnostics_v1.json
    with open(DIAGNOSTICS_META_PATH, 'r') as f:
        data = json.load(f)
    
    # Update the labels field
    data["labels"] = diagnostics
    
    # Write back to file
    with open(DIAGNOSTICS_META_PATH, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Updated {DIAGNOSTICS_META_PATH} with {len(diagnostics)} diagnostic labels")

def main():
    # Convert relative path to absolute from project root
    path = project_root / GOLDEN_DIAGNOSTIC_CHART_PATH
    diagnostic_ids = make_diagnostic_list(str(path))
    
    print(f"📋 Extracted {len(diagnostic_ids)} diagnostic IDs from golden set")
    
    # Update diagnostics_v1.json
    update_diagnostics_meta(diagnostic_ids)

if __name__ == "__main__":
    main()