"""
Data Labeler Agent: Orchestrates break labeling + solution extraction pipeline.
"""

import os
import sys
from datetime import datetime
from typing import Optional

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
    
    # Step 1: Break labeling
    print(f"\n📊 Step 1: Break labeling...")
    break_agent = BreakLabelerAgent(output_dir=output_dir)
    break_result = break_agent.label_from_json_file(
        reddit_data_file,
        subreddits=subreddits
    )
    
    if not break_result.get("success"):
        raise Exception(f"Break labeling failed: {break_result.get('error')}")
    
    break_labels_file = break_result["output_file"]
    print(f"✅ Break labels: {break_labels_file}")
    
    # Step 2: Solution extraction
    print(f"\n💡 Step 2: Solution extraction...")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    solutions_file = os.path.join(output_dir, f"solutions_{timestamp}.json")
    
    solutions_path = process_breaks_to_solutions(
        raw_file=reddit_data_file,
        labels_file=break_labels_file,
        out_file=solutions_file
    )
    
    print(f"✅ Solutions: {solutions_path}")
    print(f"\n🎉 Pipeline complete!")
    
    return {
        "reddit_data": reddit_data_file,
        "break_labels": break_labels_file,
        "solutions": solutions_path
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

