# data_labeler_agent/solution_labeler_agent.py
"""
Solution Labeler Agent: Processes BREAK-labeled posts to find solutions from comments.
Minimal, efficient pipeline with per-post error isolation.
"""

import json
import os
from typing import Dict, Any, List, Optional


def load_break_labels(labels_file: str) -> Dict[str, Dict[str, Any]]:
    """
    Load labels JSON; return dict {post_id: label_json} for break_label == "BREAK".
    
    Args:
        labels_file: Path to labeled_posts_*.json file
        
    Returns:
        Dictionary mapping post_id to full label JSON for BREAK posts only
    """
    with open(labels_file, "r", encoding="utf-8") as f:
        labels_doc = json.load(f)
    
    break_labels: Dict[str, Dict[str, Any]] = {}
    
    for result in labels_doc.get("results", []):
        if not isinstance(result, dict):
            continue
        
        # Get post ID from the 'id' field in labels
        post_id = result.get("id")
        if not post_id:
            continue
        
        # Only include BREAK posts
        if result.get("break_label") == "BREAK":
            break_labels[post_id] = result
    
    return break_labels


def build_posts_index(raw_file: str, allowed_ids: Optional[set] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load raw JSON; return dict {post_id: post} for posts where post_id (or fallback id) 
    is in allowed_ids (if provided).
    
    Args:
        raw_file: Path to reddit_research_data_*.json file
        allowed_ids: Optional set of post IDs to include (for efficiency)
        
    Returns:
        Dictionary mapping post_id to full post object with all metadata and comments
    """
    with open(raw_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    posts_index: Dict[str, Dict[str, Any]] = {}
    
    # Navigate through subreddits structure
    subreddits = raw_data.get("subreddits", {})
    for subreddit_name, subreddit_data in subreddits.items():
        for post in subreddit_data.get("posts", []):
            # Try 'post_id' first, fallback to 'id'
            post_id = post.get("post_id") or post.get("id")
            if not post_id:
                continue
            
            # If allowed_ids provided, only include those posts
            if allowed_ids is not None and post_id not in allowed_ids:
                continue
            
            # Store the full post object
            posts_index[post_id] = post
    
    return posts_index


def enrich_breaks_with_posts(break_labels: Dict[str, Dict[str, Any]], 
                              posts_index: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Join into {post_id: {"post_id": ..., "labels": <labels>, "post": <full_post>}}. 
    Skip missing posts.
    
    Args:
        break_labels: Dictionary of BREAK labels by post_id
        posts_index: Dictionary of full post objects by post_id
        
    Returns:
        Enriched dictionary with both labels and full post data
    """
    enriched: Dict[str, Dict[str, Any]] = {}
    
    for post_id, labels in break_labels.items():
        # Look up the full post
        post = posts_index.get(post_id)
        
        # Skip if post not found in raw data
        if not post:
            continue
        
        # Build enriched structure
        enriched[post_id] = {
            "post_id": post_id,
            "labels": labels,
            "post": post  # Full, unmodified post with all metadata and comments
        }
    
    return enriched


def run_solutions_on_enriched(enriched: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Iterate through all BREAK posts and call build_solution_labeler_chain for each.
    Pass the full enriched data (no alterations). Append LLM output to each record.
    
    Args:
        enriched: Dictionary of enriched BREAK posts with labels and full post data
        
    Returns:
        Enriched dictionary with solution appended to each post
    """
    # Import the chain builder
    try:
        from .data_labeler_chains import build_solution_labeler_chain
    except ImportError:
        from data_labeler_chains import build_solution_labeler_chain
    
    chain = build_solution_labeler_chain()
    
    # Iterate through each BREAK post
    for post_id, rec in enriched.items():
        try:
            post = rec.get("post", {})
            labels = rec.get("labels", {})
            
            # Extract what we need for the LLM
            title = post.get("title", "")
            user_id = post.get("author", "")
            comments = post.get("comments", [])
            
            # Build problem diagnosis from labels
            symptoms = labels.get("symptoms", [])
            asset_family = labels.get("asset_family")
            problem_diagnosis = f"asset_family={asset_family}; symptoms={symptoms}" if asset_family or symptoms else "unknown"
            
            # Convert comments to JSON string for the LLM
            comments_json = json.dumps(comments, ensure_ascii=False)
            
            # Prepare input for build_solution_labeler_chain
            chain_input = {
                "post_id": post_id,
                "title": title,
                "user_id": user_id,
                "problem_diagnosis": problem_diagnosis,
                "comments_json": comments_json
            }
            
            # Call the LLM chain
            output = chain.invoke(chain_input)
            
            # Parse the output (handle if it's a LangChain message object)
            if hasattr(output, "content"):
                solution_text = output.content
            else:
                solution_text = str(output)
            
            # Try to parse as JSON
            try:
                solution = json.loads(solution_text)
            except:
                # If parsing fails, extract JSON from text
                start = solution_text.find("{")
                end = solution_text.rfind("}")
                if start != -1 and end != -1:
                    try:
                        solution = json.loads(solution_text[start:end+1])
                    except:
                        solution = {"raw": solution_text}
                else:
                    solution = {"raw": solution_text}
            
            # Append solution to the current record
            rec["solution"] = solution
            
        except Exception as e:
            # Per-post error isolation
            rec["solution"] = {
                "summary": "No clear solution.",
                "error": str(e),
                "confidence": 0.0
            }
    
    return enriched


def write_json(data: Dict[str, Any], out_path: str) -> str:
    """
    Write JSON to disk. Keep simple.
    
    Args:
        data: Dictionary to write
        out_path: Output file path
        
    Returns:
        Path to written file
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return out_path


def process_breaks_to_solutions(raw_file: str, labels_file: str, out_file: str) -> str:
    """
    Orchestrate the full pipeline.
    
    Args:
        raw_file: Path to reddit_research_data_*.json
        labels_file: Path to labeled_posts_*.json
        out_file: Path for output solutions_*.json
        
    Returns:
        Path to output file
    """
    # Load BREAK labels
    break_labels = load_break_labels(labels_file)
    
    # Early return if no BREAK posts
    if not break_labels:
        return write_json({}, out_file)
    
    # Build posts index (only for BREAK post IDs)
    posts_index = build_posts_index(raw_file, allowed_ids=set(break_labels.keys()))
    
    # Enrich: join labels + full posts
    enriched = enrich_breaks_with_posts(break_labels, posts_index)
    
    # Run solution finding on all enriched posts
    solved = run_solutions_on_enriched(enriched)
    
    # Write to disk
    return write_json(solved, out_file)


# CLI interface when run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Solution Labeler - Find solutions from comments for BREAK posts")
    parser.add_argument("--raw", "-r", required=True, help="Path to reddit_research_data_*.json")
    parser.add_argument("--labels", "-l", required=True, help="Path to labeled_posts_*.json")
    parser.add_argument("--output", "-o", required=True, help="Path for output solutions_*.json")
    
    args = parser.parse_args()
    
    print(f"🔍 Loading BREAK labels from: {args.labels}")
    print(f"📋 Loading raw posts from: {args.raw}")
    print(f"🤖 Processing with LLM...")
    
    output_file = process_breaks_to_solutions(
        raw_file=args.raw,
        labels_file=args.labels,
        out_file=args.output
    )
    
    print(f"✅ Solutions written to: {output_file}")

