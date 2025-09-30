# research_agent/research_chains.py
"""
LangChain pipeline: Reddit JSON → Skilled Trades Industry Analysis (Markdown).
- Takes Reddit data from skilled trades subreddits
- Uses Gemini 2.5 Pro via langchain_google_genai
- Produces structured industry analysis report
"""

import os
import sys
import json
import logging
from typing import Union, Optional, Dict, Any

from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from local_secrets import GEMINI_API_KEY
except ImportError:
    try:
        # Try loading from parent directory
        import importlib.util
        spec = importlib.util.spec_from_file_location("local_secrets", os.path.join(parent_dir, "local_secrets.py"))
        local_secrets = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(local_secrets)
        GEMINI_API_KEY = local_secrets.GEMINI_API_KEY
    except:
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in local_secrets.py or environment variables")


# --------------------------
# LLM & Prompt
# --------------------------
def build_llm() -> ChatGoogleGenerativeAI:
    """Build the Gemini 2.5 Pro model instance"""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.25,
        google_api_key=GEMINI_API_KEY,
    )

# The exact prompt from the original file
PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system", """
You are an industry analyst specializing in skilled trades (HVAC, plumbing, electrical, carpentry, welding, automotive, data-center, etc.). Analyze the following Reddit dataset of posts with nested comments. Your goals: (1) produce an executive overview, (2) identify the biggest day-to-day pain points, (3) call out problems with current software (by name, if mentioned) and why those problems occur, and (4) summarize solutions tradespeople wish they had. Under each major theme, include key evidence posts.

Input

A JSON array reddit_data where each item has fields like:

post_id, title, body, author, created_utc, score, num_comments, upvote_ratio, url, subreddit, is_stickied, comments: [ {{ comment_id, body, author, created_utc, score, upvote_ratio? }} ]

Some fields may be missing; handle gracefully.

Method

Trade inference: Use subreddit and content cues to map posts to one or more trades (e.g., "r/HVAC" → HVAC; infer from text if subreddit is generic).

Engagement weight (credence): Rank evidence by:

post_weight = 0.5*normalize(score) + 0.3*log1p(num_comments) + 0.2*(upvote_ratio or 1.0)

comment_weight = 0.7*normalize(comment.score) + 0.3 if comment has many replies (approximate via presence/length if reply count missing).

When citing, prefer higher-weighted items. Ignore stickied posts unless they describe real issues.

Theme clustering: Group semantically similar issues (e.g., scheduling/dispatch, parts sourcing, documentation/knowledge, quoting/estimates, job matching/qualification, inventory, mobile app reliability, offline use, integrations, photos/video capture, warranty tracking, compliance/safety logs).

Software call-outs: Extract and normalize any software/tool names (e.g., ServiceTitan, FieldEdge, Housecall Pro, Salesforce Field Service, Jobber, custom spreadsheets). For each, summarize the complaint and the "why" (UX, missing feature, integration, data quality, pricing, mobile performance, offline).

Solutions wished for: Translate complaints into solution statements/features. Note cross-trade vs. trade-specific needs.

Quality controls: No fabrication. Use only content present in reddit_data. If evidence is thin, say so.

Output (Markdown only)

Executive Overview

2–4 bullets summarizing the most prevalent and severe pain points across trades

One line on software issues (top 2–3) and one line on desired solutions

Biggest Pain Points (Ranked)

For each theme (top 5–8):

Theme: short label — Prevalence: ~X% of posts — Severity: 1–5 — Confidence: High/Med/Low

Why it matters: concise business/operational impact (missed jobs, rework, truck rolls, margin erosion, safety/compliance risk)

Representative evidence (3 max):

[post_id] • r/{{subreddit}} • weight={{post_weight rounded}} — 1-sentence summary — [link if available]

short quote (≤20 words)

[post_id] … (same format)

(optional) high-signal comment in the same format, prefix with "Comment:"

Problems With Current Software (If Present)

{{Software Name}}: what's broken, why it happens, operational impact, any common workaround cited

Repeat per tool (limit 5). If only generic "our software/CRM" is mentioned, summarize generically.

Solutions Tradespeople Wish They Had

Bullet list of concrete features/solutions. For each: Feature — problem it solves — expected impact — cross-trade vs. trade-specific — any integration targets mentioned.

Trade-Specific Notes

Short bullets by trade (HVAC, plumbing, electrical, etc.) listing any unique pain points or priorities.

Methodology & Limits (One Short Paragraph)

Note weighting scheme, clustering approach, and any data sparsity or bias.

Constraints

Be concise and evidence-driven.

Do not reveal your reasoning steps; present results only.

Use only posts/comments from the input; no external knowledge.

Quotes ≤20 words; max 3 evidence items per theme.

If two themes are near-duplicates, merge and note synonyms.
"""),
    ("human", "Reddit JSON data follows:\n```json\n{json_data}\n```")
])


# --------------------------
# Chain Builder
# --------------------------
def build_research_chain() -> RunnableSequence:
    """Build the complete research analysis chain"""
    llm = build_llm()
    
    # Create the chain: prompt -> llm
    chain = PROMPT | llm
    
    return chain


# --------------------------
# Main Analysis Function
# --------------------------
def analyze_skilled_trades_data(reddit_data: Union[str, Dict[str, Any]]) -> str:
    """
    Analyze Reddit data for skilled trades insights
    
    Args:
        reddit_data: Either a JSON string or dict containing Reddit data
        
    Returns:
        Markdown analysis report as string
    """
    try:
        # Handle both string and dict inputs
        if isinstance(reddit_data, str):
            json_data = reddit_data
        else:
            json_data = json.dumps(reddit_data, indent=2)
        
        # Build and run the chain
        chain = build_research_chain()
        
        # Execute the analysis
        result = chain.invoke({
            "json_data": json_data
        })
        
        # Extract content from the result
        if hasattr(result, 'content'):
            return result.content
        else:
            return str(result)
            
    except Exception as e:
        logging.error(f"Error in skilled trades analysis: {e}")
        raise


# --------------------------
# Convenience Functions
# --------------------------
def analyze_from_file(json_file_path: str, output_file_path: Optional[str] = None) -> str:
    """
    Analyze Reddit data from a JSON file
    
    Args:
        json_file_path: Path to the Reddit JSON data file
        output_file_path: Optional path to save the analysis report
        
    Returns:
        Markdown analysis report as string
    """
    try:
        # Load the JSON data
        with open(json_file_path, 'r', encoding='utf-8') as f:
            reddit_data = json.load(f)
        
        # Run the analysis
        analysis_result = analyze_skilled_trades_data(reddit_data)
        
        # Save to file if requested
        if output_file_path:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(analysis_result)
            logging.info(f"Analysis saved to: {output_file_path}")
        
        return analysis_result
        
    except Exception as e:
        logging.error(f"Error analyzing from file {json_file_path}: {e}")
        raise


def analyze_latest_reddit_data(output_dir: str = "output") -> str:
    """
    Find the latest Reddit data file and analyze it
    
    Args:
        output_dir: Directory containing Reddit data files
        
    Returns:
        Markdown analysis report as string
    """
    import glob
    from datetime import datetime
    
    try:
        # Find all Reddit data files
        pattern = os.path.join(output_dir, "reddit_research_data_*.json")
        files = glob.glob(pattern)
        
        if not files:
            raise FileNotFoundError(f"No Reddit data files found in {output_dir}")
        
        # Get the most recent file
        latest_file = max(files, key=os.path.getctime)
        logging.info(f"Analyzing latest Reddit data file: {latest_file}")
        
        # Generate output filename
        base_name = os.path.basename(latest_file).replace("reddit_research_data_", "skilled_trades_analysis_")
        base_name = base_name.replace(".json", ".md")
        output_file = os.path.join(output_dir, base_name)
        
        # Run the analysis
        result = analyze_from_file(latest_file, output_file)
        
        print(f"✅ Analysis completed!")
        print(f"📊 Input: {latest_file}")
        print(f"📝 Output: {output_file}")
        
        return result
        
    except Exception as e:
        logging.error(f"Error analyzing latest Reddit data: {e}")
        raise


# --------------------------
# CLI Interface
# --------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Reddit data for skilled trades insights")
    parser.add_argument("--input", "-i", help="Input JSON file path")
    parser.add_argument("--output", "-o", help="Output markdown file path")
    parser.add_argument("--latest", "-l", action="store_true", help="Analyze the latest Reddit data file")
    
    args = parser.parse_args()
    
    try:
        if args.latest:
            # Analyze the latest file
            result = analyze_latest_reddit_data()
            print("\n" + "="*50)
            print("ANALYSIS PREVIEW (first 500 chars):")
            print("="*50)
            print(result[:500] + "..." if len(result) > 500 else result)
            
        elif args.input:
            # Analyze specific file
            result = analyze_from_file(args.input, args.output)
            print(f"✅ Analysis completed for {args.input}")
            if args.output:
                print(f"📝 Saved to: {args.output}")
            else:
                print("\n" + "="*50)
                print("ANALYSIS RESULT:")
                print("="*50)
                print(result)
        else:
            print("Please specify --input <file> or --latest")
            parser.print_help()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)