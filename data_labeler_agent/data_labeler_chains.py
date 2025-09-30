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
    """Build the Gemini 1.5 Flash model instance"""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.25,
        google_api_key=GEMINI_API_KEY,
    )

# The exact prompt from the original file
PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system", """
Given a JSON array of reddit posts, select posts  that describe a specific HVAC malfunction (equipment not working, degraded performance, error codes, leaks, trips, unsuual noise/odors)m that likely requires diagnosis or repair)

For eacg post
1. Label as BREAK or NON_BREAK
2. If break, extract fields: asset_family, symptoms[], model_text, error_codes, has_images, brand
3. Provide per field confidence 0.0-1.0 only if not NULL

Fields you may use per post:
{{title, body, score, num_comments, upvote_ratio}}

Decusion rubric (apply all):
1. BREAK only if it mentions a concrete symptom or failure mode (e.g., no cooling, tripping breaker, fan won't spin)
2. Reject posts that are brand comparisons, purhcase/quote questions, new installs, or non-HVAC topics
3. asset_family includes {{ac, furnance, heat_pump, mini_split, air_handler, other}}
4. Use null when unsure. Prefer precisison over recall
5. IF NON_BREAK, only fill in id, break_label, and break_confidence, null other fields


Output format (strict JSON, no extra text):
{{
  "results": [
    {{
      "id": "<post id>",
      "break_label": "BREAK | NON_BREAK",
      "break_confidence": "0.0|0.2|0.4|0.51|0.65|0.8|0.9",
      "asset_family": "ac|furnace|heat_pump|mini_split|air_handler|other|null",
      "asset_family_confidence": "...",
      "symptoms": ["..."],
      "symptoms_confidence": "...",
      "model_text": "IM-500X 2019",
      "model_text_confidence": "...",
      "error_codes": ["E3"],
      "error_codes_confidence": "...",
      "has_images": true,
      "brand_hint": "Carrier",
      "brand_hint_confidence": "...",
    }}
  ]
}}


"""),
    ("human", "Reddit JSON data follows:\n```json\n{json_data}\n```")
])

# --------------------------
# Chain Builder
# --------------------------
def build_data_labeler_chain() -> RunnableSequence:
    """Build the complete research analysis chain"""
    llm = build_llm()
    
    # Create the chain: prompt -> llm
    chain = PROMPT | llm
    
    return chain

