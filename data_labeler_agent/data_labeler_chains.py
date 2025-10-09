# research_agent/research_chains.py
"""
LangChain pipeline: Reddit JSON → Skilled Trades Industry Analysis (Markdown).
- Takes Reddit data from skilled trades subreddits
- Uses Gemini 2.5 flash via langchain_google_genai
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
    """Build the Gemini 2.5 Flash model instance"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.25,
        google_api_key=GEMINI_API_KEY,
    )

# The exact prompt from the original file
LABELER_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system", """
Given a JSON array of reddit posts, select posts that describe a specific HVAC malfunction (equipment not working, degraded performance, error codes, leaks, trips, unusual noise/odors) that likely requires diagnosis or repair.

For each post
1. Label as BREAK or NON_BREAK
2. If BREAK, extract fields with conservative confidence (0.0–1.0). Use null when unsure:
   - asset_family: one of {rtu, split_ac, heat_pump, mini_split, furnace, air_handler, boiler, chiller, cooling_tower, controls, other}
   - subtype_form_factor: short descriptor (e.g., packaged RTU, ductless wall, cassette, ducted, gas furnace condensing)
   - brand: manufacturer string if present (e.g., Carrier)
   - model_text: the exact printed/typed model string(s) if present
   - model_family_slug: normalized family/series slug if confident; e.g., carrier.48tc.2015_2022
   - indoor_model_id: for splits (ID from label/text if present)
   - outdoor_model_id: for splits (ID from label/text if present)
   - symptoms: list of short, concrete symptoms/failure modes
   - error_codes: list of error codes if present (e.g., E3, 33)
   - has_images: boolean if post includes images
   For any field you cannot extract with confidence, set the field and its confidence to null.

Fields you may use per post:
{{title, body, score, num_comments, upvote_ratio}}

Decision rubric (apply all):
1. BREAK only if it mentions a concrete symptom or failure mode (e.g., no cooling, tripping breaker, fan won't spin)
2. Reject posts that are brand comparisons, purchase/quote questions, new installs, or non-HVAC topics
3. asset_family must be from the allowed set above; choose other when unclear
4. Prefer precision over recall; use null when unsure
5. IF NON_BREAK, only fill in id, break_label, and break_confidence; set other fields to null


Output format (strict JSON, no extra text):
{{
  "results": [
    {{
      "id": "<post id>",
      "break_label": "BREAK | NON_BREAK",
      "break_confidence": "0.0|0.2|0.4|0.51|0.65|0.8|0.9",
      "asset_family": "rtu|split_ac|heat_pump|mini_split|furnace|air_handler|boiler|chiller|cooling_tower|controls|other|null",
      "asset_family_confidence": "...",
      "subtype_form_factor": "...",
      "subtype_form_factor_confidence": "...",
      "brand": "Carrier",
      "brand_confidence": "...",
      "model_text": "IM-500X 2019",
      "model_text_confidence": "...",
      "model_family_slug": "carrier.48tc.2015_2022",
      "model_family_slug_confidence": "...",
      "indoor_model_id": "...",
      "indoor_model_id_confidence": "...",
      "outdoor_model_id": "...",
      "outdoor_model_id_confidence": "...",
      "symptoms": ["..."],
      "symptoms_confidence": "...",
      "error_codes": ["E3"],
      "error_codes_confidence": "...",
      "has_images": true
    }}
  ]
}}


"""),
    ("human", "Reddit JSON data follows:\n```json\n{json_data}\n```")
])

# The exact prompt from the original file
from langchain_core.prompts import ChatPromptTemplate

SOLUTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", 
"""
You are an HVAC technician tasked with diagnosing and repairing a system malfunction. 

Given a post with a system malfunction outlined, you are to use the provided comments json to determine the best solution to the problem. Use ONLY the provided post context and comments.
If evidence is weak, conflicting, unsafe, or not clearly tied to the symptoms, answer exactly: "No clear solution."
Do NOT invent facts or fixes.

VALIDITY RUBRIC (per comment)
Accept a comment as valid evidence only if ALL are true:
1) Concrete action: proposes a specific fix (e.g., "replace run capacitor", "clean flame sensor", "reseat pressure switch", "rewire C wire"). Pure diagnostics or vague advice is weaker.
2) Symptom alignment: the fix plausibly addresses the OP’s described symptoms; penalize mismatches/omissions of key symptoms.
3) Support signals: higher ups (score) is a stronger signal; give additional weight to OP confirmations/edits (“this fixed it”) and independent agreement from other commenters.
4) Specific & verifiable: mentions parts/tools/readings/error codes or gives minimal reasoning tying symptoms → fix.
5) Safe & consistent: no contradictions; no unsafe instructions.

DISQUALIFIERS (hard reject)
- Speculation with no concrete action (especially without OP confirmation)
- Sales/brand talk, jokes, rants, off-topic
- “Call a professional” with no actionable fix
- Dangerous instructions (e.g., bypassing safeties)

SELECTION 
- Consider the comments that best satisfy the rubric
- Group comments by the specific action they propose (normalize verbs/objects).
- Prefer, in order:
  1) Actions explicitly confirmed by the OP as helping or fixing the issue (unless unsafe or inconsistent).
  2) Actions with clear independent agreement from multiple commenters.
  3) Actions supported by specific, verifiable details (parts/tools/readings/error codes) and strong symptom alignment.
  4) Among otherwise similar actions, prefer those discussed in clearer, more substantive comments and with higher ups (qualitatively).
- If no single action clearly stands out under the rubric, return "No clear solution."

OUTPUT (STRICT JSON, no extra text)
{{
  "post_id": "string",
  "solution": {{
    "summary": "string (or 'No clear solution.')",
    "confidence": 0.0,               // conservative 0..1; lower when any doubt remains
    "evidence_count": 0,              // number of comments you actually relied on
    "title": "string"
  }},
  "raw_comments_used": [
    {{
      "reddit_id": "string",
      "text": "string",
      "ups": 0
    }}
  ]
}}

VALIDATION
- Return ONLY valid JSON (no markdown).
- Use ONLY provided content; do not fabricate IDs or text.
- If unclear or unsafe, answer "No clear solution."
"""
    ),
    ("human",
"""POST CONTEXT
post_id: {post_id}
title: {title}
user: {user_id}
problem_diagnosis: {problem_diagnosis}
COMMENTS (JSON array; each item MAY contain: id/reddit_id, body/text/content, ups/score)
{comments_json}

Now produce the STRICT JSON output specified above."""
    ),
])



# --------------------------
# Chain Builder
# --------------------------
def build_data_labeler_chain() -> RunnableSequence:
    """Build the complete research analysis chain"""
    llm = build_llm()
    
    # Create the chain: prompt -> llm
    chain = LABELER_PROMPT | llm
    
    return chain

def build_solution_labeler_chain() -> RunnableSequence:
    """Build the complete solution labeling chain"""
    llm = build_llm()
    
    # Create the chain: prompt -> llm
    chain = SOLUTION_PROMPT | llm
    
    return chain
