# data_labeler_agent/solution_labeler_chains.py
"""
Solution Labeler Chain: Analyzes Reddit post comments to identify the best
solution for an HVAC malfunction based on evidence strength and alignment.
"""

import os
import sys
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
        import importlib.util
        spec = importlib.util.spec_from_file_location("local_secrets", os.path.join(parent_dir, "local_secrets.py"))
        local_secrets = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(local_secrets)
        GEMINI_API_KEY = local_secrets.GEMINI_API_KEY
    except:
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in local_secrets.py or environment variables")


def build_llm() -> ChatGoogleGenerativeAI:
    """Build the Gemini 2.5 Flash model instance"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.25,
        google_api_key=GEMINI_API_KEY,
    )


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
2) Symptom alignment: the fix plausibly addresses the OP's described symptoms; penalize mismatches/omissions of key symptoms.
3) Support signals: higher ups (score) is a stronger signal; give additional weight to OP confirmations/edits ("this fixed it") and independent agreement from other commenters.
4) Specific & verifiable: mentions parts/tools/readings/error codes or gives minimal reasoning tying symptoms → fix.
5) Safe & consistent: no contradictions; no unsafe instructions.

DISQUALIFIERS (hard reject)
- Speculation with no concrete action (especially without OP confirmation)
- Sales/brand talk, jokes, rants, off-topic
- "Call a professional" with no actionable fix
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
  "solution_report": {{
    "summary": "string",                 // "" when unknown or "No clear solution." if unclear
    "steps": ["string"],                 // [] when none
    "parts_needed": [                     // [] when none
      {{"name": "string"}}
    ],
    "evidence_refs": ["comment_id"],    // [] when none; ids only
    "confidence": 0.0                    // conservative 0..1; lower when any doubt remains
  }}
}}

VALIDATION
- Return ONLY valid JSON (no markdown).
- Use ONLY provided content; do not fabricate IDs or text.
- If unclear or unsafe, set summary to "No clear solution.", keep arrays empty, and confidence low.
- Follow type conventions: strings => "" when unknown; arrays => []; floats 0.0-1.0.
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


def build_solution_labeler_chain() -> RunnableSequence:
    """Build the solution labeling chain"""
    llm = build_llm()
    chain = SOLUTION_PROMPT | llm
    return chain

