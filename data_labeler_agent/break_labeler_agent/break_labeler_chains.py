# data_labeler_agent/break_labeler_chains.py
"""
Break Labeler Chain: Classifies Reddit posts as BREAK/NON_BREAK and extracts
structured HVAC malfunction details (asset_family, brand, model, symptoms, etc.).
"""

import os
import sys
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

# Import the Pydantic schema and parser
try:
    from .break_labeler_schema import parser, BreakOutput
except ImportError:
    from break_labeler_schema import parser, BreakOutput

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


LABELER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You label individual Reddit posts about HVAC. Decide if a post describes a concrete malfunction requiring diagnosis/repair.

Rules:
- Output STRICT JSON only (no prose).
- Prefer precision over recall. If unsure => NON_BREAK.
- Extract ONLY from the post (title/body/metadata). Don't invent.

BREAK criteria (need >=1):
- Concrete symptom/failure (no cooling/heat, trips, leak, won't start, unusual noise/odor, error code).
- Measured abnormality implying malfunction (pressures/temps/amps).

NON_BREAK examples:
- Shopping/quotes, brand opinions, install pics w/o issues, non-HVAC.

Fields available per post: title, body, score, num_comments, upvote_ratio, media(if present).

Glossary:
- asset_family: rtu | split_ac | heat_pump | mini_split | furnace | air_handler | boiler | chiller | cooling_tower | controls | tools | other | "" (empty string for unknown)
- subtype_form_factor: physical/config form (e.g., packaged RTU, ductless wall, cassette, horizontal)
- brand: manufacturer token as written (Carrier, Trane, Goodman)
- model_text: verbatim model strings from text (no normalization)
- model_family_slug: normalized brand+series (optional year bucket) ONLY if confident (e.g., carrier.48tc.2015_2022); else ""
- indoor_model_id / outdoor_model_id: for split/mini-split/heat-pump systems (verbatim IDs). For RTUs/furnaces, leave "".
- error_codes: controller codes/tokens (E5, 33, LPS)
- has_images: true if the post contains or explicitly references images
- quote_spans: 0-based [start,end) offsets into title/body supporting BREAK or a key field. Format: [{{"field":"title|body","start":int,"end":int}}]

Type rules (CRITICAL - must follow exactly):
- Confidences are floats 0.0-1.0 (NOT strings, NOT null).
- Arrays => [] when none (NOT null).
- Strings => "" when unknown (NOT null).
- Booleans => true/false (NOT null).
- break_label: MUST be exactly "BREAK" or "NON_BREAK"
- asset_family: MUST be one of the allowed values OR empty string ""
- All confidence fields MUST be present with float values 0.0-1.0

{format_instructions}

Output (one object per input post):
{{
  "results": [
    {{
      "id": "<post id>",
      "break_label": "BREAK" | "NON_BREAK",
      "break_confidence": 0.0,
      "asset_family": "",
      "asset_family_confidence": 0.0,
      "subtype_form_factor": "",
      "subtype_form_factor_confidence": 0.0,
      "brand": "",
      "brand_confidence": 0.0,
      "model_text": "",
      "model_text_confidence": 0.0,
      "model_family_slug": "",
      "model_family_slug_confidence": 0.0,
      "indoor_model_id": "",
      "indoor_model_id_confidence": 0.0,
      "outdoor_model_id": "",
      "outdoor_model_id_confidence": 0.0,
      "symptoms": [],
      "symptoms_confidence": 0.0,
      "error_codes": [],
      "error_codes_confidence": 0.0,
      "has_images": false,
      "quote_spans": []
    }}
  ]
}}
"""),
    ("human", "Reddit JSON data follows:\n```json\n{json_data}\n```")
])


def build_data_labeler_chain() -> RunnableSequence:
    """
    Build the BREAK/NON_BREAK labeling chain with Pydantic schema enforcement
    
    Returns:
        RunnableSequence that outputs validated BreakOutput objects
    """
    llm = build_llm()
    chain = LABELER_PROMPT | llm
    return chain
