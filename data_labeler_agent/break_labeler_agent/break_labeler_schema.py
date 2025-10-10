from typing import List, Literal, Optional
from pydantic import BaseModel, Field, conlist, confloat
from langchain.output_parsers import PydanticOutputParser

AssetFamily = Literal[
    "rtu","split_ac","heat_pump","mini_split","furnace","air_handler",
    "boiler","chiller","cooling_tower","controls","tools","other",""
]

class QuoteSpan(BaseModel):
    field: Literal["title","body"] = Field(..., description="Which text field the span indexes")
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)

class ErrorReport(BaseModel):
    """Problem description and classification"""
    break_label: Literal["BREAK","NON_BREAK"]
    break_confidence: confloat(ge=0.0, le=1.0)
    symptoms: List[str] = Field(default_factory=list)
    symptoms_confidence: confloat(ge=0.0, le=1.0) = 0.0
    error_codes: List[str] = Field(default_factory=list)
    error_codes_confidence: confloat(ge=0.0, le=1.0) = 0.0
    quote_spans: List[QuoteSpan] = Field(default_factory=list)

class SystemInfo(BaseModel):
    """Model/system identification and tagging"""
    asset_family: AssetFamily = ""
    asset_family_confidence: confloat(ge=0.0, le=1.0) = 0.0
    asset_subtype: str = ""
    asset_subtype_confidence: confloat(ge=0.0, le=1.0) = 0.0
    brand: str = ""
    brand_confidence: confloat(ge=0.0, le=1.0) = 0.0
    model_text: str = ""
    model_text_confidence: confloat(ge=0.0, le=1.0) = 0.0
    model_family_id: str = ""
    model_family_id_confidence: confloat(ge=0.0, le=1.0) = 0.0
    indoor_model_id: str = ""
    indoor_model_id_confidence: confloat(ge=0.0, le=1.0) = 0.0
    outdoor_model_id: str = ""
    outdoor_model_id_confidence: confloat(ge=0.0, le=1.0) = 0.0
    model_resolution_confidence: confloat(ge=0.0, le=1.0) = 0.0
    has_images: bool = False

class BreakItem(BaseModel):
    id: str
    error_report: ErrorReport
    system_info: SystemInfo

class BreakOutput(BaseModel):
    results: conlist(BreakItem, min_items=1)

parser = PydanticOutputParser(pydantic_object=BreakOutput)
# You can add parser.get_format_instructions() to the prompt if you prefer structured guidance.
