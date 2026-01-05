from pydantic import BaseModel
from typing import List, Optional

class Point(BaseModel):
    x: float  # normalized 0-1
    y: float  # normalized 0-1

class LineDrawing(BaseModel):
    start: Point
    end: Point
    label: str
    confidence: Optional[float] = None

class ReferenceLine(BaseModel):
    start: Point
    end: Point
    label: str
    id: Optional[str] = None

class LineDetectionRequest(BaseModel):
    input_image_base64: str
    reference_image_base64: str
    product_type: Optional[str] = None
    expected_line_count: Optional[int] = None
    line_types: Optional[List[str]] = None  # ["Width", "Height", "Depth"]
    reference_lines: Optional[List[ReferenceLine]] = None  # NEW: Labeled reference lines
    model: Optional[str] = "gemini"  # "gemini" or "seedream"

class LineDetectionResponse(BaseModel):
    lines: List[LineDrawing]
    method_used: str  # "image_generation", "direct_prediction", "hybrid", "feature_matching"
    processing_time: float
    confidence_score: Optional[float] = None
    generated_image_base64: Optional[str] = None  # Base64 encoded generated image
    input_image_base64: Optional[str] = None  # Base64 encoded resized input image (1024x1024)

