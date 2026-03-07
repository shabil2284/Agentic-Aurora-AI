from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class BoundingBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float

class Token(BaseModel):
    text: str
    bbox: BoundingBox
    confidence: float = 1.0

class Line(BaseModel):
    text: str
    bbox: BoundingBox
    tokens: List[Token] = []

class Block(BaseModel):
    type: str  # text, table, figure, etc.
    text: Optional[str] = ""
    bbox: BoundingBox
    lines: List[Line] = []
    metadata: Dict[str, Any] = {}

class Page(BaseModel):
    page_number: int
    width: float
    height: float
    blocks: List[Block] = []
    metadata: Dict[str, Any] = {}

class Document(BaseModel):
    filename: str
    pages: List[Page] = []
    metadata: Dict[str, Any] = {}
