from pydantic import BaseModel, Field
from typing import List, Optional

class DocumentIn(BaseModel):
    id: Optional[str] = None
    text: str = Field(..., min_length=1)
    metadata: dict = Field(default_factory=dict)

class UpsertResponse(BaseModel):
    added: int
    total_count: int

class SearchQuery(BaseModel):
    query: str
    k: int = 5
