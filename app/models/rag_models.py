from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class RetrieveResponseHit(BaseModel):
    id: str
    parent_id: Optional[str] = None
    text: str
    metadata: Optional[Dict[str, Any]] = None

class RetrieveResponse(BaseModel):
    query: str
    hits: List[RetrieveResponseHit]
    retrieval_lapse_time: float
    file_error_info: Optional[Dict[str, Any]] = None

class QARequest(BaseModel):
    question: str
    n_results: int = 8
    top_k_ctx: int = 4

class QAResponse(BaseModel):
    question: str
    answer: str
    citations: List[str]
    retrieval_lapse_time: float
    llm_lapse_time: float
    file_llm_status: str
    file_error_info: Optional[Dict[str, Any]] = None
