# app/models/rag_models.py
# Full, stable models: includes all legacy imports, no per-iteration 'plan', action is in retrieval_plan.

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# ---------- Search/Indexing (legacy-safe) ----------
class RetrieveResponseHit(BaseModel):
    id: str
    parent_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None

class RetrieveResponse(BaseModel):
    query: str
    hits: List[RetrieveResponseHit] = Field(default_factory=list)

class RetrievedChunk(BaseModel):
    id: str
    parent_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None

# ---------- Simple QA (legacy-safe) ----------
class QARequest(BaseModel):
    question: str

class QAResponse(BaseModel):
    question: str
    answer: str
    citations: List[str] = Field(default_factory=list)

# ---------- Agent request ----------
class RAGQueryRequest(BaseModel):
    question: str
    emit_traces: bool = True
    enable_query_variants: bool = False
    enable_output_scoring: bool = False
    max_variants: int = 3
    scoring_model: str = "heuristic_v1"
    self_reflection_iterations: int = 3
    execution_mode: str = "async"  # "async" | "sequential"

# ---------- Shared ----------
class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ErrorInfo(BaseModel):
    stage: str
    type: str
    message: str

class IterationRetrieval(BaseModel):
    query: str
    hits: int
    top_parent_ids: List[str] = Field(default_factory=list)
    action: Optional[str] = None              # "vector_search" | "get_chunk" | "index_lookup"
    tool_name: Optional[str] = None           # "retrieval.vector_search", etc.
    source_name: Optional[str] = None         # "chroma", "snowflake", etc.
    tool_latency_ms: Optional[int] = None

class IterationRecord(BaseModel):
    iteration: int
    thought: Optional[str] = None
    retrieval_plan: List[IterationRetrieval] = Field(default_factory=list)
    output: Optional[str] = None
    actual_score: Optional[float] = None
    llm_call: Optional[Dict[str, Any]] = None
    error_info: Optional[ErrorInfo] = None

class VariantScoreSummary(BaseModel):
    candidate_id: str
    actual_score: float

class VariantRecord(BaseModel):
    variant_id: str
    query_variant: str
    query_context: Dict[str, Any] = Field(default_factory=dict)
    sub_questions: List[str] = Field(default_factory=list)
    data_source_routing: List[Dict[str, Any]] = Field(default_factory=list)
    iterations: List[IterationRecord] = Field(default_factory=list)
    variant_score: Optional[VariantScoreSummary] = None
    self_reflection: Optional[Dict[str, Any]] = None

class AgentDescriptor(BaseModel):
    agent_id: str
    agent_name: str
    agent_role: Optional[str] = None
    agent_goal: Optional[str] = None

class RAGAnswer(BaseModel):
    run_id: str
    agent_graph_id: Optional[str] = None
    question: str
    answer: str
    citations: List[str] = Field(default_factory=list)

    scoring_model: Optional[str] = None
    selected_variant_id: Optional[str] = None
    selected_candidate_id: Optional[str] = None
    selected_score: Optional[float] = None

    # Parent-level LLM summary
    llm_call_status: Optional[str] = None            # "success" | "mock" | "error"
    llm_call_error_info: Optional[str] = None        # error text if any
    llm_provider: Optional[str] = None               # e.g., "openai"
    llm_model: Optional[str] = None                  # e.g., "gpt-4o-mini"
    llm_provider_fallback: Optional[str] = None      # when mock used
    llm_model_fallback: Optional[str] = None         # when mock used

    agents: List[Dict[str, Any]] = Field(default_factory=list)
    variants: List[Dict[str, Any]] = Field(default_factory=list)
    ranking_rationale: Optional[str] = None
    answer_timestamp: Optional[str] = None
    elapsed_time: Optional[int] = None
    latency_ms: Optional[int] = None
    token_usage: Optional[Dict[str, int]] = None
    cache_hit: Optional[bool] = None
    error_info: Optional[Dict[str, Any]] = None
