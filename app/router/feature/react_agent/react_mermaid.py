from fastapi import APIRouter
from pydantic import BaseModel
from app.utils.app_logging import get_logger
from app.config.app_config import AppConfig, AppConfigSingleton

cfg = AppConfigSingleton.instance()
logger = get_logger(cfg)
react_mermaid_router = APIRouter(prefix="/react_mermaid", tags=["react_mermaid_diagrams"])

class MermaidRequest(BaseModel):
    feature_name: str
    feature_endpoint_name: str

class MermaidResponse(BaseModel):
    diagram_title: str
    diagram_scope: str
    feature_name: str
    endpoint_name: str
    mermaid_text: str

def _react_agent_mermaid() -> str:
    return """flowchart TD
    U[User] -->|POST /react/ask| R[ReAct Agent]
    R --> T1[Tool Select]
    T1 -->|vector_search| VS[RetrievalTools.vector_search]
    T1 -->|index_lookup| IL[RetrievalTools.index_lookup]
    T1 -->|get_chunk| GC[RetrievalTools.get_chunk]
    VS --> C1[Citations/Context]
    GC --> C2[Context+Meta]
    R --> OA[OpenAI Chat (final turn)]
    C1 --> OA
    C2 --> OA
    OA --> A[Answer+citations]
    """

def _functions_calling_mermaid() -> str:
    return """flowchart TD
    U[User] -->|POST /functions/ask| F[Function-Calling]
    F --> OA1[OpenAI Chat (tool_calls:auto)]
    OA1 -->|tool_calls| VS[RetrievalTools.vector_search]
    VS --> TM1[tool message]
    TM1 --> OA2[OpenAI Chat (final)]
    OA1 -->|optional get_chunk| GC[RetrievalTools.get_chunk]
    GC --> TM2[tool message]
    TM2 --> OA2
    OA2 --> A[Answer+citations]
    """

def _render(feature: str, endpoint: str) -> str:
    key = f"{feature}:{endpoint}".lower()
    if "react" in key:
        return _react_agent_mermaid()
    if "functions" in key:
        return _functions_calling_mermaid()
    # fallback to a simple pass-through
    return """flowchart TD
    U[User] --> EP[Feature Endpoint]
    EP --> Core[Shared Core Services]
    Core --> A[Answer]
    """

@react_mermaid_router.post("/react-agent", response_model=MermaidResponse)
async def mermaid_react_agent(req: MermaidRequest):
    logger.info("[Mermaid] render feature=%s endpoint=%s", req.feature_name, req.feature_endpoint_name)
    text = _render(req.feature_name, req.feature_endpoint_name)
    return MermaidResponse(
        diagram_title="ReAct + Functions: React-Agent",
        diagram_scope="feature_react_functions",
        feature_name=req.feature_name,
        endpoint_name=req.feature_endpoint_name,
        mermaid_text=text
    )
