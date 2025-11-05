from fastapi import APIRouter
from pydantic import BaseModel
from app.utils.app_logging import get_logger
from app.config.app_config import AppConfig, AppConfigSingleton
from app.service.feature.react_single_agent.mermaid_service import react_mermaid, functions_mermaid

cfg = AppConfigSingleton.instance()
logger = get_logger(cfg)
router = APIRouter(prefix="/mermaid", tags=["mermaid-diagrams"])

class MermaidRequest(BaseModel):
    feature_name: str
    feature_endpoint_name: str

class MermaidResponse(BaseModel):
    diagram_title: str
    diagram_scope: str
    feature_name: str
    endpoint_name: str
    mermaid_text: str

@router.post("/react-single-agent", response_model=MermaidResponse)
async def mermaid_react_single_agent(req: MermaidRequest):
    logger.info("[Mermaid] feature=%s endpoint=%s", req.feature_name, req.feature_endpoint_name)
    return {
        "diagram_title": "ReAct + Function-Calling (React Single Agent)",
        "diagram_scope": "react_single_agent",
        "feature_name": req.feature_name,
        "endpoint_name": req.feature_endpoint_name,
        "mermaid_text": react_mermaid()
    }
