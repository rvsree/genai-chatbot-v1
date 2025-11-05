from fastapi import APIRouter
from pydantic import BaseModel
from app.utils.app_logging import get_logger
from app.config.app_config import AppConfig, AppConfigSingleton
from app.service.feature.react_agent.react_service import ReactAgent
from app.service.feature.react_agent.functions_service import FunctionCalling
from typing import Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel
from app.router.feature.react_agent.react_mermaid import _react_agent_mermaid, _functions_calling_mermaid

cfg = AppConfigSingleton.instance()
logger = get_logger(cfg)
react_router = APIRouter(prefix="/rag/react-agent", tags=["/rag/react-agent"])

react_agent = ReactAgent(max_steps=4)
func_agent = FunctionCalling()

class AskRequest(BaseModel):
    question: str

# @react_router.post("/langchain_meomory/ask")
# async def react_ask(req: AskRequest, include_diagram: Optional[bool] = Query(False)):
#     out = react_agent.run(req.question)
#     if include_diagram:
#         out.update({
#             "diagram_title": "ReAct Agent Flow",
#             "diagram_scope": "feature_react_functions",
#             "endpoint_name": "/v2/react/ask",
#             "mermaid_text": _react_agent_mermaid()
#         })
#     return out
#
# @react_router.post("/functions_calling/ask")
# async def functions_ask(req: AskRequest, include_diagram: Optional[bool] = Query(False)):
#     out = func_agent.run(req.question)
#     if include_diagram:
#         out.update({
#             "diagram_title": "Function-Calling Flow",
#             "diagram_scope": "feature_react_functions",
#             "endpoint_name": "/v2/functions/ask",
#             "mermaid_text": _functions_calling_mermaid()
#         })

class AskRequest(BaseModel):
    question: str

def _fallback(payload: dict, diagram: Optional[str], endpoint: str, title: str) -> dict:
    out = {
        "question": payload.get("question", ""),
        "answer": "",
        "citations": [],
        "tool_results": [],
        "file_llm_status": "failed",
        "file_error_info": {"stage": "react-functions", "type": "NullResponse", "message": "service returned null/empty payload"}
    }
    if diagram:
        out.update({
            "diagram_title": title,
            "diagram_scope": "feature_react_functions",
            "endpoint_name": endpoint,
            "mermaid_text": diagram
        })
    return out

@react_router.post("/langchain_meomory/ask")
async def react_ask(req: AskRequest, include_diagram: Optional[bool] = Query(False)):
    logger.info("[V2] /react/ask q='%s'", req.question)
    try:
        out = react_agent.run(req.question)
        if not out:
            return _fallback(req.model_dump(), _react_agent_mermaid() if include_diagram else None, "/rag/react-agent/react/ask", "ReAct Agent Flow")
        if include_diagram:
            out.update({
                "diagram_title": "ReAct Agent Flow",
                "diagram_scope": "feature_react_functions",
                "endpoint_name": "/rag/react-agent/react/ask",
                "mermaid_text": _react_agent_mermaid()
            })
        return out
    except Exception as e:
        logger.exception("[V2] react_ask failed: %s", e)
        fb = _fallback(req.model_dump(), _react_agent_mermaid() if include_diagram else None, "/rag/react-agent/react/ask", "ReAct Agent Flow")
        fb["file_error_info"] = {"stage":"react-ask","type":e.__class__.__name__,"message":str(e)}
        return fb

@react_router.post("/functions_calling/ask")
async def functions_ask(req: AskRequest, include_diagram: Optional[bool] = Query(False)):
    logger.info("[V2] /functions/ask q='%s'", req.question)
    try:
        out = func_agent.run(req.question)
        if not out:
            return _fallback(req.model_dump(), _functions_calling_mermaid() if include_diagram else None, "/rag/react-agent/functions_calling/ask", "Function-Calling Flow")
        if include_diagram:
            out.update({
                "diagram_title": "Function-Calling Flow",
                "diagram_scope": "feature_react_functions",
                "endpoint_name": "/rag/react-agent/functions_calling/ask",
                "mermaid_text": _functions_calling_mermaid()
            })
        return out
    except Exception as e:
        logger.exception("[V2] functions_ask failed: %s", e)
        fb = _fallback(req.model_dump(), _functions_calling_mermaid() if include_diagram else None, "/rag/react-agent/functions_calling/ask", "Function-Calling Flow")
        fb["file_error_info"] = {"stage":"functions-ask","type":e.__class__.__name__,"message":str(e)}
        return fb