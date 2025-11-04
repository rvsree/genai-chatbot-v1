from typing import Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel
from app.utils.app_logging import get_logger
from app.config.app_config import AppConfigSingleton as AppConfig
from app.service.feature.react_single_agent.react_service import ReactAgent
from app.service.feature.react_single_agent.functions_service import FunctionCallingAgent
from app.service.feature.react_single_agent.mermaid_service import react_mermaid, functions_mermaid

cfg = AppConfig()
logger = get_logger(cfg)
router = APIRouter(prefix="/react-single-agent", tags=["react-single-agent"])

react_agent = ReactAgent(max_steps=4, emit_traces=True)
func_agent = FunctionCallingAgent(emit_traces=True)

class AskRequest(BaseModel):
    question: str

def _fallback(question: str, diagram: Optional[str], endpoint: str, title: str) -> dict:
    out = {
        "question": question,
        "answer": "",
        "citations": [],
        "tool_results": [],
        "file_llm_status": "failed",
        "file_error_info": {"stage": "react_single_agent", "type": "NullResponse", "message": "service returned null/empty payload"}
    }
    if diagram:
        out.update({
            "diagram_title": title,
            "diagram_scope": "react_single_agent",
            "endpoint_name": endpoint,
            "mermaid_text": diagram
        })
    return out

@router.post("/react/ask")
async def react_ask(req: AskRequest, include_diagram: bool = Query(False), emit_traces: bool = Query(True)):
    logger.info("[RSA] /react/ask q='%s'", req.question)
    try:
        out = react_agent.run(req.question, emit_traces=emit_traces)
        if not out:
            return _fallback(req.question, react_mermaid() if include_diagram else None, "/react-single-agent/react/ask", "ReAct Agent Flow")
        # Enforce client flag: remove traces when false
        if not emit_traces and "traces" in out:
            out.pop("traces", None)
        if include_diagram:
            out.update({
                "diagram_title": "ReAct Agent Flow",
                "diagram_scope": "react_single_agent",
                "endpoint_name": "/react-single-agent/react/ask",
                "mermaid_text": react_mermaid()
            })
        return out
    except Exception as e:
        logger.exception("[RSA] react_ask failed: %s", e)
        fb = _fallback(req.question, react_mermaid() if include_diagram else None, "/react-single-agent/react/ask", "ReAct Agent Flow")
        fb["file_error_info"] = {"stage":"react-ask","type":e.__class__.__name__,"message":str(e)}
        return fb

@router.post("/functions/ask")
async def functions_ask(req: AskRequest, include_diagram: bool = Query(False), emit_traces: bool = Query(True)):
    logger.info("[RSA] /functions/ask q='%s'", req.question)
    try:
        out = func_agent.run(req.question, emit_traces=emit_traces)
        if not out:
            return _fallback(req.question, functions_mermaid() if include_diagram else None, "/react-single-agent/functions/ask", "Function-Calling Flow")
        # Enforce client flag: remove tool_results when false
        if not emit_traces and "tool_results" in out:
            out.pop("tool_results", None)
        if include_diagram:
            out.update({
                "diagram_title": "Function-Calling Flow",
                "diagram_scope": "react_single_agent",
                "endpoint_name": "/react-single-agent/functions/ask",
                "mermaid_text": functions_mermaid()
            })
        return out
    except Exception as e:
        logger.exception("[RSA] functions_ask failed: %s", e)
        fb = _fallback(req.question, functions_mermaid() if include_diagram else None, "/react-single-agent/functions/ask", "Function-Calling Flow")
        fb["file_error_info"] = {"stage":"functions-ask","type":e.__class__.__name__,"message":str(e)}
        return fb
