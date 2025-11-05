# app/router/feature/react_single_agent/react_tool_router.py
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.service.feature.react_single_agent.react_service import ReactToolCallingAgent
from app.service.feature.react_single_agent.base.react_base import AgentError

router = APIRouter()
agent = ReactToolCallingAgent()

@router.post("/rag/react-single-agent/tool-calling/ask")
async def react_tool_calling(payload: dict):
    try:
        return await agent.run(
            question=payload["question"],
            scoring_model=payload.get("scoring_model","heuristic_v1"),
            emit_traces=payload.get("emit_traces", True),
            enable_query_variants=payload.get("enable_query_variants", True),
            enable_output_scoring=payload.get("enable_output_scoring", True),
            max_variants=payload.get("max_variants", 3),
            self_reflection_iterations=payload.get("self_reflection_iterations", 3),
            agent_graph_id=payload.get("agent_graph_id","react-single-agent"),
            agent_descriptor={
                "agent_id":"react-single-agent",
                "agent_name":"Financial Filings Analyst",
                "agent_role":"researcher",
                "agent_goal":"Retrieve and verify filing facts."
            },
            execution_mode="async",
            preferred_year=payload.get("preferred_year"),
            top_k=payload.get("top_k"),
            retrieval_filters=payload.get("retrieval_filters")
        )
    except AgentError as e:
        return JSONResponse(status_code=e.http_status, content={
            "error_code": e.code,
            "message": e.message,
            "details": e.details
        })
