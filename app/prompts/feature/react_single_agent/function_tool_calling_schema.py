# from typing import Dict, Any, List
# import json
# from openai import OpenAI
# from app.utils.app_logging import get_logger
# from app.config.app_config import AppConfig
# from app.adapters.feature.react_single_agent.tool_adapters import RetrievalTools
# from app.prompts.feature.react_single_agent import react_tool_calling_schema as react_prompts
# from app.prompts.feature.react_single_agent import function_tool_calling_schema as function_prompts
# from app.prompts.feature.react_single_agent import react_tool_calling_schema as tools_schema
#
# cfg = AppConfig()
# logger = get_logger(cfg)
# client = OpenAI(api_key=cfg.openai_api_key, base_url=cfg.openai_base_url)
# model = cfg.openai_llm_model
#
# def _to_kwargs(maybe_json: Any) -> Dict[str, Any]:
#     if isinstance(maybe_json, dict):
#         return maybe_json
#     if isinstance(maybe_json, str):
#         s = maybe_json.strip()
#         if not s:
#             return {}
#         try:
#             return json.loads(s)
#         except Exception:
#             return {}
#     return {}
#
# def _call_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
#     if name == "vector_search":
#         return RetrievalTools.vector_search(**args)
#     if name == "get_chunk":
#         return RetrievalTools.get_chunk(**args)
#     return {"error": f"unknown tool {name}"}
#
# class FunctionCallingAgent:
#     """Provider function-calling over shared tools using AS-IS prompts and tool schemas."""
#
#     def run(self, question: str) -> Dict[str, Any]:
#         logger.info("[Functions] begin q='%s'", question)
#         messages: List[Dict[str, Any]] = [
#             {"role":"system","content":function_prompts.FUNCTION_SYSTEM},
#             {"role":"user","content":function_prompts.FUNCTION_USER_TEMPLATE.format(question=question)}
#         ]
#         first = client.chat.completions.create(
#             model=model,
#             messages=messages,
#             tools=[{"type":"function","function": f} for f in tools_schema.FUNCTION_TOOLS],
#             tool_choice="auto",
#             temperature=0.2, top_p=1.0, max_tokens=256
#         )
#
#         tool_results: List[Dict[str, Any]] = []
#         citations: List[str] = []
#
#         assistant_msg = first.choices[0].message
#         messages.append({"role":"assistant","content":assistant_msg.content or "", "tool_calls": assistant_msg.tool_calls or []})
#
#         for tc in assistant_msg.tool_calls or []:
#             name = tc.function.name
#             args = _to_kwargs(tc.function.arguments)
#             logger.info("[Functions] tool_call name=%s args=%s", name, args)
#             result = _call_tool(name, args)
#             tool_results.append({"name": name, "args": args, "result_keys": list(result.keys())})
#
#             if name == "vector_search":
#                 for h in result.get("hits", [])[:3]:
#                     pid = h.get("parent_id")
#                     if pid and pid not in citations:
#                         citations.append(pid)
#             elif name == "get_chunk":
#                 pid = (result.get("metadata") or {}).get("parent_id")
#                 if pid and pid not in citations:
#                     citations.append(pid)
#
#             messages.append({"role":"tool","tool_call_id": tc.id, "name": name, "content": str(result)[:3500]})
#
#         final = client.chat.completions.create(model=model, messages=messages, temperature=0.2, top_p=1.0, max_tokens=512)
#         answer = final.choices[0].message.content or ""
#         logger.info("[Functions] done citations=%s", citations[:3])
#         return {"question": question, "answer": answer, "citations": citations[:5], "tool_results": tool_results}
