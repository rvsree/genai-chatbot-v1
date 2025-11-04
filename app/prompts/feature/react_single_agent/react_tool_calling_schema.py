# # AS-IS function tool descriptions (names, descriptions, parameters) from the sample
# FUNCTION_TOOLS = [
#     {
#         "name": "vector_search",
#         "description": "Search the indexed filings and return top chunks with metadata.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "query": {"type":"string"},
#                 "n_results": {"type":"integer","default":5},
#                 "advisor_id": {"type":"string"},
#                 "client_id": {"type":"string"},
#                 "doc_type": {"type":"string"}
#             },
#             "required": ["query"]
#         }
#     },
#     {
#         "name": "get_chunk",
#         "description": "Fetch a specific chunk by id and return full text and metadata.",
#         "parameters": {
#             "type": "object",
#             "properties": {"id": {"type":"string"}},
#             "required": ["id"]
#         }
#     }
# ]