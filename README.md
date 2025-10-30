GenAI Chatbot (RAG + FastAPI)
Local FastAPI service that turns the GreatLearning financial RAG lab into a production‑ready workflow: chunked PDF indexing to Chroma, retrieval with metadata filters, and OpenAI‑backed answers using the lab prompts. Built for Windows 11 + IntelliJ with config‑first design and clear logs.

Highlights
Chunked PDF indexing with parent_id/chunk_id, year, filename, pages.

Persistent Chroma at data/chromadb with idempotent index and safe reindex.

Retrieval and LLM QA that mirror lab prompts; answers include citations and timings.

Action‑style endpoints grouped as /doc-indexing and /rag-search.

Robust logging and error_info for quick troubleshooting.

Quick start
Python 3.11+, create venv, install deps (FastAPI, Uvicorn, ChromaDB, OpenAI, PyPDF).

Ensure folders exist (auto on start): data/, data/documents/, data/chromadb/.

Run: uvicorn app.main:app --host 0.0.0.0 --port 8099 --reload

Health: GET /health

Configuration
app/config/app_config.py controls:

data_dir, documents_dir, chroma_dir, scripts_dir

openai_base_url, openai_api_key, openai_default_model
No .env in feature code; use AppConfig only.

Logging
Console and data/app.log show “Trying/Successfully/Failed” with lapse times:

file_index_lapse_time, retrieval_lapse_time, llm_lapse_time

file_error_info in API responses when failures occur.

Endpoints
/doc-indexing

GET /count → { collection_count_after }

GET /list_local_pdfs → { folder, pdfs[], files_count }

POST /index (multipart form-data: advisor_id, client_id, doc_type, file_version, strategy, file_type, document_id?, files)
→ { parent_id, file_name, chunks_indexed, collection_count_after, file_index_status, file_index_lapse_time, file_error_info? }

Idempotent: returns file_index_status=skipped if parent already exists.

POST /reindex (multipart: filename + metadata)
→ { parent_id, file_name, replaced_chunks, chunks_indexed, collection_count_after, file_index_status }

DELETE /delete/{id}
→ { id, scope: "single"|"parent", deleted, collection_count_after, message }

POST /save_metadata { id, metadata } → { id, metadata }

GET /get_metadata/{id} → { id, metadata }

/rag-search

GET /retrieve?query=...&n_results=8&advisor_id=&client_id=&doc_type=
→ { query, hits[{ id, parent_id, text, metadata }], retrieval_lapse_time, file_error_info? }

GET /get_full_text/{id} → { id, text, metadata }

POST /user_query { question, n_results, top_k_ctx }
→ { question, answer, citations[], retrieval_lapse_time, llm_lapse_time, file_llm_status, file_error_info? }

POST /user_query_debug { question, n_results, top_k_ctx }
→ { question, context_blocks[{ id, parent_id, snippet }], answer, citations[], llm_lapse_time, file_llm_status, file_error_info? }

POST /user_query_eval { questions[], n_results, top_k_ctx }
→ { results: [same shape as user_query] }

Usage flow
Index PDFs

POST /doc-indexing/index with file, or place PDF in data/documents then POST /doc-indexing/reindex with filename.

Validate

GET /doc-indexing/count, GET /rag-search/retrieve?query=...

Ask lab‑style questions

POST /rag-search/user_query with prompts like “What was Tesla’s total revenue in 2019?”

Use debug/eval endpoints for grounding and batch checks.

.gitignore
Exclude .venv, .idea, caches, coverage, logs, and data/chromadb. Keep PDFs as needed; toggle data/documents/*.pdf if you prefer not to version them.

Notes
Prompts in prompts/lab_prompts.py align with the GL lab for parity.

Parent delete removes all chunks and their metadata.

Error messages surface via file_error_info and logs for quick diagnosis.