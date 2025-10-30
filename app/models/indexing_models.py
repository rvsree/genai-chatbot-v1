from typing import Optional, Dict, Any
from pydantic import BaseModel

class IndexRequest(BaseModel):
    advisor_id: str
    client_id: str
    doc_type: str
    file_version: str
    strategy: str
    file_type: str
    document_id: Optional[str] = None  # filename stem if not provided

class IndexResponse(BaseModel):
    parent_id: str
    file_name: str
    file_version: str
    file_type: str
    files_count: int
    chunks_indexed: int
    existing_chunks: Optional[int] = None
    replaced_chunks: Optional[int] = None
    collection_count_after: int
    file_index_status: str           # success | skipped | failed
    file_llm_status: str             # not_applicable | success | failed
    file_index_lapse_time: float
    file_error_info: Optional[Dict[str, Any]] = None

class ReindexRequest(BaseModel):
    filename: str
    advisor_id: str
    client_id: str
    doc_type: str
    file_version: str
    strategy: str
    file_type: str = "pdf"
    document_id: Optional[str] = None

class DeleteResponse(BaseModel):
    id: str
    scope: str                       # parent | single
    deleted: int
    collection_count_after: int
    message: str

class SaveMetadataRequest(BaseModel):
    id: str
    metadata: Dict[str, Any]

class SaveMetadataResponse(BaseModel):
    id: str
    metadata: Dict[str, Any]
