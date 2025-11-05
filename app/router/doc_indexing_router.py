from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi import Path as FPath
from pathlib import Path
import time
from typing import Optional, Dict, Any, List

from app.config.app_config import AppConfigSingleton
from app.utils.app_logging import get_logger
from app.config.vector_db_client import VectorDBClient  # corrected import path
from app.service.indexing.chunked_indexer_service import ChunkedIndexerService
from app.models.indexing_models import (
    IndexResponse, SaveMetadataRequest, SaveMetadataResponse, DeleteResponse
)

cfg = AppConfigSingleton.instance()
logger = get_logger(cfg)

# Ensure documents directory exists
Path(cfg.documents_dir).mkdir(parents=True, exist_ok=True)

# Initialize vector DB client and indexer
vdb = VectorDBClient(backend="chroma")
indexer = ChunkedIndexerService(vdb)

indexing_router = APIRouter(prefix="/doc-indexing", tags=["doc-indexing"])

def _safe_filename(name: str) -> str:
    base = "".join(c for c in (name or "") if c.isalnum() or c in ("-", "_", ".")).strip()
    return base or "upload.pdf"

def _validate_single_file_part(request: Request) -> Optional[str]:
    ctype = request.headers.get("content-type", "")
    if "multipart/form-data" not in ctype:
        return "Expected multipart/form-data; Content-Type incorrect or missing boundary"
    return None

# --------- Helper wrappers over VectorDBClient to mirror older chromaClientService API ---------

def _collection_count() -> int:
    try:
        return vdb.count()
    except Exception as e:
        logger.error("[Indexing] count() failed: %s", e)
        return 0

def _get_ids_by_parent(parent_id: str) -> List[str]:
    try:
        res = vdb.get(where={"parent_id": parent_id})
        ids = res.get("ids", [[]])
        return ids[0] if ids and isinstance(ids[0], list) else (ids or [])
    except Exception as e:
        logger.error("[Indexing] get_ids_by_parent failed: %s", e)
        return []

def _delete_by_parent(parent_id: str) -> int:
    try:
        return vdb.delete(where={"parent_id": parent_id})
    except Exception as e:
        logger.error("[Indexing] delete_by_parent failed: %s", e)
        return 0

def _delete_single(id: str) -> int:
    try:
        return vdb.delete(ids=[id])
    except Exception as e:
        logger.error("[Indexing] delete_single failed: %s", e)
        return 0

def _save_metadata(id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return vdb.update_metadata(id=id, metadata=metadata)
    except Exception as e:
        logger.error("[Indexing] save_metadata failed: %s", e)
        raise

def _get_by_id(id: str) -> Dict[str, Any]:
    try:
        return vdb.get(ids=[id])
    except Exception as e:
        logger.error("[Indexing] get_by_id failed: %s", e)
        return {"ids": [], "metadatas": []}

# ----------------------------------- Routes -----------------------------------

@indexing_router.get("/count")
async def count() -> dict:
    c = _collection_count()
    logger.info("[Indexing] Count=%d", c)
    return {"collection_count_after": c}

@indexing_router.get("/list_local_pdfs")
async def list_local_pdfs() -> dict:
    base = Path(cfg.documents_dir)
    base.mkdir(parents=True, exist_ok=True)
    pdfs = [p.name for p in base.glob("*.pdf")]
    logger.info("[Indexing] List PDFs n=%d dir=%s", len(pdfs), base)
    return {"folder": str(base), "pdfs": pdfs, "files_count": len(pdfs)}

@indexing_router.post("/index", response_model=IndexResponse)
async def index(
        request: Request,
        advisor_id: str = Form(...),
        client_id: str = Form(...),
        doc_type: str = Form(...),
        file_version: str = Form(...),
        strategy: str = Form(...),
        file_type: str = Form(...),
        document_id: Optional[str] = Form(None),
        files: UploadFile = File(..., description="PDF file")
):
    start = time.perf_counter()
    logger.info("[Indexing] Begin index advisor=%s client=%s doc_type=%s", advisor_id, client_id, doc_type)
    prelim_error = _validate_single_file_part(request)
    if prelim_error:
        logger.error("[Indexing] Multipart error: %s", prelim_error)
        return IndexResponse(
            parent_id=document_id or "", file_name="", file_version=file_version, file_type=file_type,
            files_count=0, chunks_indexed=0, collection_count_after=_collection_count(),
            file_index_status="failed", file_llm_status="not_applicable", file_index_lapse_time=0.0,
            file_error_info={"stage":"doc-indexing","type":"MultipartError","message":prelim_error}
        )

    try:
        raw_bytes = await files.read()
        if not raw_bytes:
            logger.error("[Indexing] Empty file stream for 'files'")
            return IndexResponse(
                parent_id=document_id or "", file_name=files.filename or "", file_version=file_version, file_type=file_type,
                files_count=0, chunks_indexed=0, collection_count_after=_collection_count(),
                file_index_status="failed", file_llm_status="not_applicable", file_index_lapse_time=0.0,
                file_error_info={"stage":"doc-indexing","type":"EmptyFile","message":"No file content received for 'files' (duplicate keys?)"}
            )

        file_name = _safe_filename(files.filename or "upload.pdf")
        dest = Path(cfg.documents_dir) / file_name
        dest.write_bytes(raw_bytes)
        logger.info("[Indexing] File saved path=%s size=%d", dest, len(raw_bytes))

        parent_id = (document_id or Path(file_name).stem)
        existing_ids = _get_ids_by_parent(parent_id)
        if existing_ids:
            lapse = round((time.perf_counter() - start) * 1000, 2)
            logger.info("[Indexing] Skipped existing parent=%s chunks=%d", parent_id, len(existing_ids))
            return IndexResponse(
                parent_id=parent_id, file_name=file_name, file_version=file_version, file_type=file_type,
                files_count=1, chunks_indexed=0, existing_chunks=len(existing_ids),
                collection_count_after=_collection_count(), file_index_status="skipped",
                file_llm_status="not_applicable", file_index_lapse_time=lapse,
                file_error_info={"stage":"doc-indexing","type":"AlreadyIndexed","message":"parent_id exists; use /doc-indexing/reindex"}
            )

        base_meta = {
            "advisor_id": advisor_id, "client_id": client_id, "doc_type": doc_type,
            "version": file_version, "strategy": strategy, "file_type": file_type,
            "parent_id": parent_id,  # normalized key used across pipeline
            "document_id": parent_id
        }
        new_parent_id, chunks = indexer.index_pdf_path(dest, base_meta)
        # some indexers may rewrite parent id; prefer returned value
        parent_id = new_parent_id or parent_id

        lapse = round((time.perf_counter() - start) * 1000, 2)
        logger.info("[Indexing] Success parent=%s chunks=%d lapse_ms=%.2f", parent_id, chunks, lapse)

        return IndexResponse(
            parent_id=parent_id, file_name=file_name, file_version=file_version, file_type=file_type,
            files_count=1, chunks_indexed=chunks, collection_count_after=_collection_count(),
            file_index_status="success", file_llm_status="not_applicable", file_index_lapse_time=lapse
        )
    except Exception as e:
        lapse = round((time.perf_counter() - start) * 1000, 2)
        logger.exception("[Indexing] Failed: %s lapse_ms=%.2f", e, lapse)
        return IndexResponse(
            parent_id=document_id or "", file_name=files.filename or "", file_version=file_version, file_type=file_type,
            files_count=1, chunks_indexed=0, collection_count_after=_collection_count(),
            file_index_status="failed", file_llm_status="not_applicable", file_index_lapse_time=lapse,
            file_error_info={"stage":"doc-indexing","type":e.__class__.__name__,"message":str(e)}
        )

@indexing_router.post("/reindex", response_model=IndexResponse)
async def reindex(
        filename: str = Form(...), advisor_id: str = Form(...), client_id: str = Form(...),
        doc_type: str = Form(...), file_version: str = Form(...), strategy: str = Form(...),
        file_type: str = Form("pdf"), document_id: Optional[str] = Form(None)
):
    start = time.perf_counter()
    logger.info("[Indexing] Begin reindex filename=%s", filename)
    try:
        path = Path(cfg.documents_dir) / _safe_filename(filename)
        if not path.exists():
            logger.error("[Indexing] Reindex file not found: %s", path)
            raise HTTPException(status_code=404, detail="File not found in data/documents")

        parent_id = (document_id or path.stem)
        base_meta = {
            "advisor_id": advisor_id, "client_id": client_id, "doc_type": doc_type,
            "version": file_version, "strategy": strategy, "file_type": file_type,
            "parent_id": parent_id,
            "document_id": parent_id
        }

        removed = _delete_by_parent(parent_id)
        logger.info("[Indexing] Reindex removed parent=%s chunks=%d", parent_id, removed)
        new_parent_id, chunks = indexer.index_pdf_path(path, base_meta)
        parent_id = new_parent_id or parent_id

        lapse = round((time.perf_counter() - start) * 1000, 2)
        logger.info("[Indexing] Reindex success parent=%s chunks=%d lapse_ms=%.2f", parent_id, chunks, lapse)

        return IndexResponse(
            parent_id=parent_id, file_name=path.name, file_version=file_version, file_type=file_type,
            files_count=1, replaced_chunks=removed, chunks_indexed=chunks, collection_count_after=_collection_count(),
            file_index_status="success", file_llm_status="not_applicable", file_index_lapse_time=lapse
        )
    except HTTPException:
        raise
    except Exception as e:
        lapse = round((time.perf_counter() - start) * 1000, 2)
        logger.exception("[Indexing] Reindex failed: %s lapse_ms=%.2f", e, lapse)
        return IndexResponse(
            parent_id=document_id or "", file_name=filename, file_version=file_version, file_type=file_type,
            files_count=1, chunks_indexed=0, collection_count_after=_collection_count(),
            file_index_status="failed", file_llm_status="not_applicable", file_index_lapse_time=lapse,
            file_error_info={"stage":"doc-indexing","type":e.__class__.__name__,"message":str(e)}
        )

@indexing_router.delete("/delete/{id}", response_model=DeleteResponse)
async def delete(id: str = FPath(...)):
    try:
        if "::chunk::" in id:
            logger.info("[Indexing] Delete single id=%s", id)
            deleted = _delete_single(id)
            msg = "Deleted 1 record" if deleted else "No record found for given id"
            logger.info("[Indexing] Delete single result=%s", msg)
            return DeleteResponse(
                id=id, scope="single", deleted=deleted, collection_count_after=_collection_count(), message=msg
            )

        logger.info("[Indexing] Delete by parent id=%s", id)
        removed = _delete_by_parent(id)
        msg = "Deleted all chunks for parent" if removed else "No chunks found for given parent"
        logger.info("[Indexing] Delete by parent result=%s (removed=%d)", msg, removed)
        return DeleteResponse(
            id=id, scope="parent", deleted=removed, collection_count_after=_collection_count(), message=msg
        )
    except Exception as e:
        logger.exception("[Indexing] Delete failed id=%s: %s", id, e)
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")

@indexing_router.post("/save_metadata", response_model=SaveMetadataResponse)
async def save_metadata(req: SaveMetadataRequest):
    updated = _save_metadata(req.id, req.metadata or {})
    logger.info("[Indexing] Save metadata ok id=%s", req.id)
    return SaveMetadataResponse(id=req.id, metadata=updated)

@indexing_router.get("/get_metadata/{id}")
async def get_metadata(id: str = FPath(...)) -> dict:
    res = _get_by_id(id)
    ids = res.get("ids", [[]])
    if not ids or (isinstance(ids[0], list) and not ids[0]) or (not isinstance(ids[0], list) and not ids):
        raise HTTPException(status_code=404, detail="Document id not found")
    # metadatas shape may be [ [ {..} ] ] or [ {..} ]
    metas = res.get("metadatas", [[]])
    metadata = metas[0][0] if (metas and isinstance(metas[0], list) and metas[0]) else (metas[0] if metas else {})
    return {"id": id, "metadata": metadata}
