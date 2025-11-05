# app/service/chunked_indexer_service.py
from typing import List, Dict, Any
from app.config.vector_db_client import VectorDBClient

class ChunkedIndexerService:
    def __init__(self, db: VectorDBClient):
        self.db = db

    def upsert_chunks(self, chunks: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> int:
        self.db.upsert_items(chunks, metadatas, ids)
        return len(ids)

    def reindex_parent(self, parent_id: str, chunks: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> int:
        self.db.delete_by_parent(parent_id)
        self.db.upsert_items(chunks, metadatas, ids)
        return len(ids)

    def purge_parent(self, parent_id: str) -> int:
        return self.db.delete_by_parent(parent_id)

    def count(self) -> int:
        return self.db.count()


# from pathlib import Path
# from typing import Dict, Any, List, Tuple
# from app.utils.app_logging import get_logger
# from app.config.app_config import AppConfig, AppConfigSingleton
# from app.service.chroma_client_service import ChromaClientService
# from app.utils.pdf_text_extract import extract_text_from_pdf
# from app.utils.doc_chunking import sliding_window_chunks
#
# cfg = AppConfigSingleton.instance()
# logger = get_logger(cfg)
#
# class ChunkedIndexerService:
#     def __init__(self, chroma_service: ChromaClientService | None = None):
#         self.chroma = chroma_service or ChromaClientService()
#
#     def _year_from_filename(self, name: str) -> str:
#         for y in ("2019","2020","2021","2022","2023","2024","2025"):
#             if y in name:
#                 return y
#         return ""
#
#     def index_pdf_path(self, path: Path, base_meta: Dict[str, Any],
#                        chunk_size: int = 900, overlap: int = 150) -> Tuple[str, int]:
#         logger.info("[ChunkedIndexer] Extract text path=%s", path)
#         text, pages = extract_text_from_pdf(path)
#         chunks = sliding_window_chunks(text, size=chunk_size, overlap=overlap)
#         parent_id = base_meta.get("document_id") or path.stem
#         year = self._year_from_filename(path.name)
#
#         texts: List[str] = []
#         metas: List[Dict[str, Any]] = []
#         ids: List[str] = []
#         for i, ch in enumerate(chunks):
#             chunk_id = f"{parent_id}::chunk::{i:04d}"
#             meta = dict(base_meta)
#             meta.update({
#                 "parent_id": parent_id,
#                 "chunk_id": chunk_id,
#                 "filename": path.name,
#                 "pages": pages,
#                 "year": year
#             })
#             texts.append(ch)
#             metas.append(meta)
#             ids.append(chunk_id)
#
#         logger.info("[ChunkedIndexer] Upserting chunks n=%d parent=%s", len(ids), parent_id)
#         self.chroma.upsert_items(texts, metas, ids)
#         logger.info("[ChunkedIndexer] Upsert complete parent=%s", parent_id)
#         return parent_id, len(chunks)
