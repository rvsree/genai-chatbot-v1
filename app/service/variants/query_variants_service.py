# app/services/query_variants_service.py
# Deterministic variant generator (unchanged interface, supports future multi-agent use)

from typing import List
from app.utils.app_logging import get_logger
from app.config.app_config import AppConfigSingleton as AppConfig, AppConfigSingleton

cfg = AppConfigSingleton.instance()
logger = get_logger(cfg)

class QueryVariantsService:
    @staticmethod
    def generate(question: str, max_variants: int = 3) -> List[str]:
        logger.info("[Variants] generate max=%d q='%s'", max_variants, question)
        q = (question or "").strip()
        if not q: return []
        out = []
        low = q.lower()
        if "2019" in q and len(out) < max_variants:
            out.append(q.replace("2019", "FY 2019"))
        if "total revenue" in low and len(out) < max_variants:
            out.append(q.replace("total revenue", "revenue (total)"))
        if "latest filing" in low and len(out) < max_variants:
            out.append(q.replace("latest filing", "most recent annual report"))
        # ensure unique
        seen = set([q]); uniq = []
        for v in out:
            if v not in seen:
                seen.add(v); uniq.append(v)
        logger.info("[Variants] produced=%d", len(uniq))
        return uniq[:max_variants]
