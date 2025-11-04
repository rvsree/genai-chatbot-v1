from __future__ import annotations
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field
import os

class FeatureFlags(BaseModel):
    react_single_agent_enabled: bool = True
    functions_enabled: bool = True
    manager_llm: Optional[str] = None
    function_calling_llm: Optional[str] = None

class DBProviderConfig(BaseModel):
    name: str
    driver: str
    dsn: str
    user: Optional[str] = None
    password: Optional[str] = None
    pool_size: int = 5
    pool_timeout_sec: int = 30
    enabled: bool = False

class AppConfig(BaseModel):
    project_root: str = Field(default_factory=lambda: str(Path(__file__).resolve().parents[2]))
    data_dir: str = Field(default_factory=lambda: "data")
    documents_dir: str = Field(default_factory=lambda: str(Path("data") / "documents"))
    chroma_dir: str = Field(default_factory=lambda: str(Path("data") / "chromadb"))
    scripts_dir: str = Field(default_factory=lambda: str(Path("data") / "scripts"))

    log_file: str = Field(default_factory=lambda: str(Path("data") / "app.log"))
    log_level: str = "INFO"

    USE_ENV_FOR_OPENAI: bool = False  # flip to True to read from environment

    if USE_ENV_FOR_OPENAI:
        openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
        openai_base_url: str = Field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", ""))
        openai_llm_model: str = Field(default_factory=lambda: os.getenv("OPENAI_LLM_MODEL", ""))
        openai_embed_model: str = Field(default_factory=lambda: os.getenv("OPENAI_EMBEDDED_MODEL", ""))
        # Back-compat alias
        openai_default_model: str = Field(default_factory=lambda: os.getenv("OPENAI_LLM_MODEL", ""))
    else:
        # NWP hardcoded values (local only). Replace locally; never commit real keys.
        openai_api_key: str = "gl-U2FsdGVkX19/1rWLMPzQgQLCBVcwv1fzxGQFz5sAB5JDk1s0cKq5rkl9hOZVF8ur"
        openai_base_url: str = "https://aibe.mygreatlearning.com/openai/v1"
        openai_llm_model: str = "gpt-4o-mini"
        openai_embed_model: str = "text-embedding-3-small"
        # Back-compat alias
        openai_default_model: str = "gpt-4o-mini"

    feature_flags: FeatureFlags = Field(default_factory=FeatureFlags)
    db_providers: List[DBProviderConfig] = Field(default_factory=list)

    auth_required: bool = False
    rate_limit_per_minute: Optional[int] = None

    def ensure_dirs(self) -> None:
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.documents_dir).mkdir(parents=True, exist_ok=True)
        Path(self.chroma_dir).mkdir(parents=True, exist_ok=True)
        Path(self.scripts_dir).mkdir(parents=True, exist_ok=True)

    def get_db_provider(self, name: str) -> Optional[DBProviderConfig]:
        for p in self.db_providers:
            if p.name.lower() == name.lower() and p.enabled:
                return p
        return None

_cfg: Optional[AppConfig] = None

def AppConfigSingleton() -> AppConfig:
    global _cfg
    if _cfg is None:
        _cfg = AppConfig()
        _cfg.ensure_dirs()
    return _cfg

# Backwards-compatible helper for legacy imports and new call sites.
def ensure_data_dirs(cfg: Optional[AppConfig] = None) -> None:
    """
    Create required data folders. Accepts an optional cfg for convenience,
    but will fallback to the singleton if not provided.
    """
    if cfg is None:
        cfg = AppConfigSingleton()
    cfg.ensure_dirs()
