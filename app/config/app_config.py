# app/config/app_config.py
# Restored, previously-working explicit AppConfig with safe singleton accessor.
# Do NOT modify callers; both AppConfigSingleton.instance() and AppConfigSingleton() are supported.

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import os

@dataclass
class AppConfig:
    project_root: str
    data_dir: str
    documents_dir: str
    chroma_dir: str
    scripts_dir: str
    log_file: str
    log_level: str

    USE_ENV_FOR_OPENAI: bool
    openai_api_key: str
    openai_base_url: str
    openai_llm_model: str
    openai_embed_model: str
    openai_default_model: str

    feature_flags: Dict[str, bool] = field(default_factory=dict)
    db_providers: Dict[str, Any] = field(default_factory=dict)
    auth_required: bool = False
    rate_limit_per_minute: int = 60

class AppConfigSingleton:
    _instance: Optional[AppConfig] = None

    @classmethod
    def instance(cls) -> AppConfig:
        if cls._instance is None:
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            data = os.path.join(root, "data")
            documents = os.path.join(data, "documents")
            chroma = os.path.join(data, "chromadb")
            scripts = os.path.join(data, "scripts")
            logs_dir = os.path.join(root, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            log_file = os.path.join(logs_dir, "app.log")

            use_env = os.getenv("USE_ENV_FOR_OPENAI", "false").lower() == "true"
            # Preserve historical behavior: api_key pulled from env either way
            # api_key = os.getenv("OPENAI_API_KEY", "")
            # base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            # llm_model = os.getenv("OPENAI_LLM_MODEL", "")
            # embed_model = os.getenv("OPENAI_EMBED_MODEL", "")
            # default_model = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")

            api_key: str = "gl-U2FsdGVkX19/1rWLMPzQgQLCBVcwv1fzxGQFz5sAB5JDk1s0cKq5rkl9hOZVF8ur"
            base_url: str = "https://aibe.mygreatlearning.com/openai/v1"
            llm_model: str = "gpt-4o-mini"
            embed_model: str = "text-embedding-3-small"
            default_model: str = "gpt-4o-mini"

            cls._instance = AppConfig(
                project_root=root,
                data_dir=data,
                documents_dir=documents,
                chroma_dir=chroma,
                scripts_dir=scripts,
                log_file=log_file,
                log_level=os.getenv("LOG_LEVEL", "INFO"),
                USE_ENV_FOR_OPENAI=use_env,
                openai_api_key=api_key,
                openai_base_url=base_url,
                openai_llm_model=llm_model,
                openai_embed_model=embed_model,
                openai_default_model=default_model,
                feature_flags={
                    "react_variants": True,
                    "output_scoring": True,
                },
                db_providers={
                    "postgres": {"enabled": os.getenv("POSTGRES_ENABLED", "false").lower() == "true"},
                    "snowflake": {"enabled": os.getenv("SNOWFLAKE_ENABLED", "false").lower() == "true"}
                },
                auth_required=os.getenv("AUTH_REQUIRED", "false").lower() == "true",
                rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
            )
        return cls._instance

    # Legacy style: AppConfigSingleton() returns the instance to support old call sites (e.g., cfg = AppConfig())
    def __new__(cls):
        return cls.instance()


# from __future__ import annotations
# from pathlib import Path
# from typing import Optional, List
# from pydantic import BaseModel, Field
# import os
# from dataclasses import dataclass
#
#
# class FeatureFlags(BaseModel):
#     react_single_agent_enabled: bool = True
#     functions_enabled: bool = True
#     manager_llm: Optional[str] = None
#     function_calling_llm: Optional[str] = None
#
# class DBProviderConfig(BaseModel):
#     name: str
#     driver: str
#     dsn: str
#     user: Optional[str] = None
#     password: Optional[str] = None
#     pool_size: int = 5
#     pool_timeout_sec: int = 30
#     enabled: bool = False
#
# @dataclass
# class AppConfig(BaseModel):
#     project_root: str = Field(default_factory=lambda: str(Path(__file__).resolve().parents[2]))
#     data_dir: str = Field(default_factory=lambda: "data")
#     documents_dir: str = Field(default_factory=lambda: str(Path("data") / "documents"))
#     chroma_dir: str = Field(default_factory=lambda: str(Path("data") / "chromadb"))
#     scripts_dir: str = Field(default_factory=lambda: str(Path("data") / "scripts"))
#
#     log_file: str = Field(default_factory=lambda: str(Path("data") / "app.log"))
#     log_level: str = "INFO"
#
#     USE_ENV_FOR_OPENAI: bool = False  # flip to True to read from environment
#
#     if USE_ENV_FOR_OPENAI:
#         openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
#         openai_base_url: str = Field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", ""))
#         openai_llm_model: str = Field(default_factory=lambda: os.getenv("OPENAI_LLM_MODEL", ""))
#         openai_embed_model: str = Field(default_factory=lambda: os.getenv("OPENAI_EMBEDDED_MODEL", ""))
#         # Back-compat alias
#         openai_default_model: str = Field(default_factory=lambda: os.getenv("OPENAI_LLM_MODEL", ""))
#     else:
#         # NWP hardcoded values (local only). Replace locally; never commit real keys.
#         openai_api_key: str = "gl-U2FsdGVkX19/1rWLMPzQgQLCBVcwv1fzxGQFz5sAB5JDk1s0cKq5rkl9hOZVF8ur"
#         openai_base_url: str = "https://aibe.mygreatlearning.com/openai/v1"
#         openai_llm_model: str = "gpt-4o-mini"
#         openai_embed_model: str = "text-embedding-3-small"
#         # Back-compat alias
#         openai_default_model: str = "gpt-4o-mini"
#
#     feature_flags: FeatureFlags = Field(default_factory=FeatureFlags)
#     db_providers: List[DBProviderConfig] = Field(default_factory=list)
#
#     auth_required: bool = False
#     rate_limit_per_minute: Optional[int] = None
#
#     def ensure_dirs(self) -> None:
#         Path(self.data_dir).mkdir(parents=True, exist_ok=True)
#         Path(self.documents_dir).mkdir(parents=True, exist_ok=True)
#         Path(self.chroma_dir).mkdir(parents=True, exist_ok=True)
#         Path(self.scripts_dir).mkdir(parents=True, exist_ok=True)
#
#     def get_db_provider(self, name: str) -> Optional[DBProviderConfig]:
#         for p in self.db_providers:
#             if p.name.lower() == name.lower() and p.enabled:
#                 return p
#         return None
#
# _cfg: Optional[AppConfig] = None
#
# def AppConfigSingleton() -> AppConfig:
#     global _cfg
#     if _cfg is None:
#         _cfg = AppConfig()
#         _cfg.ensure_dirs()
#     return _cfg
#
# # Backwards-compatible helper for legacy imports and new call sites.
# def ensure_data_dirs(cfg: Optional[AppConfig] = None) -> None:
#     """
#     Create required data folders. Accepts an optional cfg for convenience,
#     but will fallback to the singleton if not provided.
#     """
#     if cfg is None:
#         cfg = AppConfigSingleton()
#     cfg.ensure_dirs()
#
#
# class AppConfigSingleton:
#     _instance = None
#
#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = AppConfig()
#         return cls._instance
