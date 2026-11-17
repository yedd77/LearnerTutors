"""Application settings.

Two sources are combined:

* ``.env`` / environment variables (``Settings``): secrets, provider URLs and
  timeouts, the embedding model, storage paths and limits.
* ``storage/runtime_config.json``: the chat/image provider, model and
  Fernet-encrypted API keys that Moodle pushes via ``POST /admin/config``.
  ``get_settings()`` re-reads it whenever the file changes, so every worker
  process picks up a new model or key on its next request.
"""
from pathlib import Path
import json
import logging
import os
import threading
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
from cryptography.fernet import Fernet, InvalidToken

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
RUNTIME_CONFIG_PATH = ROOT / "storage" / "runtime_config.json"
logger = logging.getLogger(__name__)

# Relevance cut-offs measured per embedding model (provider, model). Distances are not
# comparable across embedding models, so unlisted models get no default threshold.
# text-embedding-3-small, 2026-11-16, 291 indexed chunks in 3 courses: on-topic best
# match 0.19-0.63, other-course topics 0.64-0.87, off-topic 0.77-0.94.
CALIBRATED_MAX_DISTANCE: dict[tuple[str, str], float] = {
    ("openai", "text-embedding-3-small"): 0.70,
}


class Settings(BaseSettings):
    """Configuration read from environment variables and ``.env`` (names are case-insensitive)."""
    app_name: str = "RAG Assistant API"
    app_env: str = "development"
    log_level: str = "INFO"
    log_prompts: bool = False
    # Shared Moodle-to-FastAPI secret. Source it only from the backend environment/.env.
    rag_api_key: str = Field(default="", validation_alias="RAG_API_KEY")

    # Text-generation selection is supplied by Moodle and persisted at runtime.
    llm_provider: str = Field(default="", validation_alias="MOODLE_RUNTIME_LLM_PROVIDER")
    llm_model: str = Field(default="", validation_alias="MOODLE_RUNTIME_LLM_MODEL")
    image_model: str = Field(default="", validation_alias="MOODLE_RUNTIME_IMAGE_MODEL")
    image_provider: str = Field(default="openai", validation_alias="MOODLE_RUNTIME_IMAGE_PROVIDER")
    reuse_chat_api_key: bool = Field(default=False, validation_alias="MOODLE_RUNTIME_REUSE_CHAT_API_KEY")
    image_openai_api_key: str = ""
    image_gemini_api_key: str = ""

    # Environment credentials remain as fallback for existing deployments.
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_embed_model: str = "text-embedding-3-small"
    openai_timeout_seconds: int = 120
    # Sampling temperature per provider; a negative value sends none (provider default).
    # A model that rejects the value (e.g. OpenAI reasoning models) is retried once
    # without it and then keeps using its own default (see llm_provider).
    openai_temperature: float = 0.5
    openai_max_completion_tokens: int = 0

    gemini_api_key: str = ""
    gemini_embed_model: str = "gemini-embedding-001"
    gemini_embed_dimensions: int = 768
    gemini_timeout_seconds: int = 180
    gemini_temperature: float = 0.5
    gemini_max_output_tokens: int = 0
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_timeout_seconds: int = 180
    ollama_temperature: float = 0.5

    # LangSmith tracing would send student questions and course content to a
    # third party, so it stays off unless explicitly enabled.
    langsmith_tracing: bool = False

    embedding_provider: str = "openai"
    embedding_model: str = ""

    chroma_path: str = "./storage/chroma"
    chroma_collection: str = "ragassistant_chunks"
    top_k: int = 5
    # Drop retrieved chunks whose cosine distance is above this value, so off-topic
    # questions get the no-context reply. Unset = use the calibrated default for the
    # active embedding model (see CALIBRATED_MAX_DISTANCE); 0 = disabled.
    retrieval_max_distance: float | None = None
    # RAG graph limits. Total retrieval attempts per question (first search plus
    # rewritten retries), hard-capped at 3; and the timeout for the auxiliary
    # classify/rewrite LLM calls, after which the graph falls back to defaults.
    rag_max_retrieval_attempts: int = 2
    rag_aux_timeout_seconds: float = 20
    retrieval_workers: int = 64
    chunk_size: int = 900
    chunk_overlap: int = 150
    max_upload_mb: int = 20
    max_pages: int = 300
    max_chunks: int = 2000
    tunnel_url: str = "http://localhost:8000"

    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def chroma_absolute_path(self) -> str:
        """Chroma directory as an absolute path (relative to the repo root), created if missing."""
        path = Path(self.chroma_path)
        if not path.is_absolute():
            path = ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    @property
    def resolved_retrieval_max_distance(self) -> float:
        """Effective relevance threshold; 0 means no filtering."""
        if self.retrieval_max_distance is not None:
            return max(0.0, self.retrieval_max_distance)
        key = (self.embedding_provider.lower(), self.resolved_embedding_model)
        return CALIBRATED_MAX_DISTANCE.get(key, 0.0)

    @property
    def resolved_embedding_model(self) -> str:
        """``EMBEDDING_MODEL`` if set, otherwise the provider's default embedding model."""
        if self.embedding_model.strip():
            return self.embedding_model.strip()
        defaults = {
            "openai": self.openai_embed_model,
            "gemini": self.gemini_embed_model,
            "ollama": self.ollama_embed_model,
        }
        return defaults.get(self.embedding_provider.lower(), "")


_TRACING_ENV_VARS = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")


def apply_tracing_policy(settings: "Settings | None" = None) -> None:
    """Force LangSmith tracing off unless LANGSMITH_TRACING is enabled in Settings.

    Call this at process start, before LangChain runs: langsmith caches the
    environment lookups.
    """
    settings = settings or Settings()
    if not settings.langsmith_tracing:
        for name in _TRACING_ENV_VARS:
            os.environ[name] = "false"


def _runtime_fernet() -> Fernet:
    """Build the Fernet cipher from RUNTIME_CONFIG_ENCRYPTION_KEY, or raise RuntimeError."""
    key = os.getenv("RUNTIME_CONFIG_ENCRYPTION_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "RUNTIME_CONFIG_ENCRYPTION_KEY is required to store provider credentials securely. "
            "Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )
    try:
        return Fernet(key.encode("ascii"))
    except (ValueError, UnicodeEncodeError) as exc:
        raise RuntimeError("RUNTIME_CONFIG_ENCRYPTION_KEY must be a valid Fernet key.") from exc


def encrypt_provider_secret(value: str) -> str:
    """Encrypt a provider API key for storage in the runtime config file."""
    return _runtime_fernet().encrypt(value.encode("utf-8")).decode("ascii")


def decrypt_provider_secret(value: str) -> str:
    """Decrypt a stored provider API key; RuntimeError if the encryption key is wrong."""
    try:
        return _runtime_fernet().decrypt(value.encode("ascii")).decode("utf-8")
    except (InvalidToken, ValueError, UnicodeEncodeError) as exc:
        raise RuntimeError(
            "Stored provider credential cannot be decrypted. Check RUNTIME_CONFIG_ENCRYPTION_KEY."
        ) from exc


# Runtime-config fields Moodle may override, and where its encrypted keys go.
_RUNTIME_FIELDS = {"llm_provider", "llm_model", "image_model", "image_provider", "reuse_chat_api_key"}
_RUNTIME_KEY_FIELDS = (
    # Moodle-managed chat credentials take precedence over environment values.
    ("provider_api_keys", "openai", "openai_api_key"),
    ("provider_api_keys", "gemini", "gemini_api_key"),
    # Image credentials are stored separately from chat credentials.
    ("image_provider_api_keys", "openai", "image_openai_api_key"),
    ("image_provider_api_keys", "gemini", "image_gemini_api_key"),
)


def _runtime_overrides(values: Any) -> dict[str, Any]:
    """Turn the parsed runtime config into Settings overrides, decrypting stored keys."""
    if not isinstance(values, dict):
        raise ValueError("Runtime configuration must be a JSON object.")
    overrides = {key: value for key, value in values.items() if key in _RUNTIME_FIELDS}
    for section, provider, field in _RUNTIME_KEY_FIELDS:
        stored = values.get(section, {})
        encrypted = stored.get(provider) if isinstance(stored, dict) else None
        if encrypted:
            overrides[field] = decrypt_provider_secret(encrypted)
    return overrides


_settings_lock = threading.Lock()
_settings_cache: Settings | None = None
_settings_cache_mtime: int | None = None


def get_settings() -> Settings:
    """Load runtime settings and notice updates made by any FastAPI worker.

    Cached until the runtime config file's mtime changes. If the file cannot be
    read or decrypted, the error is logged and ``.env`` values are used alone.
    """
    global _settings_cache, _settings_cache_mtime
    try:
        current_mtime = RUNTIME_CONFIG_PATH.stat().st_mtime_ns
    except OSError:
        current_mtime = None

    with _settings_lock:
        if _settings_cache is not None and current_mtime == _settings_cache_mtime:
            return _settings_cache

        settings = Settings()
        if RUNTIME_CONFIG_PATH.is_file():
            try:
                values = json.loads(RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
                overrides = _runtime_overrides(values)
                if overrides:
                    settings = settings.model_copy(update=overrides)
            except (OSError, ValueError, TypeError, InvalidToken, RuntimeError):
                # A bad encryption key should be visible to operators. Keep the
                # service process alive so health and authentication diagnostics work.
                logger.exception("Could not load runtime configuration.")

        _settings_cache = settings
        _settings_cache_mtime = current_mtime
        return settings
