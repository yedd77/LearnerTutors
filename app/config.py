from functools import lru_cache
from pathlib import Path
import json
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
from cryptography.fernet import Fernet, InvalidToken

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    app_name: str = "RAG Assistant API"
    app_env: str = "development"
    log_level: str = "INFO"
    log_prompts: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    # Shared Moodle-to-FastAPI secret. Source it only from the backend environment/.env.
    rag_api_key: str = Field(default="", validation_alias="RAG_API_KEY")

    # Text-generation selection is supplied by Moodle and persisted at runtime.
    llm_provider: str = Field(default="", validation_alias="MOODLE_RUNTIME_LLM_PROVIDER")
    llm_model: str = Field(default="", validation_alias="MOODLE_RUNTIME_LLM_MODEL")
    image_model: str = Field(default="", validation_alias="MOODLE_RUNTIME_IMAGE_MODEL")

    # Environment credentials remain as fallback for existing deployments.
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_embed_model: str = "text-embedding-3-small"
    openai_timeout_seconds: int = 120
    openai_temperature: float = 0
    openai_max_completion_tokens: int = 0

    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_embed_model: str = "gemini-embedding-001"
    gemini_embed_dimensions: int = 768
    gemini_timeout_seconds: int = 180
    gemini_temperature: float = 0.2
    gemini_max_output_tokens: int = 0
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_timeout_seconds: int = 180

    embedding_provider: str = "openai"
    embedding_model: str = ""

    chroma_path: str = "./storage/chroma"
    chroma_collection: str = "ragassistant_chunks"
    top_k: int = 5
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
        path = Path(self.chroma_path)
        if not path.is_absolute():
            path = ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    @property
    def resolved_embedding_model(self) -> str:
        if self.embedding_model.strip():
            return self.embedding_model.strip()
        defaults = {
            "openai": self.openai_embed_model,
            "gemini": self.gemini_embed_model,
            "ollama": self.ollama_embed_model,
        }
        return defaults.get(self.embedding_provider.lower(), "")


RUNTIME_CONFIG_PATH = ROOT / "storage" / "runtime_config.json"


def _runtime_fernet() -> Fernet:
    key = __import__("os").getenv("RUNTIME_CONFIG_ENCRYPTION_KEY", "").strip()
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
    return _runtime_fernet().encrypt(value.encode("utf-8")).decode("ascii")


def decrypt_provider_secret(value: str) -> str:
    try:
        return _runtime_fernet().decrypt(value.encode("ascii")).decode("utf-8")
    except (InvalidToken, ValueError, UnicodeEncodeError) as exc:
        raise RuntimeError(
            "Stored provider credential cannot be decrypted. Check RUNTIME_CONFIG_ENCRYPTION_KEY."
        ) from exc


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if RUNTIME_CONFIG_PATH.is_file():
        try:
            values = json.loads(RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
            allowed = {"llm_provider", "llm_model", "image_model"}
            overrides = {key: value for key, value in values.items() if key in allowed}
            secrets = values.get("provider_api_keys", {})
            if isinstance(secrets, dict):
                # Moodle-managed credentials take precedence over environment values.
                for provider, field in (("openai", "openai_api_key"), ("gemini", "gemini_api_key")):
                    encrypted = secrets.get(provider)
                    if encrypted:
                        overrides[field] = decrypt_provider_secret(encrypted)
            if overrides:
                settings = settings.model_copy(update=overrides)
        except (OSError, ValueError, TypeError, InvalidToken, RuntimeError):
            # Do not silently hide a credential decryption error: it must be visible
            # to operators, but must not prevent health/admin routes from loading.
            import logging
            logging.getLogger(__name__).exception("Could not load runtime configuration.")
    return settings
