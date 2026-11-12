from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
import os

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    app_name: str = "RAG Assistant API"
    app_env: str = "development"
    log_level: str = "INFO"
    log_prompts: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    rag_api_key: str = os.getenv("RAG_API_KEY", "replace-with-a-long-random-secret")

    # Text-generation selection is supplied by Moodle and persisted at runtime.
    llm_provider: str = ""
    llm_model: str = ""
    image_model: str = ""

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_embed_model: str = "text-embedding-3-small"
    openai_timeout_seconds: int = 120
    openai_temperature: float = 0
    openai_max_completion_tokens: int = 0

    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_embed_model: str = "gemini-embedding-001"
    gemini_embed_dimensions: int = 768  # 0 = use the model's default size
    gemini_timeout_seconds: int = 180
    # Set to a negative value to omit temperature and use the model default.
    gemini_temperature: float = 0.2
    # Optional cap on generated tokens; 0 means no cap.
    gemini_max_output_tokens: int = 0
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_timeout_seconds: int = 180

    # Embedding selection is independent from chat generation. The same
    # provider/model is used for both indexing and query embeddings.
    embedding_provider: str = "openai"
    embedding_model: str = ""

    chroma_path: str = "./storage/chroma"
    chroma_collection: str = "ragassistant_chunks"
    top_k: int = 5
    # Dedicated worker pool for synchronous Chroma and embedding retrieval.
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

@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if RUNTIME_CONFIG_PATH.is_file():
        try:
            import json
            values = json.loads(RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
            allowed = {
                "llm_provider", "llm_model", "image_model",
                "embedding_provider", "embedding_model",
                "gemini_embed_dimensions",
            }
            overrides = {key: value for key, value in values.items() if key in allowed}
            if overrides:
                settings = settings.model_copy(update=overrides)
        except (OSError, ValueError, TypeError):
            # Invalid persisted configuration must not prevent the API starting.
            pass
    return settings
