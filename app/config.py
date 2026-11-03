from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent

# Application settings.
class Settings(BaseSettings):
    app_name: str = "RAG Assistant API"
    app_env: str = "development"
    log_level: str = "INFO"
    log_prompts: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    rag_api_key: str = "replace-with-a-long-random-secret" # The API key for the RAG Assistant
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_chat_model: str = "gemini-3.6-flash"
    gemini_temperature : float = 0.2
    gemini_embed_model: str = "gemini-embedding-001"
    gemini_embed_dimensions: int = 768  # 0 = use the model's default size
    gemini_timeout_seconds: int = 180
    # Set to a negative value to omit temperature and use the model default.
    gemini_temperature: float = 0.2
    # Optional cap on generated tokens; 0 means no cap.
    gemini_max_output_tokens: int = 0
    chroma_path: str = "./storage/chroma"
    chroma_collection: str = "ragassistant_chunks"
    top_k: int = 5
    chunk_size: int = 900
    chunk_overlap: int = 150
    max_upload_mb: int = 20
    max_pages: int = 300
    max_chunks: int = 2000
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # Return the absolute path to the Chroma storage directory, create if not exist
    @property
    def chroma_absolute_path(self) -> str:
        path = Path(self.chroma_path)
        if not path.is_absolute():
            path = ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

# Get the application settings, cached for performance.
@lru_cache
def get_settings() -> Settings:
    return Settings()