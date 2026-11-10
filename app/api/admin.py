import json
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.config import RUNTIME_CONFIG_PATH, get_settings
from app.security import RequireApiKey
from app.services.vector_store import reset_vector_store_cache

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[RequireApiKey])


class RuntimeConfigRequest(BaseModel):
    llm_provider: Literal["openai", "gemini", "ollama"] = "openai"
    llm_model: str = Field(default="gpt-4o-luna", min_length=1, max_length=160)
    embedding_provider: Literal["openai", "gemini", "ollama"] = "openai"
    embedding_model: str = Field(default="", max_length=160)
    gemini_embed_dimensions: int = Field(default=768, ge=0, le=3072)

    @field_validator("llm_model", "embedding_model")
    @classmethod
    def strip_model(cls, value: str) -> str:
        return value.strip()


@router.get("/config")
def read_config():
    settings = get_settings()
    return {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "resolved_embedding_model": settings.resolved_embedding_model,
        "gemini_embed_dimensions": settings.gemini_embed_dimensions,
    }


@router.post("/config")
def update_config(request: RuntimeConfigRequest):
    values = request.model_dump()
    if not values["llm_model"]:
        raise HTTPException(422, "A chat model is required.")
    try:
        RUNTIME_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = RUNTIME_CONFIG_PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
        temporary.replace(RUNTIME_CONFIG_PATH)
        get_settings.cache_clear()
        reset_vector_store_cache()
        settings = get_settings()
        return {
            "status": "updated",
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.resolved_embedding_model,
            "note": "Changing embedding configuration selects a separate Chroma collection. Reindex source documents to populate it.",
        }
    except OSError as exc:
        raise HTTPException(500, f"Could not persist runtime configuration: {exc}") from exc
