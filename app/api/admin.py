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
    llm_provider: Literal["openai", "gemini", "ollama"]
    llm_model: str = Field(min_length=1, max_length=160)
    image_model: str = Field(default="", max_length=160)
    embedding_provider: Literal["openai", "gemini", "ollama"] = "openai"
    embedding_model: str = Field(default="", max_length=160)
    gemini_embed_dimensions: int = Field(default=768, ge=0, le=3072)

    @field_validator("llm_model", "image_model", "embedding_model")
    @classmethod
    def strip_model(cls, value: str) -> str:
        return value.strip()


@router.get("/config")
def read_config():
    settings = get_settings()
    return {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "image_model": settings.image_model,
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
    current = get_settings()
    current_values = {
        "llm_provider": current.llm_provider,
        "llm_model": current.llm_model,
        "image_model": current.image_model,
        "embedding_provider": current.embedding_provider,
        "embedding_model": current.embedding_model,
        "gemini_embed_dimensions": current.gemini_embed_dimensions,
    }
    if values == current_values:
        return {
            "status": "unchanged",
            "llm_changed": False,
            "embedding_changed": False,
            "llm_provider": current.llm_provider,
            "llm_model": current.llm_model,
            "image_model": current.image_model,
            "embedding_provider": current.embedding_provider,
            "embedding_model": current.resolved_embedding_model,
        }

    llm_changed = (
        values["llm_provider"] != current.llm_provider
        or values["llm_model"] != current.llm_model
    )
    # Only settings that change the vector space invalidate existing document
    # embeddings. Chat-generation settings must never trigger a document rebuild.
    embedding_changed = (
        values["embedding_provider"] != current.embedding_provider
        or values["embedding_model"] != current.embedding_model
        or (
            values["embedding_provider"] == "gemini"
            and values["gemini_embed_dimensions"] != current.gemini_embed_dimensions
        )
    )
    try:
        RUNTIME_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = RUNTIME_CONFIG_PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
        temporary.replace(RUNTIME_CONFIG_PATH)
        get_settings.cache_clear()
        if embedding_changed:
            reset_vector_store_cache()
        settings = get_settings()
        return {
            "status": "updated",
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "image_model": settings.image_model,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.resolved_embedding_model,
            "llm_changed": llm_changed,
            "embedding_changed": embedding_changed,
            "reindex_required": embedding_changed,
            "note": (
                "Embedding configuration changed. Moodle must reindex stored documents."
                if embedding_changed
                else "Chat-generation configuration changed. Existing document embeddings remain valid."
                if llm_changed
                else None
            ),
        }
    except OSError as exc:
        raise HTTPException(500, f"Could not persist runtime configuration: {exc}") from exc
