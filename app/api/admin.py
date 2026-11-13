import json
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.config import RUNTIME_CONFIG_PATH, get_settings
from app.security import RequireApiKey

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[RequireApiKey])


class RuntimeConfigRequest(BaseModel):
    llm_provider: Literal["openai", "gemini", "ollama"]
    llm_model: str = Field(min_length=1, max_length=160)
    image_model: str = Field(default="", max_length=160)

    @field_validator("llm_model", "image_model")
    @classmethod
    def strip_model(cls, value: str) -> str:
        return value.strip()


@router.post("/connection-test")
def test_connection():
    """Return a minimal response only after the shared API key was validated."""
    return {"status": "ok", "authenticated": True}


@router.get("/config")
def read_config():
    settings = get_settings()
    return {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "image_model": settings.image_model,
        # Expose embedding details for diagnostics only. Moodle cannot change them.
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

    # Embedding settings are backend-owned (.env), never accepted from Moodle.
    # Changing chat or image generation configuration cannot invalidate vectors.
    embedding_changed = False

    try:
        RUNTIME_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = RUNTIME_CONFIG_PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
        temporary.replace(RUNTIME_CONFIG_PATH)
        get_settings.cache_clear()

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
            "reindex_required": False,
            "note": (
                "Chat-generation configuration changed. Existing document embeddings remain valid."
                if llm_changed
                else None
            ),
        }
    except OSError as exc:
        raise HTTPException(500, f"Could not persist runtime configuration: {exc}") from exc
