import json
import os
import threading
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.config import RUNTIME_CONFIG_PATH, encrypt_provider_secret, get_settings
from app.security import RequireApiKey

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[RequireApiKey])
_config_lock = threading.Lock()


class RuntimeConfigRequest(BaseModel):
    llm_provider: Literal["openai", "gemini", "ollama"]
    llm_model: str = Field(min_length=1, max_length=160)
    image_provider: Literal["openai", "gemini"] = "openai"
    image_model: str = Field(default="", max_length=160)
    reuse_chat_api_key: bool = False
    openai_api_key: str | None = Field(default=None, max_length=512)
    gemini_api_key: str | None = Field(default=None, max_length=512)
    openai_image_api_key: str | None = Field(default=None, max_length=512)
    gemini_image_api_key: str | None = Field(default=None, max_length=512)
    clear_openai_api_key: bool = False
    clear_gemini_api_key: bool = False
    clear_openai_image_api_key: bool = False
    clear_gemini_image_api_key: bool = False

    @field_validator("llm_model", "image_model")
    @classmethod
    def strip_model(cls, value: str) -> str:
        return value.strip()

    @field_validator(
        "openai_api_key", "gemini_api_key",
        "openai_image_api_key", "gemini_image_api_key",
    )
    @classmethod
    def validate_provider_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if value and len(value) < 20:
            raise ValueError("Provider API keys must contain at least 20 characters.")
        return value or None


@router.post("/connection-test")
def test_connection():
    return {"status": "ok", "authenticated": True}


@router.get("/config")
def read_config():
    settings = get_settings()
    return {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "image_provider": settings.image_provider,
        "image_model": settings.image_model,
        "reuse_chat_api_key": settings.reuse_chat_api_key,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "resolved_embedding_model": settings.resolved_embedding_model,
        "gemini_embed_dimensions": settings.gemini_embed_dimensions,
        "openai_api_key_configured": bool(settings.openai_api_key),
        "gemini_api_key_configured": bool(settings.gemini_api_key),
        "openai_image_api_key_configured": bool(settings.image_openai_api_key),
        "gemini_image_api_key_configured": bool(settings.image_gemini_api_key),
    }


@router.post("/config")
def update_config(request: RuntimeConfigRequest):
    values = {
        "llm_provider": request.llm_provider,
        "llm_model": request.llm_model,
        "image_provider": request.image_provider,
        "image_model": request.image_model,
        "reuse_chat_api_key": request.reuse_chat_api_key,
    }
    if not values["llm_model"]:
        raise HTTPException(422, "A chat model is required.")
    if request.reuse_chat_api_key and request.llm_provider != request.image_provider:
        raise HTTPException(
            422,
            "The chat API key can only be reused when chat and image providers are the same.",
        )

    try:
        with _config_lock:
            RUNTIME_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            current_data = {}
            if RUNTIME_CONFIG_PATH.is_file():
                try:
                    current_data = json.loads(RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    current_data = {}

            chat_secrets = current_data.get("provider_api_keys", {})
            if not isinstance(chat_secrets, dict):
                chat_secrets = {}
            image_secrets = current_data.get("image_provider_api_keys", {})
            if not isinstance(image_secrets, dict):
                image_secrets = {}

            # Empty/omitted fields preserve existing credentials. Explicit clear
            # flags are required to remove a Moodle-managed provider key.
            if request.clear_openai_api_key:
                chat_secrets.pop("openai", None)
            elif request.openai_api_key:
                chat_secrets["openai"] = encrypt_provider_secret(request.openai_api_key)

            if request.clear_gemini_api_key:
                chat_secrets.pop("gemini", None)
            elif request.gemini_api_key:
                chat_secrets["gemini"] = encrypt_provider_secret(request.gemini_api_key)

            if request.clear_openai_image_api_key:
                image_secrets.pop("openai", None)
            elif request.openai_image_api_key:
                image_secrets["openai"] = encrypt_provider_secret(request.openai_image_api_key)

            if request.clear_gemini_image_api_key:
                image_secrets.pop("gemini", None)
            elif request.gemini_image_api_key:
                image_secrets["gemini"] = encrypt_provider_secret(request.gemini_image_api_key)

            values["provider_api_keys"] = chat_secrets
            values["image_provider_api_keys"] = image_secrets
            temporary = RUNTIME_CONFIG_PATH.with_suffix(".tmp")
            temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
            if os.name == "posix":
                try:
                    os.chmod(temporary, 0o600)
                except OSError:
                    # Fernet encryption remains the primary protection.
                    pass
            temporary.replace(RUNTIME_CONFIG_PATH)
            settings = get_settings()

        return {
            "status": "updated",
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "image_provider": settings.image_provider,
            "image_model": settings.image_model,
            "reuse_chat_api_key": settings.reuse_chat_api_key,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.resolved_embedding_model,
            "llm_changed": True,
            "embedding_changed": False,
            "reindex_required": False,
            "openai_api_key_configured": bool(settings.openai_api_key),
            "gemini_api_key_configured": bool(settings.gemini_api_key),
            "openai_image_api_key_configured": bool(settings.image_openai_api_key),
            "gemini_image_api_key_configured": bool(settings.image_gemini_api_key),
            "note": "Chat and image providers, models, and credentials are applied immediately; no FastAPI restart or document reindex is required.",
        }
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(500, "Could not persist runtime configuration.") from exc
