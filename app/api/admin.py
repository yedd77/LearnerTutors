"""Admin endpoints Moodle uses to push the chat/image provider, model and API keys.

Keys are Fernet-encrypted into ``storage/runtime_config.json``. ``get_settings()``
reloads that file when it changes, so new values apply to the next request in
every worker without a restart. Key values are never returned.
"""
import json
import os
import threading
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.config import RUNTIME_CONFIG_PATH, Settings, encrypt_provider_secret, get_settings
from app.security import RequireApiKey

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[RequireApiKey])
_config_lock = threading.Lock()

# Every Moodle-managed key: (request field, clear flag, runtime-config section, provider).
_KEY_FIELDS = (
    ("openai_api_key", "clear_openai_api_key", "provider_api_keys", "openai"),
    ("gemini_api_key", "clear_gemini_api_key", "provider_api_keys", "gemini"),
    ("openai_image_api_key", "clear_openai_image_api_key", "image_provider_api_keys", "openai"),
    ("gemini_image_api_key", "clear_gemini_image_api_key", "image_provider_api_keys", "gemini"),
)


class RuntimeConfigRequest(BaseModel):
    """Body of ``POST /admin/config``. Omitted or blank keys keep the stored value."""
    llm_provider: Literal["openai", "gemini", "ollama"]
    llm_model: str = Field(min_length=1, max_length=160)
    image_provider: Literal["openai", "gemini"] = "openai"
    image_model: str = Field(default="", max_length=160)
    reuse_chat_api_key: bool = False
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    openai_image_api_key: str | None = None
    gemini_image_api_key: str | None = None
    clear_openai_api_key: bool = False
    clear_gemini_api_key: bool = False
    clear_openai_image_api_key: bool = False
    clear_gemini_image_api_key: bool = False

    @field_validator("llm_model", "image_model")
    @classmethod
    def strip_model(cls, value: str) -> str:
        """Trim surrounding whitespace from model names."""
        return value.strip()

    @field_validator(
        "openai_api_key", "gemini_api_key",
        "openai_image_api_key", "gemini_image_api_key",
    )
    @classmethod
    def validate_provider_key(cls, value: str | None) -> str | None:
        """Trim keys and treat a blank value as "not supplied" (keep the stored key)."""
        if value is None:
            return None
        return value.strip() or None


def _key_status(settings: Settings) -> dict[str, bool]:
    """Report which keys are configured, as booleans only: never the key values."""
    return {
        "openai_api_key_configured": bool(settings.openai_api_key),
        "gemini_api_key_configured": bool(settings.gemini_api_key),
        "openai_image_api_key_configured": bool(settings.image_openai_api_key),
        "gemini_image_api_key_configured": bool(settings.image_gemini_api_key),
    }


def _read_runtime_file() -> dict[str, Any]:
    """Return the current runtime config, or an empty dict if it is missing or unreadable."""
    if not RUNTIME_CONFIG_PATH.is_file():
        return {}
    try:
        data = json.loads(RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_runtime_file(values: dict[str, Any]) -> None:
    """Atomically replace the runtime config file (write a temp file, then rename)."""
    temporary = RUNTIME_CONFIG_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
    if os.name == "posix":
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            # Fernet encryption remains the primary protection.
            pass
    temporary.replace(RUNTIME_CONFIG_PATH)


@router.post("/connection-test")
def test_connection():
    """Let Moodle confirm that the backend is reachable and the API key is accepted."""
    return {"status": "ok", "authenticated": True}


@router.get("/config")
def read_config():
    """Return the active provider/model selection and which keys are configured."""
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
        **_key_status(settings),
    }


@router.post("/config")
def update_config(request: RuntimeConfigRequest):
    """Store Moodle's provider/model selection and encrypt any keys it sent.

    Blank or omitted keys keep the stored value; a ``clear_*`` flag removes it.
    Embedding settings are read-only here (they live in ``.env``). Errors:
    422 for invalid input, 503 if the encryption key is missing, 500 if the
    file cannot be written.
    """
    values = {
        "llm_provider": request.llm_provider,
        "llm_model": request.llm_model,
        "image_provider": request.image_provider,
        "image_model": request.image_model,
        "reuse_chat_api_key": request.reuse_chat_api_key,
    }
    if not values["llm_model"]:
        raise HTTPException(422, "A chat model is required.")
    # Validate credentials without placing their values in Pydantic error responses.
    for field, _, _, _ in _KEY_FIELDS:
        credential = getattr(request, field)
        if credential is not None and not 20 <= len(credential) <= 512:
            raise HTTPException(422, "Provider API keys must contain between 20 and 512 characters.")
    if request.reuse_chat_api_key and request.llm_provider != request.image_provider:
        raise HTTPException(
            422,
            "The chat API key can only be reused when chat and image providers are the same.",
        )

    try:
        with _config_lock:
            RUNTIME_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            current_data = _read_runtime_file()
            stored = {}
            for section in ("provider_api_keys", "image_provider_api_keys"):
                section_data = current_data.get(section, {})
                stored[section] = section_data if isinstance(section_data, dict) else {}

            # Empty/omitted fields preserve existing credentials. Explicit clear
            # flags are required to remove a Moodle-managed provider key.
            for field, clear_flag, section, provider in _KEY_FIELDS:
                if getattr(request, clear_flag):
                    stored[section].pop(provider, None)
                elif getattr(request, field):
                    stored[section][provider] = encrypt_provider_secret(getattr(request, field))

            values.update(stored)
            _write_runtime_file(values)
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
            **_key_status(settings),
            "note": "Chat and image providers, models, and credentials are applied immediately; no FastAPI restart or document reindex is required.",
        }
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(500, "Could not persist runtime configuration.") from exc
