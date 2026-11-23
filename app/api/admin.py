"""Admin endpoints Moodle uses to push the chat/image/embedding provider, model and API keys.

Keys are Fernet-encrypted into ``storage/runtime_config.json``. ``get_settings()``
reloads that file when it changes, so new values apply to the next request in
every worker without a restart. Key values are never returned.

The embedding selection is optional: ``embedding_provider=""`` (the default,
and what older Moodle clients imply) leaves the ``.env`` embedding settings in
charge. ``embedding_reuse_chat_api_key`` makes embeddings use the chat key
(like ``reuse_chat_api_key`` for images); it needs matching non-Ollama chat and
embedding providers. Changing the embedding identity (provider, resolved model,
Gemini dimensions) returns ``reindex_required: true``; Moodle then queues a
reindex. Keys and the reuse flag are not part of the identity.
"""
import json
import os
import threading
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.config import (
    RUNTIME_CONFIG_PATH, Settings, encrypt_provider_secret, get_settings, reload_settings,
)
from app.security import RequireApiKey
from app.services.configuration_check import run_configuration_checks
from app.services.embedding_provider import chat_key_reuse_error

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[RequireApiKey])
_config_lock = threading.Lock()

# Every Moodle-managed key: (request field, clear flag, runtime-config section, provider).
_KEY_FIELDS = (
    ("openai_api_key", "clear_openai_api_key", "provider_api_keys", "openai"),
    ("gemini_api_key", "clear_gemini_api_key", "provider_api_keys", "gemini"),
    ("openai_image_api_key", "clear_openai_image_api_key", "image_provider_api_keys", "openai"),
    ("gemini_image_api_key", "clear_gemini_image_api_key", "image_provider_api_keys", "gemini"),
    ("openai_embedding_api_key", "clear_openai_embedding_api_key", "embedding_provider_api_keys", "openai"),
    ("gemini_embedding_api_key", "clear_gemini_embedding_api_key", "embedding_provider_api_keys", "gemini"),
)
_KEY_SECTIONS = tuple(dict.fromkeys(section for _, _, section, _ in _KEY_FIELDS))


class _ProviderSelection(BaseModel):
    """Provider/model selection and keys shared by the config and test-configuration bodies."""
    llm_provider: Literal["openai", "gemini", "ollama"]
    llm_model: str = Field(min_length=1, max_length=160)
    image_provider: Literal["openai", "gemini"] = "openai"
    image_model: str = Field(default="", max_length=160)
    reuse_chat_api_key: bool = False
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    openai_image_api_key: str | None = None
    gemini_image_api_key: str | None = None
    # "" = no Moodle override: the .env embedding settings apply. A blank model
    # means the provider's default embedding model.
    embedding_provider: Literal["", "openai", "gemini", "ollama"] = ""
    embedding_model: str = Field(default="", max_length=160)
    openai_embedding_api_key: str | None = None
    gemini_embedding_api_key: str | None = None
    # Use the chat key for the embedding provider (counterpart of reuse_chat_api_key).
    embedding_reuse_chat_api_key: bool = False

    @field_validator("llm_model", "image_model", "embedding_model")
    @classmethod
    def strip_model(cls, value: str) -> str:
        """Trim surrounding whitespace from model names."""
        return value.strip()

    @field_validator(
        "openai_api_key", "gemini_api_key",
        "openai_image_api_key", "gemini_image_api_key",
        "openai_embedding_api_key", "gemini_embedding_api_key",
    )
    @classmethod
    def validate_provider_key(cls, value: str | None) -> str | None:
        """Trim keys and treat a blank value as "not supplied" (keep the stored key)."""
        if value is None:
            return None
        return value.strip() or None


class RuntimeConfigRequest(_ProviderSelection):
    """Body of ``POST /admin/config``. Omitted or blank keys keep the stored value."""
    clear_openai_api_key: bool = False
    clear_gemini_api_key: bool = False
    clear_openai_image_api_key: bool = False
    clear_gemini_image_api_key: bool = False
    clear_openai_embedding_api_key: bool = False
    clear_gemini_embedding_api_key: bool = False


class ConfigurationTestRequest(_ProviderSelection):
    """Body of ``POST /admin/test-configuration``: the settings form, saved or not.

    Blank or omitted keys use the stored key. ``fallback_model`` is the Ollama
    model for the local fallback; blank skips that test. ``test_embedding``
    runs the live embedding check; Moodle sets it only when needed.
    """
    fallback_model: str = Field(default="", max_length=160)
    test_embedding: bool = False

    @field_validator("fallback_model")
    @classmethod
    def strip_fallback_model(cls, value: str) -> str:
        """Trim surrounding whitespace from the fallback model name."""
        return value.strip()


def _validate_key_lengths(request: _ProviderSelection) -> None:
    """422 if a supplied key has an implausible length, without echoing its value."""
    for field, _, _, _ in _KEY_FIELDS:
        credential = getattr(request, field)
        if credential is not None and not 20 <= len(credential) <= 512:
            raise HTTPException(422, "Provider API keys must contain between 20 and 512 characters.")


def _key_status(settings: Settings) -> dict[str, bool]:
    """Report which keys are configured, as booleans only: never the key values."""
    return {
        "openai_api_key_configured": bool(settings.openai_api_key),
        "gemini_api_key_configured": bool(settings.gemini_api_key),
        "openai_image_api_key_configured": bool(settings.image_openai_api_key),
        "gemini_image_api_key_configured": bool(settings.image_gemini_api_key),
        "openai_embedding_api_key_configured": bool(settings.embedding_openai_api_key),
        "gemini_embedding_api_key_configured": bool(settings.embedding_gemini_api_key),
    }


def _embedding_source(runtime_data: dict[str, Any]) -> Literal["moodle", "env"]:
    """Whether the embedding provider/model comes from a Moodle override or ``.env``."""
    return "moodle" if runtime_data.get("embedding_provider") else "env"


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


# Request key field -> Settings field it overrides for a configuration test.
_TEST_KEY_OVERRIDES = {
    "openai_api_key": "openai_api_key",
    "gemini_api_key": "gemini_api_key",
    "openai_image_api_key": "image_openai_api_key",
    "gemini_image_api_key": "image_gemini_api_key",
    "openai_embedding_api_key": "embedding_openai_api_key",
    "gemini_embedding_api_key": "embedding_gemini_api_key",
}


@router.post("/test-configuration")
async def test_configuration(request: ConfigurationTestRequest):
    """Test the settings Moodle sent, including unsaved changes, with live provider calls.

    Reaching this handler means the ``X-RAG-API-Key`` check passed (a wrong key
    gets 401 first). The main LLM, the Ollama fallback model, image generation
    and (when ``test_embedding`` is set) embeddings are then tested
    independently and concurrently; each reports ``pass``, ``fail`` or
    ``skipped``. ``embedding_provider=""`` tests the active ``.env`` embedding
    settings. Returns 200 whatever the results. Nothing is saved, and key
    values are never returned. 422 for invalid input.
    """
    _validate_key_lengths(request)
    overrides: dict[str, Any] = {
        "llm_provider": request.llm_provider,
        "llm_model": request.llm_model,
        "image_provider": request.image_provider,
        "image_model": request.image_model,
        "reuse_chat_api_key": request.reuse_chat_api_key,
    }
    if request.embedding_provider:
        overrides["embedding_provider"] = request.embedding_provider
        overrides["embedding_model"] = request.embedding_model
        # A mismatch is reported by the embedding check, not as a 422.
        overrides["embedding_reuse_chat_api_key"] = request.embedding_reuse_chat_api_key
    for field, setting in _TEST_KEY_OVERRIDES.items():
        if getattr(request, field):
            overrides[setting] = getattr(request, field)
    # A temporary copy: the stored runtime configuration is not changed.
    settings = get_settings().model_copy(update=overrides)

    results = await run_configuration_checks(settings, request.fallback_model,
                                             request.test_embedding)
    return {
        "results": {
            "authentication": {"status": "pass", "message": "Backend accepted the API key."},
            **results,
        }
    }


@router.get("/config")
def read_config():
    """Return the active provider/model selection and which keys are configured.

    ``embedding_source`` is ``moodle`` when Moodle overrides the embedding
    provider/model, otherwise ``env``.
    ``relevance_filter`` is ``calibrated``, ``override`` or ``off`` (no
    calibrated threshold for the active embedding model).
    """
    settings = get_settings()
    return {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "image_provider": settings.image_provider,
        "image_model": settings.image_model,
        "reuse_chat_api_key": settings.reuse_chat_api_key,
        "embedding_reuse_chat_api_key": settings.embedding_reuse_chat_api_key,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "resolved_embedding_model": settings.resolved_embedding_model,
        "gemini_embed_dimensions": settings.gemini_embed_dimensions,
        "embedding_source": _embedding_source(_read_runtime_file()),
        # "off" means off-topic questions are not filtered for this embedding model
        # (uncalibrated); see CALIBRATED_MAX_DISTANCE and eval/calibrate_threshold.py.
        "relevance_filter": settings.relevance_filter_status,
        "relevance_threshold": settings.resolved_retrieval_max_distance,
        **_key_status(settings),
    }


@router.post("/config")
def update_config(request: RuntimeConfigRequest):
    """Store Moodle's provider/model selection and encrypt any keys it sent.

    Blank or omitted keys keep the stored value; a ``clear_*`` flag removes it.
    ``embedding_provider=""`` stores no embedding override (any previous one is
    removed, so ``.env`` applies again). ``embedding_changed`` and
    ``reindex_required`` are true when the embedding identity (provider,
    resolved model, Gemini dimensions) changed; a key-only or reuse-flag-only
    change is not an identity change. ``embedding_reuse_chat_api_key`` requires
    matching non-Ollama chat and embedding providers (422 otherwise); with
    ``embedding_provider=""`` it is ignored and stored as false. Errors: 422 for
    invalid input, 503 if the encryption key is missing, 500 if the file cannot
    be written.
    """
    values = {
        "llm_provider": request.llm_provider,
        "llm_model": request.llm_model,
        "image_provider": request.image_provider,
        "image_model": request.image_model,
        "reuse_chat_api_key": request.reuse_chat_api_key,
    }
    # With .env in charge of embeddings there is no Moodle selection to reuse for.
    embedding_reuse = bool(request.embedding_provider) and request.embedding_reuse_chat_api_key
    values["embedding_reuse_chat_api_key"] = embedding_reuse
    if request.embedding_provider:
        values["embedding_provider"] = request.embedding_provider
        values["embedding_model"] = request.embedding_model
    if not values["llm_model"]:
        raise HTTPException(422, "A chat model is required.")
    # Validate credentials without placing their values in Pydantic error responses.
    _validate_key_lengths(request)
    if request.reuse_chat_api_key and request.llm_provider != request.image_provider:
        raise HTTPException(
            422,
            "The chat API key can only be reused when chat and image providers are the same.",
        )
    if embedding_reuse:
        reuse_error = chat_key_reuse_error(request.embedding_provider, request.llm_provider)
        if reuse_error:
            raise HTTPException(422, reuse_error)

    try:
        with _config_lock:
            RUNTIME_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            current_data = _read_runtime_file()
            identity_before = get_settings().embedding_identity
            stored = {}
            for section in _KEY_SECTIONS:
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
            settings = reload_settings()
            embedding_changed = settings.embedding_identity != identity_before

        return {
            "status": "updated",
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "image_provider": settings.image_provider,
            "image_model": settings.image_model,
            "reuse_chat_api_key": settings.reuse_chat_api_key,
            "embedding_reuse_chat_api_key": settings.embedding_reuse_chat_api_key,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.resolved_embedding_model,
            "llm_changed": True,
            "embedding_changed": embedding_changed,
            "reindex_required": embedding_changed,
            **_key_status(settings),
            "note": (
                "Chat, image and embedding providers, models, and credentials are applied "
                "immediately; no FastAPI restart is required. Changing the embedding provider "
                "or model requires a full document reindex (see reindex_required)."
            ),
        }
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(500, "Could not persist runtime configuration.") from exc
