"""Live checks behind ``POST /admin/test-configuration``.

Each check makes one small real request with the configuration Moodle sent,
including unsaved values, and reports ``pass``, ``fail`` or ``skipped``. The
checks run concurrently and each catches its own errors, so one failing
provider never stops the others. Nothing is persisted.

Client messages are fixed, safe wording chosen from the HTTP status or error
type; provider error text is only logged server-side, with any supplied keys
redacted.
"""
import asyncio
import logging
from typing import Literal, TypedDict

from app.config import Settings
from app.services import illustrations, llm_provider
from app.services.llm_provider import SUPPORTED_PROVIDERS, ProviderError

logger = logging.getLogger(__name__)

# Upper bounds per check. Image generation is slower than a one-word chat reply.
LLM_CHECK_TIMEOUT_SECONDS = 60.0
IMAGE_CHECK_TIMEOUT_SECONDS = 120.0

_PROVIDER_LABELS = {"openai": "OpenAI", "gemini": "Gemini", "ollama": "Ollama"}


class CheckResult(TypedDict):
    """Outcome of one check, as returned to Moodle."""
    status: Literal["pass", "fail", "skipped"]
    message: str


def _result(status: Literal["pass", "fail", "skipped"], message: str) -> CheckResult:
    return {"status": status, "message": message}


def _redact(text: str, settings: Settings) -> str:
    """Remove every configured provider key from ``text`` before it is logged."""
    for secret in (settings.openai_api_key, settings.gemini_api_key,
                   settings.image_openai_api_key, settings.image_gemini_api_key):
        if secret:
            text = text.replace(secret, "[redacted]")
    return text


def _failure_message(exc: BaseException, label: str) -> str:
    """Safe client wording for a provider failure; never includes provider error text."""
    if isinstance(exc, TimeoutError):
        return f"{label} did not respond in time."
    status, _ = llm_provider._http_details(exc)
    if status in (401, 403):
        return f"{label} rejected the API key (HTTP {status})."
    if status == 404:
        return f"{label} model was not found or is not available (HTTP 404)."
    if status == 429:
        return f"{label} is rate-limited or out of quota (HTTP 429)."
    if status is not None and status >= 500:
        return f"{label} provider is temporarily unavailable (HTTP {status})."
    if status is not None:
        return f"{label} rejected the request (HTTP {status}). Check the model name."
    if llm_provider._is_connection_failure(exc):
        return f"{label} could not be reached."
    return f"{label} test failed. See the backend log for details."


def _log_failure(check: str, exc: BaseException, settings: Settings) -> None:
    detail = _redact(f"{type(exc).__name__}: {exc}", settings)[:500]
    logger.warning("config_check.fail check=%s error=%s", check, detail)


def _missing_key_message(provider: str, settings: Settings) -> str | None:
    """Return a failure message if ``provider`` needs a chat key that is not configured."""
    if provider == "openai" and not settings.openai_api_key:
        return "No OpenAI API key is configured."
    if provider == "gemini" and not settings.gemini_api_key:
        return "No Gemini API key is configured."
    return None


async def _check_generation(check: str, provider: str, model: str, settings: Settings,
                            label: str, success: str) -> CheckResult:
    """Validate the selection, then send one tiny prompt within the check timeout."""
    if provider not in SUPPORTED_PROVIDERS:
        return _result("fail", f"Unsupported chat provider: {provider}.")
    if not model:
        return _result("fail", "No model is selected.")
    missing = _missing_key_message(provider, settings)
    if missing:
        return _result("fail", missing)
    try:
        await asyncio.wait_for(llm_provider.check_generation(provider, model, settings),
                               LLM_CHECK_TIMEOUT_SECONDS)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # ProviderError, TimeoutError or anything unexpected
        _log_failure(check, exc, settings)
        if isinstance(exc, ProviderError) and str(exc).endswith(llm_provider._GENERATION.empty):
            return _result("fail", f"{label} returned an empty response.")
        return _result("fail", _failure_message(exc, label))
    return _result("pass", success)


async def check_llm(settings: Settings) -> CheckResult:
    """Main chat model: the selected provider and model."""
    provider = (settings.llm_provider or "").lower()
    label = f"Main LLM ({_PROVIDER_LABELS.get(provider, provider)})"
    return await _check_generation("llm", provider, settings.llm_model.strip(), settings,
                                   label, "Main LLM is responding.")


async def check_fallback_model(settings: Settings, fallback_model: str) -> CheckResult:
    """Local fallback: always Ollama, with the administrator's fallback model."""
    if not fallback_model:
        return _result("skipped", "No fallback model is configured.")
    return await _check_generation("fallback_model", "ollama", fallback_model, settings,
                                   "Ollama fallback model", "Ollama fallback model is responding.")


async def check_image_generation(settings: Settings) -> CheckResult:
    """Image provider: generate one small image and discard it."""
    if not settings.image_model.strip():
        return _result("skipped", "No image generation model is configured.")
    provider = (settings.image_provider or "openai").lower()
    label = f"Image generation ({_PROVIDER_LABELS.get(provider, provider)})"
    try:
        # Validation only (provider, key, reuse rule); these messages contain no secrets.
        illustrations._selected_image_key(settings)
    except RuntimeError as exc:
        return _result("fail", str(exc))
    try:
        # The thread cannot be cancelled; on timeout it finishes in the background.
        await asyncio.wait_for(asyncio.to_thread(illustrations.check_image_generation, settings),
                               IMAGE_CHECK_TIMEOUT_SECONDS)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        _log_failure("image_generation", exc, settings)
        return _result("fail", _failure_message(exc, label))
    return _result("pass", "Image generation is working.")


async def run_configuration_checks(settings: Settings,
                                   fallback_model: str) -> dict[str, CheckResult]:
    """Run the three provider checks concurrently and return a result per check.

    An unexpected error in one check becomes that check's ``fail`` result.
    """
    names = ("llm", "fallback_model", "image_generation")
    outcomes = await asyncio.gather(
        check_llm(settings),
        check_fallback_model(settings, fallback_model),
        check_image_generation(settings),
        return_exceptions=True,
    )
    results: dict[str, CheckResult] = {}
    for name, outcome in zip(names, outcomes):
        if isinstance(outcome, asyncio.CancelledError):
            raise outcome
        if isinstance(outcome, BaseException):
            _log_failure(name, outcome, settings)
            outcome = _result("fail", "The test could not be completed. See the backend log.")
        results[name] = outcome
    return results
