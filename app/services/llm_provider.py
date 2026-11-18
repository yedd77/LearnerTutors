"""Single chat-generation abstraction for every provider.

Generation uses LangChain chat models. Moodle sends the provider and model with
each request (or stores them via POST /admin/config), and a new model is built
from ``get_settings()`` on every call, so switching provider, model or key
applies to the next request without a restart or any .env change.

``generate_structured`` is the only structured-output API other modules should use.
"""
import asyncio
import json
import logging
import re
import threading
from typing import Any, NamedTuple, TypeVar

from pydantic import BaseModel, ValidationError

from app.config import Settings, apply_tracing_policy, get_settings
from app.prompt_builder import build_prompt_bundle
from app.prompt_router import RouteResult

logger = logging.getLogger(__name__)
# app.main already does this; repeated here for scripts that import this module directly.
apply_tracing_policy()

SUPPORTED_PROVIDERS = ("openai", "gemini", "ollama")
SchemaT = TypeVar("SchemaT", bound=BaseModel)


class ProviderError(RuntimeError):
    """A chat provider call failed or was misconfigured. The chat API maps it to 502.

    ``retryable`` is True for temporary failures (429, 5xx, timeouts, connection
    errors), where trying a fallback model can help; configuration errors such
    as a bad key or unknown model are never retryable.
    """

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class _Labels(NamedTuple):
    """Error wording per entry point, kept identical to the pre-LangChain messages."""
    unsupported: str
    no_model: str
    failed: str
    empty: str


_CHAT = _Labels("Unsupported chat provider", "No chat model configured",
                "chat request failed", "returned an empty answer.")
_GENERATION = _Labels("Unsupported generation provider", "No model configured",
                      "generation failed", "returned an empty response.")
_STRUCTURED = _Labels("Unsupported generation provider", "No model configured",
                      "structured generation failed", "returned an empty answer.")


def _history_text(history: list[dict[str, Any]]) -> str:
    """Format the last 6 user/assistant turns as ``ROLE: text`` lines (2000 chars each)."""
    return "\n".join(
        f"{item.get('role', 'user').upper()}: {str(item.get('content', ''))[:2000]}"
        for item in history[-6:]
        if item.get("role") in ("user", "assistant")
    ) or "(No previous conversation.)"


def _resolve(provider: str | None, model: str | None,
             labels: _Labels) -> tuple[Settings, str, str]:
    """Validate the provider/model/key selection before any network call."""
    settings = get_settings()
    selected_provider = (provider or settings.llm_provider).lower()
    selected_model = (model or settings.llm_model).strip()
    if selected_provider not in SUPPORTED_PROVIDERS:
        raise ProviderError(f"{labels.unsupported}: {selected_provider}")
    if not selected_model:
        raise ProviderError(f"{labels.no_model} for provider '{selected_provider}'.")
    if selected_provider == "openai" and not settings.openai_api_key:
        raise ProviderError("OPENAI_API_KEY is not configured.")
    if selected_provider == "gemini" and not settings.gemini_api_key:
        raise ProviderError("GEMINI_API_KEY is not configured.")
    return settings, selected_provider, selected_model


def _finish(answer: Any, provider: str, labels: _Labels) -> str:
    """Return the stripped answer, or raise ProviderError if it is empty or not text."""
    if not isinstance(answer, str) or not answer.strip():
        raise ProviderError(f"{provider} {labels.empty}")
    return answer.strip()


# ---------------------------------------------------------------------------
# LangChain backend
# ---------------------------------------------------------------------------
# (provider, model) pairs that rejected an explicit temperature in this process.
# They are sent no temperature from then on, so the provider's default applies.
_TEMPERATURE_REJECTED: set[tuple[str, str]] = set()


def _temperature(settings: Settings, provider: str, model: str) -> float | None:
    """Configured temperature for this provider, or None to use the provider default."""
    if (provider, model) in _TEMPERATURE_REJECTED:
        return None
    value = {
        "openai": settings.openai_temperature,
        "gemini": settings.gemini_temperature,
        "ollama": settings.ollama_temperature,
    }.get(provider, -1)
    return value if value >= 0 else None


def _drop_rejected_temperature(exc: BaseException, provider: str, model: str) -> bool:
    """If ``exc`` is the provider refusing our temperature, remember it and return True.

    Only a 400/422 whose body mentions "temperature" counts, and only when a
    temperature was actually sent, so the caller retries at most once.
    """
    if _temperature(get_settings(), provider, model) is None:
        return False
    status, body = _http_details(exc)
    if status not in (400, 422) or "temperature" not in body.lower():
        return False
    _TEMPERATURE_REJECTED.add((provider, model))
    logger.warning("%s/%s rejected the configured temperature; retrying with the "
                   "provider default and using it for this model from now on.", provider, model)
    return True


def _invoke(provider: str, model: str, lc_messages: list, *, json_mode: bool = False) -> Any:
    """``invoke`` a fresh model, retrying once without temperature if the model rejects it."""
    try:
        return _chat_model(provider, model, json_mode=json_mode).invoke(lc_messages)
    except Exception as exc:
        if not _drop_rejected_temperature(exc, provider, model):
            raise
    return _chat_model(provider, model, json_mode=json_mode).invoke(lc_messages)


async def _ainvoke(provider: str, model: str, lc_messages: list, *, json_mode: bool = False,
                   schema: type[BaseModel] | None = None,
                   settings: Settings | None = None) -> Any:
    """Async ``_invoke``; with ``schema`` it uses ``with_structured_output``.

    ``settings`` overrides ``get_settings()`` (used to test unsaved configuration).
    """
    def build():
        if settings is None:
            llm = _chat_model(provider, model, json_mode=json_mode)
        else:
            llm = _chat_model(provider, model, json_mode=json_mode, settings=settings)
        return llm.with_structured_output(schema) if schema is not None else llm

    try:
        return await build().ainvoke(lc_messages)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if not _drop_rejected_temperature(exc, provider, model):
            raise
    return await build().ainvoke(lc_messages)


def _chat_model(provider: str, model: str, *, json_mode: bool = False,
                settings: Settings | None = None):
    """Build a new LangChain chat model from the current settings.

    Never cache the result: keys and limits can change at runtime through
    POST /admin/config. Retries are disabled so failures surface exactly once.
    ``settings`` overrides ``get_settings()`` (used to test unsaved configuration).
    """
    settings = settings or get_settings()
    temperature = _temperature(settings, provider, model)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": model,
            "api_key": settings.openai_api_key,
            "base_url": settings.openai_base_url,
            "timeout": settings.openai_timeout_seconds,
            "max_retries": 0,
        }
        # OpenAI-compatible reasoning models only support the API default
        # (temperature=1) and reject other values. ChatOpenAI already drops it for
        # gpt-5; other models that reject it are retried without it (_invoke/_ainvoke).
        if temperature is not None:
            kwargs["temperature"] = temperature
        if settings.openai_max_completion_tokens > 0:
            kwargs["max_completion_tokens"] = settings.openai_max_completion_tokens
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return ChatOpenAI(**kwargs)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Temperature and retries are explicit because the library defaults are 0.7 and 6.
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.gemini_api_key,
            temperature=temperature,
            max_output_tokens=settings.gemini_max_output_tokens or None,
            timeout=settings.gemini_timeout_seconds,
            max_retries=0,
            response_mime_type="application/json" if json_mode else None,
        )
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        # Explicit because Ollama's own default (usually 0.8) is high for grounded answers.
        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=settings.ollama_base_url,
            client_kwargs={"timeout": settings.ollama_timeout_seconds},
            format="json" if json_mode else None,
        )
    raise ProviderError(f"Unsupported provider: {provider}")


def _to_lc_messages(messages: list[dict[str, str]], provider: str) -> list:
    """Convert {"role","content"} dicts into LangChain message objects."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    if provider == "gemini":
        # Gemini takes one system instruction, so join all system messages.
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        rest = [m for m in messages if m["role"] != "system"]
        converted: list = [SystemMessage(content=system)] if system else []
    else:
        rest, converted = messages, []
    for message in rest:
        role, content = message["role"], message["content"]
        if role == "system":
            converted.append(SystemMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))
        else:
            converted.append(HumanMessage(content=content))
    return converted


def _message_text(message: Any) -> Any:
    """Extract plain text from a LangChain message across langchain-core versions."""
    text = getattr(message, "text", None)
    if isinstance(text, str):  # langchain-core 1.x: a property (a callable str subclass)
        return str(text)
    if callable(text):  # langchain-core < 1.0 exposed text() as a method
        text = text()
    return text


def _http_details(exc: BaseException) -> tuple[int | None, str]:
    """Find an HTTP status and body anywhere in the exception's cause chain.

    Covers openai.APIStatusError (status_code/response), google.genai APIError
    (code/message), ollama.ResponseError (status_code/error) and httpx errors.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        status = None
        for candidate in (getattr(current, "status_code", None), getattr(current, "code", None),
                          getattr(response, "status_code", None)):
            if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
                status = candidate
                break
        if status is not None:
            body = ""
            try:
                body = getattr(response, "text", "") or ""
            except Exception:  # e.g. httpx.ResponseNotRead on streamed responses
                body = ""
            if not isinstance(body, str) or not body:
                body = str(getattr(current, "message", None) or getattr(current, "error", None)
                           or current)
            return status, body[:1000]
        current = current.__cause__ or current.__context__
    return None, ""


_RETRYABLE_STATUSES = {408, 429}


def _is_connection_failure(exc: BaseException) -> bool:
    """True if a timeout or network error appears anywhere in the cause chain."""
    import httpx

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (TimeoutError, ConnectionError, httpx.TransportError)):
            return True
        # openai.APIConnectionError / APITimeoutError, without importing openai here.
        if type(current).__name__ in ("APIConnectionError", "APITimeoutError"):
            return True
        current = current.__cause__ or current.__context__
    return False


def _langchain_error(exc: Exception, provider: str, labels: _Labels) -> ProviderError:
    """Wrap any provider/SDK exception as ProviderError, with the HTTP status when known."""
    status, body = _http_details(exc)
    if status is not None:
        return ProviderError(f"{provider} returned HTTP {status}: {body}",
                             retryable=status in _RETRYABLE_STATUSES or status >= 500)
    return ProviderError(f"{provider} {labels.failed}: {type(exc).__name__}: {exc}",
                         retryable=_is_connection_failure(exc))


async def _with_fallback(call, provider: str | None, model: str | None,
                         fallback: tuple[str, str] | None):
    """Run ``call(provider, model)``; on a temporary failure, try ``fallback`` once.

    Only retryable ProviderErrors fall back, and only to a different model. Each
    switch is logged. If the fallback fails too, the original error is raised.
    """
    try:
        return await call(provider, model)
    except ProviderError as exc:
        if not exc.retryable or not fallback:
            raise
        settings = get_settings()
        primary = ((provider or settings.llm_provider).lower(), (model or settings.llm_model).strip())
        target = (fallback[0].lower(), fallback[1].strip())
        if target == primary:
            raise
        logger.warning("llm.fallback from=%s/%s to=%s/%s reason=%s",
                       *primary, *target, str(exc)[:200])
        try:
            return await call(*target)
        except ProviderError as fallback_exc:
            logger.warning("llm.fallback.failed to=%s/%s error=%s",
                           *target, str(fallback_exc)[:200])
            raise exc from None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_messages(messages: list[dict[str, str]], provider: str | None = None,
                      model: str | None = None, json_mode: bool = False) -> str:
    """Generate text using the provider and model selected in Moodle."""
    _, selected_provider, selected_model = _resolve(provider, model, _GENERATION)
    try:
        answer = _message_text(_invoke(selected_provider, selected_model,
                                       _to_lc_messages(messages, selected_provider),
                                       json_mode=json_mode))
        return _finish(answer, selected_provider, _GENERATION)
    except ProviderError:
        raise
    except Exception as exc:
        raise _langchain_error(exc, selected_provider, _GENERATION) from exc


async def generate_messages_async(messages: list[dict[str, str]], provider: str | None = None,
                                  model: str | None = None, *,
                                  fallback: tuple[str, str] | None = None) -> str:
    """Async counterpart of ``generate_messages`` for prebuilt message lists.

    ``fallback`` is an optional ``(provider, model)`` tried once on a temporary failure.
    """
    async def call(provider: str | None, model: str | None) -> str:
        _, selected_provider, selected_model = _resolve(provider, model, _GENERATION)
        try:
            result = await _ainvoke(selected_provider, selected_model,
                                    _to_lc_messages(messages, selected_provider))
            return _finish(_message_text(result), selected_provider, _GENERATION)
        except asyncio.CancelledError:
            raise
        except ProviderError:
            raise
        except Exception as exc:
            raise _langchain_error(exc, selected_provider, _GENERATION) from exc

    return await _with_fallback(call, provider, model, fallback)


async def generate_answer_async(route: RouteResult, context: str,
                                history: list[dict] | None = None,
                                provider: str | None = None,
                                model: str | None = None, *,
                                fallback: tuple[str, str] | None = None) -> str:
    """Generate an answer with cancellable asynchronous provider requests.

    ``fallback`` is an optional ``(provider, model)`` tried once on a temporary failure.
    """
    async def call(provider: str | None, model: str | None) -> str:
        settings, selected_provider, selected_model = _resolve(provider, model, _CHAT)
        messages = build_prompt_bundle(
            route=route, context=context, history=_history_text(history or [])
        )["messages"]
        if settings.log_prompts:
            logger.debug("Chat prompt sent to %s/%s", selected_provider, selected_model)

        try:
            result = await _ainvoke(selected_provider, selected_model,
                                    _to_lc_messages(messages, selected_provider))
            return _finish(_message_text(result), selected_provider, _CHAT)
        except asyncio.CancelledError:
            logger.info("Cancelled in-flight %s generation request.", selected_provider)
            raise
        except ProviderError:
            raise
        except Exception as exc:
            raise _langchain_error(exc, selected_provider, _CHAT) from exc

    return await _with_fallback(call, provider, model, fallback)


# Statuses where retrying in JSON mode cannot help (auth, missing model, rate limit).
_NO_FALLBACK_STATUSES = {401, 403, 404, 429}
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


async def _structured_once(provider: str, model: str, messages: list[dict[str, str]],
                           schema: type[SchemaT]) -> SchemaT:
    """One structured-output attempt: native structured output, then a JSON-mode fallback.

    The fallback is skipped for 401/403/404/429 errors, where retrying cannot help.
    """
    lc_messages = _to_lc_messages(messages, provider)
    try:
        result = await _ainvoke(provider, model, lc_messages, schema=schema)
        if isinstance(result, schema):
            return result
        if isinstance(result, dict):
            return schema.model_validate(result)
        reason = f"returned {type(result).__name__}"
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        status, _ = _http_details(exc)
        error = _langchain_error(exc, provider, _STRUCTURED)
        # An overloaded or unreachable model will fail in JSON mode too.
        if status in _NO_FALLBACK_STATUSES or error.retryable:
            raise error from exc
        reason = f"{type(exc).__name__}: {exc}"
    logger.info("Structured output from %s/%s failed (%s); retrying in JSON mode.",
                provider, model, reason[:300])

    # Fallback for models without tool calling / JSON-schema support.
    instruction = (
        "Return only a JSON object that matches this JSON schema. No prose, no code fences.\n"
        + json.dumps(schema.model_json_schema())
    )
    fallback = _to_lc_messages([{"role": "system", "content": instruction}, *messages], provider)
    try:
        result = await _ainvoke(provider, model, fallback, json_mode=True)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise _langchain_error(exc, provider, _STRUCTURED) from exc
    text = _finish(_message_text(result), provider, _STRUCTURED)
    try:
        return schema.model_validate_json(_FENCE.sub("", text))
    except (ValidationError, ValueError) as exc:
        raise ProviderError(
            f"{provider} returned malformed structured output: {str(exc)[:500]}"
        ) from exc


async def generate_structured(messages: list[dict[str, str]], schema: type[SchemaT],
                              provider: str | None = None, model: str | None = None,
                              *, timeout: float | None = None,
                              fallback: tuple[str, str] | None = None) -> SchemaT:
    """Return a validated ``schema`` instance.

    Tries ``with_structured_output`` first, then JSON mode plus
    ``schema.model_validate_json``. Every failure is raised as ProviderError;
    cancellation propagates unchanged. ``fallback`` is an optional
    ``(provider, model)`` tried once on a temporary failure, with its own timeout.
    """
    async def call(provider: str | None, model: str | None) -> SchemaT:
        _, selected_provider, selected_model = _resolve(provider, model, _STRUCTURED)
        run = _structured_once(selected_provider, selected_model, messages, schema)
        try:
            if timeout is not None:
                return await asyncio.wait_for(run, timeout)
            return await run
        except asyncio.CancelledError:
            raise
        except ProviderError:
            raise
        except TimeoutError as exc:
            raise ProviderError(
                f"{selected_provider} structured generation timed out after {timeout}s.",
                retryable=True,
            ) from exc
        except Exception as exc:
            raise _langchain_error(exc, selected_provider, _STRUCTURED) from exc

    return await _with_fallback(call, provider, model, fallback)


async def check_generation(provider: str, model: str, settings: Settings) -> str:
    """Send one tiny prompt to ``provider``/``model`` using ``settings`` and return the reply.

    Used by the admin configuration test, so ``settings`` may hold unsaved keys.
    No fallback model is tried. Failures are raised as ProviderError; the caller
    validates the provider, model and key before calling this.
    """
    messages = [{"role": "user", "content": "Reply with the single word OK."}]
    try:
        result = await _ainvoke(provider, model, _to_lc_messages(messages, provider),
                                settings=settings)
        return _finish(_message_text(result), provider, _GENERATION)
    except asyncio.CancelledError:
        raise
    except ProviderError:
        raise
    except Exception as exc:
        raise _langchain_error(exc, provider, _GENERATION) from exc


_bridge_lock = threading.Lock()
_bridge_loop: asyncio.AbstractEventLoop | None = None


def _bridge() -> asyncio.AbstractEventLoop:
    """Return the long-lived event loop (on a daemon thread) used by generate_structured_sync."""
    global _bridge_loop
    with _bridge_lock:
        if _bridge_loop is None or _bridge_loop.is_closed():
            loop = asyncio.new_event_loop()
            threading.Thread(target=loop.run_forever, name="llm-sync-bridge", daemon=True).start()
            _bridge_loop = loop
        return _bridge_loop


def generate_structured_sync(messages: list[dict[str, str]], schema: type[SchemaT],
                             provider: str | None = None, model: str | None = None,
                             *, timeout: float | None = None) -> SchemaT:
    """``generate_structured`` for sync callers such as threadpool endpoints.

    Runs on one long-lived event loop instead of ``asyncio.run`` per call:
    langchain-openai caches its async httpx client process-wide, and a pooled
    connection bound to an already-closed loop fails with "Event loop is closed".
    """
    future = asyncio.run_coroutine_threadsafe(
        generate_structured(messages, schema, provider, model, timeout=timeout), _bridge()
    )
    return future.result()
