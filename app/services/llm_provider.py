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
    pass


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
    if not isinstance(answer, str) or not answer.strip():
        raise ProviderError(f"{provider} {labels.empty}")
    return answer.strip()


# ---------------------------------------------------------------------------
# LangChain backend
# ---------------------------------------------------------------------------
def _chat_model(provider: str, model: str, *, json_mode: bool = False):
    """Build a new LangChain chat model from the current settings.

    Never cache the result: keys and limits can change at runtime through
    POST /admin/config. Retries are disabled so failures surface exactly once.
    """
    settings = get_settings()
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": model,
            "api_key": settings.openai_api_key,
            "base_url": settings.openai_base_url,
            "timeout": settings.openai_timeout_seconds,
            "max_retries": 0,
        }
        # Omit temperature for OpenAI-compatible reasoning models that only
        # support the API default (temperature=1). Explicit values such as
        # 0.0 can cause the request to be rejected even when chat works.
        # ChatOpenAI's default (None) leaves it out; openai_temperature stays unused.
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
            temperature=settings.gemini_temperature if settings.gemini_temperature >= 0 else None,
            max_output_tokens=settings.gemini_max_output_tokens or None,
            timeout=settings.gemini_timeout_seconds,
            max_retries=0,
            response_mime_type="application/json" if json_mode else None,
        )
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
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


def _langchain_error(exc: Exception, provider: str, labels: _Labels) -> ProviderError:
    status, body = _http_details(exc)
    if status is not None:
        return ProviderError(f"{provider} returned HTTP {status}: {body}")
    return ProviderError(f"{provider} {labels.failed}: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_messages(messages: list[dict[str, str]], provider: str | None = None,
                      model: str | None = None, json_mode: bool = False) -> str:
    """Generate text using the provider and model selected in Moodle."""
    _, selected_provider, selected_model = _resolve(provider, model, _GENERATION)
    try:
        llm = _chat_model(selected_provider, selected_model, json_mode=json_mode)
        answer = _message_text(llm.invoke(_to_lc_messages(messages, selected_provider)))
        return _finish(answer, selected_provider, _GENERATION)
    except ProviderError:
        raise
    except Exception as exc:
        raise _langchain_error(exc, selected_provider, _GENERATION) from exc


async def generate_answer_async(route: RouteResult, context: str,
                                history: list[dict] | None = None,
                                provider: str | None = None,
                                model: str | None = None) -> str:
    """Generate an answer with cancellable asynchronous provider requests."""
    settings, selected_provider, selected_model = _resolve(provider, model, _CHAT)
    messages = build_prompt_bundle(
        route=route, context=context, history=_history_text(history or [])
    )["messages"]
    if settings.log_prompts:
        logger.debug("Chat prompt sent to %s/%s", selected_provider, selected_model)

    try:
        llm = _chat_model(selected_provider, selected_model)
        result = await llm.ainvoke(_to_lc_messages(messages, selected_provider))
        return _finish(_message_text(result), selected_provider, _CHAT)
    except asyncio.CancelledError:
        logger.info("Cancelled in-flight %s generation request.", selected_provider)
        raise
    except ProviderError:
        raise
    except Exception as exc:
        raise _langchain_error(exc, selected_provider, _CHAT) from exc


# Statuses where retrying in JSON mode cannot help (auth, missing model, rate limit).
_NO_FALLBACK_STATUSES = {401, 403, 404, 429}
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


async def _structured_once(provider: str, model: str, messages: list[dict[str, str]],
                           schema: type[SchemaT]) -> SchemaT:
    lc_messages = _to_lc_messages(messages, provider)
    try:
        llm = _chat_model(provider, model).with_structured_output(schema)
        result = await llm.ainvoke(lc_messages)
        if isinstance(result, schema):
            return result
        if isinstance(result, dict):
            return schema.model_validate(result)
        reason = f"returned {type(result).__name__}"
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        status, _ = _http_details(exc)
        if status in _NO_FALLBACK_STATUSES:
            raise _langchain_error(exc, provider, _STRUCTURED) from exc
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
        result = await _chat_model(provider, model, json_mode=True).ainvoke(fallback)
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
                              *, timeout: float | None = None) -> SchemaT:
    """Return a validated ``schema`` instance.

    Tries ``with_structured_output`` first, then JSON mode plus
    ``schema.model_validate_json``. Every failure is raised as ProviderError;
    cancellation propagates unchanged.
    """
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
            f"{selected_provider} structured generation timed out after {timeout}s."
        ) from exc
    except Exception as exc:
        raise _langchain_error(exc, selected_provider, _STRUCTURED) from exc


_bridge_lock = threading.Lock()
_bridge_loop: asyncio.AbstractEventLoop | None = None


def _bridge() -> asyncio.AbstractEventLoop:
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
