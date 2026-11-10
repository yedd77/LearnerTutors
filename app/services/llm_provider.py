import logging
from typing import Any

import requests
from app.config import get_settings
from app.prompt_builder import build_prompt_bundle
from app.prompt_router import RouteResult

logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    pass


def _history_text(history: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{item.get('role', 'user').upper()}: {str(item.get('content', ''))[:2000]}"
        for item in history[-6:]
        if item.get("role") in ("user", "assistant")
    ) or "(No previous conversation.)"


def generate_answer(route: RouteResult, context: str, history: list[dict] | None = None,
                    provider: str | None = None, model: str | None = None) -> str:
    settings = get_settings()
    selected_provider = (provider or settings.llm_provider).lower()
    defaults = {
        "openai": settings.openai_chat_model or settings.llm_model,
        "gemini": settings.gemini_chat_model,
        "ollama": settings.ollama_chat_model,
    }
    selected_model = model or settings.llm_model or defaults.get(selected_provider)
    if selected_provider not in defaults:
        raise ProviderError(f"Unsupported chat provider: {selected_provider}")
    if not selected_model:
        raise ProviderError(f"No chat model configured for provider '{selected_provider}'.")

    messages = build_prompt_bundle(
        route=route, context=context, history=_history_text(history or [])
    )["messages"]
    if settings.log_prompts:
        logger.debug("Chat prompt sent to %s/%s:\n%s", selected_provider, selected_model,
                     "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages))

    try:
        if selected_provider == "openai":
            if not settings.openai_api_key:
                raise ProviderError("OPENAI_API_KEY is not configured.")
            payload: dict[str, Any] = {"model": selected_model, "messages": messages}
            if settings.openai_max_completion_tokens > 0:
                payload["max_completion_tokens"] = settings.openai_max_completion_tokens
            response = requests.post(
                f"{settings.openai_base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                json=payload, timeout=settings.openai_timeout_seconds,
            )
            response.raise_for_status()
            answer = response.json()["choices"][0]["message"]["content"]
        elif selected_provider == "ollama":
            response = requests.post(
                f"{settings.ollama_base_url.rstrip('/')}/api/chat",
                json={"model": selected_model, "messages": messages, "stream": False},
                timeout=settings.ollama_timeout_seconds,
            )
            response.raise_for_status()
            answer = response.json()["message"]["content"]
        else:
            from google import genai
            from google.genai import types
            if not settings.gemini_api_key:
                raise ProviderError("GEMINI_API_KEY is not configured.")
            system_parts = [m["content"] for m in messages if m["role"] == "system"]
            contents = [
                types.Content(
                    role="model" if m["role"] == "assistant" else "user",
                    parts=[types.Part(text=m["content"])],
                )
                for m in messages if m["role"] != "system"
            ]
            client = genai.Client(api_key=settings.gemini_api_key)
            response = client.models.generate_content(
                model=selected_model, contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction="\n\n".join(system_parts) or None,
                    temperature=settings.gemini_temperature if settings.gemini_temperature >= 0 else None,
                    max_output_tokens=settings.gemini_max_output_tokens or None,
                ),
            )
            answer = response.text
        answer = (answer or "").strip()
        if not answer:
            raise ProviderError(f"{selected_provider} returned an empty answer.")
        return answer
    except ProviderError:
        raise
    except requests.HTTPError as exc:
        body = exc.response.text[:1000] if exc.response is not None else str(exc)
        raise ProviderError(f"{selected_provider} returned HTTP {exc.response.status_code if exc.response else 'unknown'}: {body}") from exc
    except Exception as exc:
        raise ProviderError(f"{selected_provider} chat request failed: {type(exc).__name__}: {exc}") from exc



def generate_messages(messages: list[dict[str, str]], provider: str | None = None,
                      model: str | None = None, json_mode: bool = False) -> str:
    """Generate text from a provider using the configured default when omitted."""
    settings = get_settings()
    selected_provider = (provider or settings.llm_provider).lower()
    defaults = {
        "openai": settings.openai_chat_model or settings.llm_model,
        "gemini": settings.gemini_chat_model,
        "ollama": settings.ollama_chat_model,
    }
    selected_model = model or settings.llm_model or defaults.get(selected_provider)
    if selected_provider not in defaults:
        raise ProviderError(f"Unsupported generation provider: {selected_provider}")
    if not selected_model:
        raise ProviderError(f"No model configured for provider '{selected_provider}'.")
    try:
        if selected_provider == "openai":
            if not settings.openai_api_key:
                raise ProviderError("OPENAI_API_KEY is not configured.")
            payload: dict[str, Any] = {"model": selected_model, "messages": messages}
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            if settings.openai_temperature >= 0:
                payload["temperature"] = settings.openai_temperature
            if settings.openai_max_completion_tokens > 0:
                payload["max_completion_tokens"] = settings.openai_max_completion_tokens
            response = requests.post(
                f"{settings.openai_base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                json=payload, timeout=settings.openai_timeout_seconds,
            )
            response.raise_for_status()
            answer = response.json()["choices"][0]["message"]["content"]
        elif selected_provider == "ollama":
            ollama_payload = {"model": selected_model, "messages": messages, "stream": False}
            if json_mode:
                ollama_payload["format"] = "json"
            response = requests.post(
                f"{settings.ollama_base_url.rstrip('/')}/api/chat",
                json=ollama_payload,
                timeout=settings.ollama_timeout_seconds,
            )
            response.raise_for_status()
            answer = response.json()["message"]["content"]
        else:
            from google import genai
            from google.genai import types
            if not settings.gemini_api_key:
                raise ProviderError("GEMINI_API_KEY is not configured.")
            system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
            contents = [types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part(text=m["content"])],
            ) for m in messages if m["role"] != "system"]
            client = genai.Client(api_key=settings.gemini_api_key)
            response = client.models.generate_content(
                model=selected_model, contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system or None,
                    response_mime_type="application/json" if json_mode else None,
                    temperature=settings.gemini_temperature if settings.gemini_temperature >= 0 else None,
                    max_output_tokens=settings.gemini_max_output_tokens or None,
                ),
            )
            answer = response.text
        if not isinstance(answer, str) or not answer.strip():
            raise ProviderError(f"{selected_provider} returned an empty response.")
        return answer.strip()
    except ProviderError:
        raise
    except requests.HTTPError as exc:
        body = exc.response.text[:1000] if exc.response is not None else str(exc)
        raise ProviderError(f"{selected_provider} returned HTTP {exc.response.status_code if exc.response else 'unknown'}: {body}") from exc
    except Exception as exc:
        raise ProviderError(f"{selected_provider} generation failed: {type(exc).__name__}: {exc}") from exc


async def generate_answer_async(route: RouteResult, context: str,
                                history: list[dict] | None = None,
                                provider: str | None = None,
                                model: str | None = None) -> str:
    """Generate an answer with cancellable asynchronous provider requests."""
    import httpx

    settings = get_settings()
    selected_provider = (provider or settings.llm_provider).lower()
    defaults = {
        "openai": settings.openai_chat_model or settings.llm_model,
        "gemini": settings.gemini_chat_model,
        "ollama": settings.ollama_chat_model,
    }
    selected_model = model or settings.llm_model or defaults.get(selected_provider)
    if selected_provider not in defaults:
        raise ProviderError(f"Unsupported chat provider: {selected_provider}")
    if not selected_model:
        raise ProviderError(f"No chat model configured for provider '{selected_provider}'.")

    messages = build_prompt_bundle(
        route=route, context=context, history=_history_text(history or [])
    )["messages"]
    if settings.log_prompts:
        logger.debug("Chat prompt sent to %s/%s", selected_provider, selected_model)

    try:
        if selected_provider in ("openai", "ollama"):
            timeout_seconds = (
                settings.openai_timeout_seconds
                if selected_provider == "openai"
                else settings.ollama_timeout_seconds
            )
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                if selected_provider == "openai":
                    if not settings.openai_api_key:
                        raise ProviderError("OPENAI_API_KEY is not configured.")
                    payload: dict[str, Any] = {
                        "model": selected_model,
                        "messages": messages,
                    }
                    if settings.openai_max_completion_tokens > 0:
                        payload["max_completion_tokens"] = settings.openai_max_completion_tokens
                    response = await client.post(
                        f"{settings.openai_base_url.rstrip('/')}/chat/completions",
                        headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                        json=payload,
                    )
                    response.raise_for_status()
                    answer = response.json()["choices"][0]["message"]["content"]
                else:
                    response = await client.post(
                        f"{settings.ollama_base_url.rstrip('/')}/api/chat",
                        json={"model": selected_model, "messages": messages, "stream": False},
                    )
                    response.raise_for_status()
                    answer = response.json()["message"]["content"]
        else:
            from google import genai
            from google.genai import types

            if not settings.gemini_api_key:
                raise ProviderError("GEMINI_API_KEY is not configured.")
            system_parts = [m["content"] for m in messages if m["role"] == "system"]
            contents = [
                types.Content(
                    role="model" if m["role"] == "assistant" else "user",
                    parts=[types.Part(text=m["content"])],
                )
                for m in messages if m["role"] != "system"
            ]
            client = genai.Client(api_key=settings.gemini_api_key)
            try:
                response = await client.aio.models.generate_content(
                    model=selected_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction="\n\n".join(system_parts) or None,
                        temperature=settings.gemini_temperature if settings.gemini_temperature >= 0 else None,
                        max_output_tokens=settings.gemini_max_output_tokens or None,
                    ),
                )
                answer = response.text
            finally:
                close = getattr(client.aio, "aclose", None)
                if close is not None:
                    await close()

        answer = (answer or "").strip()
        if not answer:
            raise ProviderError(f"{selected_provider} returned an empty answer.")
        return answer
    except asyncio.CancelledError:
        logger.info("Cancelled in-flight %s generation request.", selected_provider)
        raise
    except ProviderError:
        raise
    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:1000]
        raise ProviderError(
            f"{selected_provider} returned HTTP {exc.response.status_code}: {body}"
        ) from exc
    except Exception as exc:
        raise ProviderError(
            f"{selected_provider} chat request failed: {type(exc).__name__}: {exc}"
        ) from exc
