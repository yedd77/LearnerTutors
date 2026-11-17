"""Provider abstraction: model factory, runtime provider switching, structured output, errors.

All offline. LangChain models are either inspected (constructor fields and the
request payload they would send) or replaced with small fakes.
"""
import asyncio
import copy
import json
import os

import httpx
import pytest
from pydantic import BaseModel

import app.services.llm_provider as llm
import app.services.suggestion_generator as sg
from app.config import apply_tracing_policy, encrypt_provider_secret, get_settings
from app.prompt_router import route_message
from app.services.llm_provider import ProviderError
from helpers import chat_body, index_text

MESSAGES = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]


# --------------------------------------------------------------------------- fakes
class FakeLLM:
    """Duck-typed stand-in for a LangChain chat model."""

    def __init__(self, reply="Fake answer.", error=None, structured=None, structured_error=None):
        self.reply, self.error = reply, error
        self.structured, self.structured_error = structured, structured_error
        self.calls: list = []
        self.schema = None

    def _respond(self, messages):
        from langchain_core.messages import AIMessage

        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return AIMessage(content=self.reply)

    def invoke(self, messages):
        return self._respond(messages)

    async def ainvoke(self, messages):
        if self.schema is not None:
            self.calls.append(messages)
            if self.structured_error is not None:
                raise self.structured_error
            return self.structured
        return self._respond(messages)

    def with_structured_output(self, schema):
        clone = copy.copy(self)  # shares .calls; keeps subclass overrides
        clone.schema = schema
        return clone


@pytest.fixture
def fake_models(monkeypatch):
    """Replace the model factory. Returns the list of (provider, model, json_mode, fake)."""
    built: list = []
    state = {"factory": lambda: FakeLLM()}

    def fake_chat_model(provider, model, *, json_mode=False):
        fake = state["factory"]()
        built.append((provider, model, json_mode, fake))
        return fake

    monkeypatch.setattr(llm, "_chat_model", fake_chat_model)

    class Handle(list):
        def use(self, factory):
            state["factory"] = factory

    handle = Handle()
    handle.built = built
    return handle


def openai_status_error(status: int, body: str = '{"error":"nope"}'):
    import openai

    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx.Response(status, text=body, request=request)
    return openai.APIStatusError("provider failed", response=response, body=None)


def langchain_backend(configure, **extra):
    configure(OPENAI_API_KEY="sk-test", GEMINI_API_KEY="gm-test", **extra)


# --------------------------------------------------------------------------- settings / tracing
def test_tracing_disabled_by_default():
    assert get_settings().langsmith_tracing is False


def test_tracing_env_vars_are_forced_off_by_default(monkeypatch):
    import app.main  # noqa: F401  (startup applies the policy)

    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    apply_tracing_policy()
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"


def test_tracing_left_alone_when_explicitly_enabled(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    apply_tracing_policy()
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"


# --------------------------------------------------------------------------- model factory
def test_openai_model_settings(configure):
    configure(OPENAI_API_KEY="sk-test", OPENAI_BASE_URL="https://proxy.example/v1/",
              OPENAI_TIMEOUT_SECONDS="42", OPENAI_MAX_COMPLETION_TOKENS="321")
    model = llm._chat_model("openai", "gpt-x")
    assert model.max_retries == 0
    assert model.request_timeout == 42
    assert model.root_client.max_retries == 0
    assert str(model.root_client.base_url).rstrip("/") == "https://proxy.example/v1"
    assert model.openai_api_key.get_secret_value() == "sk-test"
    payload = model._get_request_payload(llm._to_lc_messages(MESSAGES, "openai"))
    assert payload["model"] == "gpt-x"
    assert payload["max_completion_tokens"] == 321
    assert payload["temperature"] == 0.5
    assert payload["messages"] == [{"role": "system", "content": "Be brief."},
                                   {"role": "user", "content": "Hi"}]


def test_openai_token_limit_omitted_when_zero(configure):
    configure(OPENAI_API_KEY="sk-test", OPENAI_MAX_COMPLETION_TOKENS="0")
    payload = llm._chat_model("openai", "gpt-x")._get_request_payload(
        llm._to_lc_messages(MESSAGES, "openai"))
    assert "max_completion_tokens" not in payload and "max_tokens" not in payload


def test_openai_json_mode_sets_response_format(configure):
    configure(OPENAI_API_KEY="sk-test")
    payload = llm._chat_model("openai", "gpt-x", json_mode=True)._get_request_payload(
        llm._to_lc_messages(MESSAGES, "openai"))
    assert payload["response_format"] == {"type": "json_object"}


def test_gemini_model_settings(configure):
    configure(GEMINI_API_KEY="gm-test", GEMINI_TEMPERATURE="0.3",
              GEMINI_MAX_OUTPUT_TOKENS="0", GEMINI_TIMEOUT_SECONDS="77")
    model = llm._chat_model("gemini", "gemini-x")
    assert model.model.endswith("gemini-x")
    assert model.google_api_key.get_secret_value() == "gm-test"
    assert model.temperature == 0.3
    assert model.max_output_tokens is None
    assert model.timeout == 77
    assert model.max_retries == 0
    assert model.response_mime_type is None


def test_gemini_negative_temperature_means_provider_default(configure):
    configure(GEMINI_API_KEY="gm-test", GEMINI_TEMPERATURE="-1", GEMINI_MAX_OUTPUT_TOKENS="512")
    model = llm._chat_model("gemini", "gemini-x", json_mode=True)
    assert model.temperature is None
    assert model.max_output_tokens == 512
    assert model.response_mime_type == "application/json"


def test_ollama_model_settings(configure):
    configure(OLLAMA_BASE_URL="http://127.0.0.1:11999", OLLAMA_TIMEOUT_SECONDS="33")
    model = llm._chat_model("ollama", "llama-x", json_mode=True)
    assert model.model == "llama-x"
    assert model.base_url == "http://127.0.0.1:11999"
    assert model.client_kwargs == {"timeout": 33}
    assert model.format == "json"
    assert llm._chat_model("ollama", "llama-x").format is None


# --------------------------------------------------------------------------- temperature
@pytest.fixture(autouse=True)
def fresh_temperature_memory(monkeypatch):
    """Each test starts with no model remembered as rejecting temperature."""
    monkeypatch.setattr(llm, "_TEMPERATURE_REJECTED", set())


TEMPERATURE_REJECTION = (
    '{"error": {"message": "Unsupported value: \'temperature\' does not support 0.5 with '
    'this model. Only the default (1) value is supported.", "param": "temperature"}}'
)


def test_default_temperature_is_half_for_every_provider(configure):
    configure(OPENAI_API_KEY="sk-test", GEMINI_API_KEY="gm-test")
    assert llm._chat_model("openai", "gpt-x").temperature == 0.5
    assert llm._chat_model("gemini", "gemini-x").temperature == 0.5
    assert llm._chat_model("ollama", "llama-x").temperature == 0.5


def test_negative_temperature_sends_none(configure):
    configure(OPENAI_API_KEY="sk-test", OPENAI_TEMPERATURE="-1", OLLAMA_TEMPERATURE="-1")
    payload = llm._chat_model("openai", "gpt-x")._get_request_payload(
        llm._to_lc_messages(MESSAGES, "openai"))
    assert "temperature" not in payload
    assert llm._chat_model("ollama", "llama-x").temperature is None


class RejectsTemperatureOnce(FakeLLM):
    """First model built raises the OpenAI 'unsupported temperature' 400; later ones answer."""
    built = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        type(self).built += 1
        if type(self).built == 1:
            self.error = self.structured_error = openai_status_error(400, TEMPERATURE_REJECTION)


def test_rejected_temperature_retries_with_provider_default(configure, fake_models, caplog):
    langchain_backend(configure)
    RejectsTemperatureOnce.built = 0
    fake_models.use(RejectsTemperatureOnce)
    with caplog.at_level("WARNING", logger="app.services.llm_provider"):
        answer = llm.generate_messages(MESSAGES, provider="openai", model="o3-x")
    assert answer == "Fake answer."
    assert len(fake_models.built) == 2
    assert "rejected the configured temperature" in caplog.text
    # Remembered: this pair is now built without a temperature; other models keep 0.5.
    assert llm._temperature(get_settings(), "openai", "o3-x") is None
    assert llm._temperature(get_settings(), "openai", "gpt-other") == 0.5


def test_remembered_rejection_omits_temperature_from_real_model(configure):
    configure(OPENAI_API_KEY="sk-test")
    llm._TEMPERATURE_REJECTED.add(("openai", "o3-x"))
    payload = llm._chat_model("openai", "o3-x")._get_request_payload(
        llm._to_lc_messages(MESSAGES, "openai"))
    assert "temperature" not in payload


def test_rejected_temperature_retry_async_and_structured(configure, fake_models):
    langchain_backend(configure)
    RejectsTemperatureOnce.built = 0
    fake_models.use(RejectsTemperatureOnce)
    answer = asyncio.run(llm.generate_answer_async(route_message("q"), "", [], "openai", "o3-x"))
    assert answer == "Fake answer."

    llm._TEMPERATURE_REJECTED.clear()
    RejectsTemperatureOnce.built = 0
    fake_models.use(lambda: RejectsTemperatureOnce(structured=Card(title="x", score=1)))
    result = asyncio.run(llm.generate_structured(MESSAGES, Card, "openai", "o3-x"))
    assert result == Card(title="x", score=1)
    # Retried natively, not by the JSON-mode fallback.
    assert [json_mode for _, _, json_mode, _ in fake_models.built[-2:]] == [False, False]


def test_other_400_errors_are_not_retried(configure, fake_models):
    langchain_backend(configure)
    fake_models.use(lambda: FakeLLM(error=openai_status_error(400, '{"error":"bad request"}')))
    with pytest.raises(ProviderError, match="HTTP 400"):
        llm.generate_messages(MESSAGES, provider="openai", model="m")
    assert len(fake_models.built) == 1
    assert not llm._TEMPERATURE_REJECTED


def test_temperature_rejection_retried_only_once(configure, fake_models):
    langchain_backend(configure)
    fake_models.use(lambda: FakeLLM(error=openai_status_error(400, TEMPERATURE_REJECTION)))
    with pytest.raises(ProviderError, match="HTTP 400"):
        llm.generate_messages(MESSAGES, provider="openai", model="m")
    assert len(fake_models.built) == 2


def test_runtime_key_change_applies_to_next_model(configure, runtime_path):
    configure(OPENAI_API_KEY="")

    def store(key):
        runtime_path.write_text(json.dumps({
            "llm_provider": "openai", "llm_model": "gpt-x",
            "provider_api_keys": {"openai": encrypt_provider_secret(key)},
        }), encoding="utf-8")
        os.utime(runtime_path, ns=(0, len(key) * 10**9 + hash(key) % 10**9))

    store("sk-first")
    assert llm._chat_model("openai", "gpt-x").openai_api_key.get_secret_value() == "sk-first"
    store("sk-second-key")
    assert llm._chat_model("openai", "gpt-x").openai_api_key.get_secret_value() == "sk-second-key"


@pytest.mark.parametrize("provider,env,message", [
    ("openai", {"OPENAI_API_KEY": ""}, "OPENAI_API_KEY is not configured."),
    ("gemini", {"GEMINI_API_KEY": ""}, "GEMINI_API_KEY is not configured."),
    ("mistral", {}, "Unsupported generation provider: mistral"),
])
def test_validation_happens_before_any_model_is_built(configure, fake_models, provider, env,
                                                      message):
    configure(**env)
    with pytest.raises(ProviderError, match=message):
        llm.generate_messages(MESSAGES, provider=provider, model="m")
    assert fake_models.built == []


def test_empty_model_is_rejected(configure):
    configure(OPENAI_API_KEY="sk-test")
    with pytest.raises(ProviderError, match="No model configured for provider 'openai'"):
        llm.generate_messages(MESSAGES, provider="openai", model="  ")


# --------------------------------------------------------------------------- message conversion
def test_message_conversion():
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    messages = [*MESSAGES, {"role": "assistant", "content": "Hello"}]
    converted = llm._to_lc_messages(messages, "openai")
    assert [type(m) for m in converted] == [SystemMessage, HumanMessage, AIMessage]
    assert [m.content for m in converted] == ["Be brief.", "Hi", "Hello"]


def test_gemini_system_messages_are_joined():
    messages = [{"role": "system", "content": "A"}, {"role": "user", "content": "Q"},
                {"role": "system", "content": "B"}]
    converted = llm._to_lc_messages(messages, "gemini")
    assert converted[0].content == "A\n\nB"
    assert [m.content for m in converted[1:]] == ["Q"]


# --------------------------------------------------------------------------- langchain backend
@pytest.mark.parametrize("provider", ["openai", "gemini", "ollama"])
def test_langchain_generate_messages(configure, fake_models, provider):
    langchain_backend(configure)
    fake_models.use(lambda: FakeLLM(reply="  answer  "))
    assert llm.generate_messages(MESSAGES, provider=provider, model="m", json_mode=True) == "answer"
    [(built_provider, model, json_mode, _)] = fake_models.built
    assert (built_provider, model, json_mode) == (provider, "m", True)


@pytest.mark.parametrize("provider", ["openai", "gemini", "ollama"])
def test_langchain_generate_answer_async(configure, fake_models, provider):
    langchain_backend(configure)
    route = route_message("What is photosynthesis?")
    answer = asyncio.run(llm.generate_answer_async(route, "[S1] ctx", [], provider, "m"))
    assert answer == "Fake answer."
    fake = fake_models.built[0][3]
    sent = fake.calls[0]
    assert sent[0].type == "system" and sent[-1].type == "human"
    assert "[S1] ctx" in "".join(m.content for m in sent)


def test_langchain_empty_answer(configure, fake_models):
    langchain_backend(configure)
    fake_models.use(lambda: FakeLLM(reply="   "))
    with pytest.raises(ProviderError, match="^openai returned an empty answer.$"):
        asyncio.run(llm.generate_answer_async(route_message("q"), "", [], "openai", "m"))


def test_langchain_http_error_mapping(configure, fake_models):
    langchain_backend(configure)
    fake_models.use(lambda: FakeLLM(error=openai_status_error(401, '{"error":"bad key"}')))
    with pytest.raises(ProviderError, match=r'^openai returned HTTP 401: \{"error":"bad key"\}$'):
        llm.generate_messages(MESSAGES, provider="openai", model="m")


def test_langchain_wrapped_cause_is_found(configure, fake_models):
    """Library wrappers (e.g. ChatGoogleGenerativeAIError) keep the SDK error as __cause__."""
    from google.genai import errors as genai_errors

    def wrapped():
        try:
            raise genai_errors.APIError(503, {"error": {"message": "overloaded", "status": "UNAVAILABLE"}})
        except genai_errors.APIError as inner:
            try:
                raise RuntimeError("Error calling model") from inner
            except RuntimeError as outer:
                return outer

    langchain_backend(configure)
    fake_models.use(lambda: FakeLLM(error=wrapped()))
    with pytest.raises(ProviderError, match="^gemini returned HTTP 503: overloaded"):
        llm.generate_messages(MESSAGES, provider="gemini", model="m")


def test_ollama_response_error_mapping(configure, fake_models):
    import ollama

    langchain_backend(configure)
    fake_models.use(lambda: FakeLLM(error=ollama.ResponseError("model 'x' not found", 404)))
    with pytest.raises(ProviderError, match="^ollama returned HTTP 404: model 'x' not found"):
        llm.generate_messages(MESSAGES, provider="ollama", model="x")


def test_langchain_non_http_error(configure, fake_models):
    langchain_backend(configure)
    fake_models.use(lambda: FakeLLM(error=ConnectionError("refused")))
    with pytest.raises(ProviderError, match="^ollama chat request failed: ConnectionError: refused$"):
        asyncio.run(llm.generate_answer_async(route_message("q"), "", [], "ollama", "m"))


def test_chat_endpoint_returns_502_on_provider_error(client, auth, configure, fake_models):
    langchain_backend(configure)
    index_text("bio", "Photosynthesis converts light energy into chemical energy.")
    fake_models.use(lambda: FakeLLM(error=openai_status_error(429, "rate limited")))
    response = client.post("/api/v1/chat", headers=auth, json=chat_body("What is photosynthesis?"))
    assert response.status_code == 502
    assert response.json()["detail"] == "openai returned HTTP 429: rate limited"


def test_cancellation_propagates(configure, fake_models):
    configure(OPENAI_API_KEY="sk-test")
    started = asyncio.Event()

    async def hang(*args, **kwargs):
        started.set()
        await asyncio.sleep(3600)

    class HangingLLM(FakeLLM):
        async def ainvoke(self, messages):
            return await hang()

    fake_models.use(HangingLLM)

    async def scenario():
        task = asyncio.create_task(
            llm.generate_answer_async(route_message("q"), "", [], "openai", "m"))
        await started.wait()
        task.cancel()
        await task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(scenario())


def test_sync_generate_answer_is_removed():
    assert not hasattr(llm, "generate_answer")


# --------------------------------------------------------------------------- structured output
class Card(BaseModel):
    title: str
    score: int


@pytest.mark.parametrize("provider", ["openai", "gemini", "ollama"])
def test_generate_structured_primary_path(configure, fake_models, provider):
    langchain_backend(configure)
    fake_models.use(lambda: FakeLLM(structured=Card(title="x", score=1)))
    result = asyncio.run(llm.generate_structured(MESSAGES, Card, provider, "m"))
    assert result == Card(title="x", score=1)
    assert len(fake_models.built) == 1


@pytest.mark.parametrize("provider", ["openai", "gemini", "ollama"])
def test_generate_structured_falls_back_to_json_mode(configure, fake_models, provider):
    langchain_backend(configure)
    fake_models.use(lambda: FakeLLM(
        reply='```json\n{"title": "y", "score": 3}\n```',
        structured_error=NotImplementedError("no tool calling"),
    ))
    result = asyncio.run(llm.generate_structured(MESSAGES, Card, provider, "m"))
    assert result == Card(title="y", score=3)
    assert [json_mode for _, _, json_mode, _ in fake_models.built] == [False, True]
    fallback_messages = fake_models.built[1][3].calls[-1]
    assert "JSON schema" in fallback_messages[0].content


@pytest.mark.parametrize("reply", ["not json", '{"title": "y"}', '{"title": "y", "score": "high"}'])
def test_generate_structured_malformed_output(configure, fake_models, reply):
    langchain_backend(configure)
    fake_models.use(lambda: FakeLLM(reply=reply, structured_error=ValueError("parse failed")))
    with pytest.raises(ProviderError, match="malformed structured output"):
        asyncio.run(llm.generate_structured(MESSAGES, Card, "openai", "m"))


def test_generate_structured_auth_error_skips_fallback(configure, fake_models):
    langchain_backend(configure)
    fake_models.use(lambda: FakeLLM(structured_error=openai_status_error(401, "bad key")))
    with pytest.raises(ProviderError, match="^openai returned HTTP 401: bad key$"):
        asyncio.run(llm.generate_structured(MESSAGES, Card, "openai", "m"))
    assert len(fake_models.built) == 1


def test_generate_structured_timeout(configure, fake_models):
    langchain_backend(configure)

    class Slow(FakeLLM):
        async def ainvoke(self, messages):
            await asyncio.sleep(5)

    fake_models.use(Slow)
    with pytest.raises(ProviderError, match="timed out after 0.05s"):
        asyncio.run(llm.generate_structured(MESSAGES, Card, "openai", "m", timeout=0.05))


def test_generate_structured_validates_before_network(configure, fake_models):
    configure(OPENAI_API_KEY="")
    with pytest.raises(ProviderError, match="OPENAI_API_KEY is not configured."):
        asyncio.run(llm.generate_structured(MESSAGES, Card, "openai", "m"))
    assert fake_models.built == []


def test_generate_structured_sync_reuses_one_event_loop(configure, fake_models):
    """langchain-openai caches its async httpx client, so every sync call must share a loop."""
    langchain_backend(configure)
    loops = []

    class LoopRecorder(FakeLLM):
        async def ainvoke(self, messages):
            loops.append(asyncio.get_running_loop())
            return Card(title="x", score=1)

    fake_models.use(LoopRecorder)
    for _ in range(2):
        assert llm.generate_structured_sync(MESSAGES, Card, "openai", "m") == Card(title="x", score=1)
    assert len(loops) == 2 and loops[0] is loops[1] and loops[0].is_running()


def test_generate_structured_sync_raises_provider_error(configure, fake_models):
    langchain_backend(configure)
    fake_models.use(lambda: FakeLLM(reply="nope", structured_error=ValueError("x")))
    with pytest.raises(ProviderError, match="malformed structured output"):
        llm.generate_structured_sync(MESSAGES, Card, "openai", "m")


# --------------------------------------------------------------------------- suggestions
SECTIONS = [{"page": 1, "text": "Photosynthesis converts light energy into chemical energy."}]


def configure_suggestions(configure, runtime_path):
    configure(OPENAI_API_KEY="sk-test")
    runtime_path.write_text(json.dumps({"llm_provider": "openai", "llm_model": "m"}), encoding="utf-8")


def test_suggestions_use_structured_output(configure, runtime_path, monkeypatch):
    configure_suggestions(configure, runtime_path)
    calls = []

    async def fake_structured(messages, schema, provider=None, model=None, **kwargs):
        calls.append((schema, provider, model))
        return schema(suggestions=["What is A?", "What is a?", "Why B?", "How C?"])

    monkeypatch.setattr(llm, "generate_structured", fake_structured)
    monkeypatch.setattr(sg, "generate_messages", lambda **k: pytest.fail("native path used"))
    assert sg.generate_document_suggestions("bio.pdf", SECTIONS) == ["What is A?", "Why B?", "How C?"]
    assert calls == [(sg.Suggestions, "openai", "m")]


def test_structured_suggestion_failure_uses_fallback(configure, runtime_path, monkeypatch):
    configure_suggestions(configure, runtime_path)

    async def failing(*args, **kwargs):
        raise ProviderError("openai returned malformed structured output")

    monkeypatch.setattr(llm, "generate_structured", failing)
    suggestions = sg.generate_document_suggestions("bio.pdf", SECTIONS)
    assert len(suggestions) == 3
    assert suggestions == sg._fallback_suggestions("bio.pdf", SECTIONS)


# --------------------------------------------------------------------------- runtime provider switching
def test_moodle_can_switch_provider_and_model_between_requests(client, auth, runtime_path, monkeypatch):
    """Admin switches from local Ollama to Gemini in Moodle; no restart or .env change."""
    built = []
    real_chat_model = llm._chat_model

    def spy(provider, model, *, json_mode=False):
        real = real_chat_model(provider, model, json_mode=json_mode)  # real settings resolution
        built.append((provider, model, real))
        return FakeLLM(reply=f"answer from {provider}/{model}")

    monkeypatch.setattr(llm, "_chat_model", spy)
    index_text("bio", "Photosynthesis converts light energy into chemical energy.")

    def ask(provider, model):
        response = client.post("/api/v1/chat", headers=auth,
                               json=chat_body("What is photosynthesis?", provider=provider, model=model))
        assert response.status_code == 200, response.text
        return response.json()["answer"]

    assert ask("ollama", "qwen2.5:7b-instruct") == "answer from ollama/qwen2.5:7b-instruct"

    # Moodle pushes the Gemini key through the admin endpoint, then sends the new selection.
    response = client.post("/api/v1/admin/config", headers=auth, json={
        "llm_provider": "gemini", "llm_model": "gemini-4-argon", "gemini_api_key": "gm-runtime-key-0123456789abcdef",
    })
    assert response.status_code == 200, response.text
    assert ask("gemini", "gemini-4-argon") == "answer from gemini/gemini-4-argon"

    (p1, m1, ollama_model), (p2, m2, gemini_model) = built
    assert (p1, m1, type(ollama_model).__name__) == ("ollama", "qwen2.5:7b-instruct", "ChatOllama")
    assert (p2, m2, type(gemini_model).__name__) == ("gemini", "gemini-4-argon", "ChatGoogleGenerativeAI")
    assert gemini_model.google_api_key.get_secret_value() == "gm-runtime-key-0123456789abcdef"
    assert gemini_model.model.endswith("gemini-4-argon")


def test_gemini_without_key_fails_cleanly(client, auth, fake_models):
    index_text("bio", "Photosynthesis converts light energy into chemical energy.")
    response = client.post("/api/v1/chat", headers=auth,
                           json=chat_body("What is photosynthesis?", provider="gemini", model="gemini-4-argon"))
    assert response.status_code == 502
    assert response.json()["detail"] == "GEMINI_API_KEY is not configured."
    assert fake_models.built == []


# --------------------------------------------------------------------------- fallback model
FALLBACK = ("ollama", "local-model")


@pytest.fixture
def per_provider_models(monkeypatch):
    """Model factory that fails per provider. Set ``errors[provider]``; records builds."""
    errors: dict = {}
    built: list = []

    def fake_chat_model(provider, model, *, json_mode=False):
        built.append((provider, model, json_mode))
        error = errors.get(provider)
        return FakeLLM(reply=f"answer from {provider}", error=error, structured_error=error,
                       structured=Card(title=provider, score=1))

    monkeypatch.setattr(llm, "_chat_model", fake_chat_model)
    return errors, built


@pytest.mark.parametrize("error", [
    openai_status_error(503, "overloaded"),
    openai_status_error(429, "rate limited"),
    openai_status_error(500, "server error"),
    httpx.ConnectError("connection refused"),
    httpx.ReadTimeout("timed out"),
])
def test_temporary_failure_falls_back_to_the_other_model(configure, per_provider_models, caplog, error):
    langchain_backend(configure)
    errors, built = per_provider_models
    errors["gemini"] = error
    with caplog.at_level("WARNING", logger="app.services.llm_provider"):
        answer = asyncio.run(llm.generate_answer_async(
            route_message("q"), "", [], "gemini", "gem-x", fallback=FALLBACK))
    assert answer == "answer from ollama"
    assert [(p, m) for p, m, _ in built] == [("gemini", "gem-x"), ("ollama", "local-model")]
    assert "llm.fallback from=gemini/gem-x to=ollama/local-model" in caplog.text


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_configuration_errors_do_not_fall_back(configure, per_provider_models, status):
    langchain_backend(configure)
    errors, built = per_provider_models
    errors["gemini"] = openai_status_error(status, "bad config")
    with pytest.raises(ProviderError, match=f"HTTP {status}"):
        asyncio.run(llm.generate_answer_async(
            route_message("q"), "", [], "gemini", "gem-x", fallback=FALLBACK))
    assert [p for p, _, _ in built] == ["gemini"]


def test_no_fallback_means_original_error(configure, per_provider_models):
    langchain_backend(configure)
    errors, built = per_provider_models
    errors["gemini"] = openai_status_error(503, "overloaded")
    with pytest.raises(ProviderError, match="HTTP 503"):
        asyncio.run(llm.generate_messages_async(MESSAGES, "gemini", "gem-x"))
    assert len(built) == 1


def test_fallback_to_the_same_model_is_skipped(configure, per_provider_models):
    langchain_backend(configure)
    errors, built = per_provider_models
    errors["ollama"] = openai_status_error(503, "busy")
    with pytest.raises(ProviderError, match="HTTP 503"):
        asyncio.run(llm.generate_messages_async(MESSAGES, "ollama", "local-model",
                                                fallback=("OLLAMA", " local-model ")))
    assert len(built) == 1


def test_failed_fallback_raises_the_original_error(configure, per_provider_models, caplog):
    langchain_backend(configure)
    errors, built = per_provider_models
    errors["gemini"] = openai_status_error(503, "overloaded")
    errors["ollama"] = httpx.ConnectError("ollama is not running")
    with caplog.at_level("WARNING", logger="app.services.llm_provider"):
        with pytest.raises(ProviderError, match="^gemini returned HTTP 503: overloaded$"):
            asyncio.run(llm.generate_messages_async(MESSAGES, "gemini", "gem-x", fallback=FALLBACK))
    assert [p for p, _, _ in built] == ["gemini", "ollama"]
    assert "llm.fallback.failed to=ollama/local-model" in caplog.text


def test_structured_overload_skips_json_mode_and_falls_back(configure, per_provider_models):
    langchain_backend(configure)
    errors, built = per_provider_models
    errors["gemini"] = openai_status_error(503, "overloaded")
    result = asyncio.run(llm.generate_structured(MESSAGES, Card, "gemini", "gem-x", fallback=FALLBACK))
    assert result == Card(title="ollama", score=1)
    # No JSON-mode retry against the overloaded model.
    assert built == [("gemini", "gem-x", False), ("ollama", "local-model", False)]


def test_structured_timeout_falls_back(configure, monkeypatch):
    langchain_backend(configure)

    class Slow(FakeLLM):
        async def ainvoke(self, messages):
            await asyncio.sleep(5)

    def factory(provider, model, *, json_mode=False):
        if provider == "gemini":
            return Slow()
        return FakeLLM(structured=Card(title="local", score=2))

    monkeypatch.setattr(llm, "_chat_model", factory)
    result = asyncio.run(llm.generate_structured(MESSAGES, Card, "gemini", "gem-x",
                                                 timeout=0.05, fallback=FALLBACK))
    assert result == Card(title="local", score=2)


def test_provider_error_retryable_flag():
    assert ProviderError("x").retryable is False
    assert llm._langchain_error(openai_status_error(502), "openai", llm._CHAT).retryable is True
    assert llm._langchain_error(openai_status_error(401), "openai", llm._CHAT).retryable is False
    assert llm._langchain_error(ValueError("bad json"), "openai", llm._CHAT).retryable is False
