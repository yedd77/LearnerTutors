"""POST /admin/test-configuration: independent live checks of unsaved Moodle settings.

All offline: the chat-model factory and the image HTTP call are replaced with fakes.
"""
import asyncio
import base64
import logging

import httpx
import pytest
import requests

import app.services.configuration_check as checks
import app.services.illustrations as illustrations
import app.services.llm_provider as llm

URL = "/api/v1/admin/test-configuration"
OPENAI_KEY = "sk-unsaved-openai-" + "a" * 30
STORED_KEY = "sk-stored-openai-" + "s" * 30
IMAGE_KEY = "sk-unsaved-image-" + "c" * 30
GEMINI_KEY = "gemini-unsaved-key-" + "g" * 30
FULL = {
    "llm_provider": "openai", "llm_model": "gpt-test",
    "fallback_model": "llama-test",
    "image_provider": "openai", "image_model": "image-test",
    "openai_api_key": OPENAI_KEY, "openai_image_api_key": IMAGE_KEY,
}


class FakeChat:
    """Stand-in LangChain chat model; ``error`` is raised instead of replying."""

    def __init__(self, error=None, reply="OK", delay=0.0):
        self.error, self.reply, self.delay = error, reply, delay

    async def ainvoke(self, messages):
        from langchain_core.messages import AIMessage

        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return AIMessage(content=self.reply)


def openai_status_error(status: int, body: str = '{"error":"nope"}'):
    import openai

    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx.Response(status, text=body, request=request)
    return openai.APIStatusError("provider failed", response=response, body=None)


@pytest.fixture
def chat_models(monkeypatch):
    """Fake model factory. ``behaviour[provider]`` builds the FakeChat for that provider."""
    built: list[dict] = []
    behaviour = {"openai": FakeChat, "gemini": FakeChat, "ollama": FakeChat}

    def fake_chat_model(provider, model, *, json_mode=False, settings=None):
        built.append({"provider": provider, "model": model, "settings": settings})
        return behaviour[provider]()

    monkeypatch.setattr(llm, "_chat_model", fake_chat_model)
    return {"built": built, "behaviour": behaviour}


class FakeImageResponse:
    def __init__(self, status=200, text=""):
        self.status_code, self.text = status, text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self):
        return {"data": [{"b64_json": base64.b64encode(b"\x89PNG fake").decode()}]}


@pytest.fixture
def image_api(monkeypatch):
    """Fake OpenAI images endpoint. Set ``.response`` to change the reply."""
    class Handle:
        calls: list[dict] = []
        response = FakeImageResponse()

    handle = Handle()
    handle.calls = []

    def fake_post(url, headers, json, timeout):
        handle.calls.append({"url": url, "headers": headers, "json": json})
        return handle.response

    monkeypatch.setattr(illustrations.requests, "post", fake_post)
    return handle


def results(response):
    assert response.status_code == 200, response.text
    return response.json()["results"]


def test_requires_api_key(client, chat_models, image_api):
    response = client.post(URL, json=FULL)
    assert response.status_code == 401
    assert chat_models["built"] == [] and image_api.calls == []


def test_all_checks_pass(client, auth, chat_models, image_api):
    body = results(client.post(URL, headers=auth, json=FULL))
    assert body == {
        "authentication": {"status": "pass", "message": "Backend accepted the API key."},
        "llm": {"status": "pass", "message": "Main LLM is responding."},
        "fallback_model": {"status": "pass", "message": "Ollama fallback model is responding."},
        "image_generation": {"status": "pass", "message": "Image generation is working."},
        "embedding": {"status": "skipped", "message": "Embedding check not requested."},
    }
    calls = {(c["provider"], c["model"]) for c in chat_models["built"]}
    assert calls == {("openai", "gpt-test"), ("ollama", "llama-test")}
    assert image_api.calls[0]["json"]["model"] == "image-test"
    assert image_api.calls[0]["json"]["quality"] == "low"


def test_unsaved_keys_are_used_but_never_saved_or_returned(client, auth, chat_models,
                                                           image_api, runtime_path):
    response = client.post(URL, headers=auth, json=FULL)
    openai_call = next(c for c in chat_models["built"] if c["provider"] == "openai")
    assert openai_call["settings"].openai_api_key == OPENAI_KEY
    assert image_api.calls[0]["headers"]["Authorization"] == f"Bearer {IMAGE_KEY}"
    assert not runtime_path.exists()
    for secret in (OPENAI_KEY, IMAGE_KEY):
        assert secret not in response.text


def test_blank_key_falls_back_to_stored_key(client, auth, chat_models, image_api):
    saved = client.post("/api/v1/admin/config", headers=auth,
                        json={"llm_provider": "openai", "llm_model": "saved-model",
                              "openai_api_key": STORED_KEY})
    assert saved.status_code == 200
    body = results(client.post(URL, headers=auth, json={**FULL, "openai_api_key": "  "}))
    assert body["llm"]["status"] == "pass"
    openai_call = next(c for c in chat_models["built"] if c["provider"] == "openai")
    assert openai_call["settings"].openai_api_key == STORED_KEY
    assert openai_call["model"] == "gpt-test"  # the unsaved model, not "saved-model"


def test_one_failure_does_not_stop_the_others(client, auth, chat_models, image_api):
    secret_body = '{"error": {"message": "Incorrect API key provided: sk-...secret"}}'
    chat_models["behaviour"]["openai"] = lambda: FakeChat(error=openai_status_error(401, secret_body))
    body = results(client.post(URL, headers=auth, json=FULL))
    assert body["llm"] == {"status": "fail",
                           "message": "Main LLM (OpenAI) rejected the API key (HTTP 401)."}
    assert body["fallback_model"]["status"] == "pass"
    assert body["image_generation"]["status"] == "pass"


@pytest.mark.parametrize("status, expected", [
    (404, "Ollama fallback model: the model was not found"),
    (429, "rate-limited"),
    (503, "temporarily unavailable"),
    (400, "rejected the request (HTTP 400)"),
])
def test_http_failures_get_safe_messages(client, auth, chat_models, image_api, status, expected):
    chat_models["behaviour"]["ollama"] = lambda: FakeChat(
        error=openai_status_error(status, "raw provider detail"))
    body = results(client.post(URL, headers=auth, json=FULL))
    assert body["fallback_model"]["status"] == "fail"
    assert expected in body["fallback_model"]["message"]
    assert "raw provider detail" not in body["fallback_model"]["message"]


def test_unreachable_ollama(client, auth, chat_models, image_api):
    chat_models["behaviour"]["ollama"] = lambda: FakeChat(error=httpx.ConnectError("refused"))
    body = results(client.post(URL, headers=auth, json=FULL))
    assert body["fallback_model"] == {"status": "fail",
                                      "message": "Ollama fallback model could not be reached."}


def test_empty_reply_fails(client, auth, chat_models, image_api):
    chat_models["behaviour"]["openai"] = lambda: FakeChat(reply="   ")
    body = results(client.post(URL, headers=auth, json=FULL))
    assert body["llm"] == {"status": "fail",
                           "message": "Main LLM (OpenAI) returned an empty response."}


def test_timeout_fails(client, auth, chat_models, image_api, monkeypatch):
    monkeypatch.setattr(checks, "LLM_CHECK_TIMEOUT_SECONDS", 0.05)
    chat_models["behaviour"]["openai"] = lambda: FakeChat(delay=1.0)
    body = results(client.post(URL, headers=auth, json=FULL))
    assert body["llm"] == {"status": "fail", "message": "Main LLM (OpenAI) did not respond in time."}
    assert body["fallback_model"]["status"] == "pass"


def test_blank_optional_models_are_skipped(client, auth, chat_models, image_api):
    body = results(client.post(URL, headers=auth,
                               json={**FULL, "fallback_model": " ", "image_model": ""}))
    assert body["fallback_model"] == {"status": "skipped",
                                      "message": "No fallback model is configured."}
    assert body["image_generation"] == {"status": "skipped",
                                        "message": "No image generation model is configured."}
    assert body["llm"]["status"] == "pass"
    assert {c["provider"] for c in chat_models["built"]} == {"openai"}
    assert image_api.calls == []


def test_missing_keys_fail_without_network_calls(client, auth, chat_models, image_api):
    body = results(client.post(URL, headers=auth, json={
        "llm_provider": "gemini", "llm_model": "gemini-test",
        "image_provider": "openai", "image_model": "image-test",
    }))
    assert body["llm"] == {"status": "fail", "message": "No Gemini API key is configured."}
    assert body["image_generation"]["status"] == "fail"
    assert "No OPENAI image API key" in body["image_generation"]["message"]
    assert chat_models["built"] == [] and image_api.calls == []


def test_whitespace_main_model_fails(client, auth, chat_models, image_api):
    body = results(client.post(URL, headers=auth, json={**FULL, "llm_model": "   "}))
    assert body["llm"] == {"status": "fail", "message": "No model is selected."}


def test_reuse_with_mismatched_providers_fails_image_only(client, auth, chat_models, image_api):
    body = results(client.post(URL, headers=auth, json={
        **FULL, "image_provider": "gemini", "reuse_chat_api_key": True,
    }))
    assert body["image_generation"]["status"] == "fail"
    assert "only be reused" in body["image_generation"]["message"]
    assert body["llm"]["status"] == "pass"


def test_reuse_uses_unsaved_chat_key_for_images(client, auth, chat_models, image_api):
    body = results(client.post(URL, headers=auth, json={
        **FULL, "openai_image_api_key": None, "reuse_chat_api_key": True,
    }))
    assert body["image_generation"]["status"] == "pass"
    assert image_api.calls[0]["headers"]["Authorization"] == f"Bearer {OPENAI_KEY}"


def test_image_failure_is_safe_and_logged_redacted(client, auth, chat_models, image_api, caplog):
    image_api.response = FakeImageResponse(401, f"bad key {IMAGE_KEY}")
    with caplog.at_level(logging.WARNING, logger="app.services.configuration_check"):
        body = results(client.post(URL, headers=auth, json=FULL))
    assert body["image_generation"] == {
        "status": "fail", "message": "Image generation (OpenAI) rejected the API key (HTTP 401)."}
    assert "config_check.fail check=image_generation" in caplog.text
    assert IMAGE_KEY not in caplog.text
    assert body["llm"]["status"] == "pass"


def test_unexpected_error_in_one_check_is_contained(client, auth, chat_models, image_api,
                                                    monkeypatch):
    async def broken(settings):
        raise KeyError("bug")

    monkeypatch.setattr(checks, "check_image_generation", broken)
    body = results(client.post(URL, headers=auth, json=FULL))
    assert body["image_generation"]["status"] == "fail"
    assert body["llm"]["status"] == "pass" and body["fallback_model"]["status"] == "pass"


def test_short_key_is_rejected_without_echo(client, auth, chat_models, image_api):
    response = client.post(URL, headers=auth, json={**FULL, "gemini_api_key": "short-secret"})
    assert response.status_code == 422
    assert "short-secret" not in response.text
    assert chat_models["built"] == []


def test_chat_model_uses_supplied_settings(configure):
    """The real factory honours a settings override instead of the stored configuration."""
    configure(OPENAI_API_KEY="sk-env-" + "e" * 30)
    settings = llm.get_settings().model_copy(update={"openai_api_key": OPENAI_KEY})
    model = llm._chat_model("openai", "gpt-x", settings=settings)
    assert model.openai_api_key.get_secret_value() == OPENAI_KEY
    assert llm._chat_model("openai", "gpt-x").openai_api_key.get_secret_value().startswith("sk-env-")


# --------------------------------------------------------------------------- embedding check
import app.services.embedding_provider as embeddings  # noqa: E402

EMBED_KEY = "sk-unsaved-embed-" + "k" * 30


class FakeEmbeddingResponse:
    def __init__(self, status=200, vectors=None, text=""):
        self.status_code, self.text = status, text
        self.vectors = [[0.1, 0.2, 0.3]] if vectors is None else vectors

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self):
        return {"data": [{"index": i, "embedding": v} for i, v in enumerate(self.vectors)]}


@pytest.fixture
def embedding_api(monkeypatch):
    """Fake OpenAI /embeddings endpoint. Set ``.response`` (or ``.error``) to change the reply."""
    class Handle:
        response = FakeEmbeddingResponse()
        error = None

    handle = Handle()
    handle.calls = []
    # illustrations and embedding_provider share the requests module: hand
    # non-embedding calls to whatever fake was installed before (image_api).
    previous = embeddings.requests.post

    def fake_post(url, headers=None, json=None, timeout=None):
        if not url.endswith("/embeddings"):
            return previous(url, headers=headers, json=json, timeout=timeout)
        handle.calls.append({"url": url, "headers": headers, "json": json})
        if handle.error is not None:
            raise handle.error
        return handle.response

    monkeypatch.setattr(embeddings.requests, "post", fake_post)
    return handle


EMBED = {**FULL, "test_embedding": True, "embedding_provider": "openai",
         "embedding_model": "embed-unsaved", "openai_embedding_api_key": EMBED_KEY}


def test_embedding_check_is_skipped_unless_requested(client, auth, chat_models, image_api,
                                                     embedding_api):
    body = results(client.post(URL, headers=auth, json={**EMBED, "test_embedding": False}))
    assert body["embedding"] == {"status": "skipped", "message": "Embedding check not requested."}
    assert embedding_api.calls == []


def test_embedding_check_passes_with_unsaved_values(client, auth, chat_models, image_api,
                                                    embedding_api, runtime_path):
    response = client.post(URL, headers=auth, json=EMBED)
    body = results(response)
    assert body["embedding"] == {
        "status": "pass", "message": "openai embed-unsaved returned a 3-dimension vector."}
    call = embedding_api.calls[0]
    assert call["json"] == {"model": "embed-unsaved", "input": ["configuration test"]}
    assert call["headers"]["Authorization"] == f"Bearer {EMBED_KEY}"
    assert EMBED_KEY not in response.text
    assert not runtime_path.exists()            # nothing saved


def test_blank_embedding_provider_tests_the_env_settings(client, auth, chat_models, image_api,
                                                         embedding_api):
    payload = {**FULL, "test_embedding": True, "embedding_provider": "",
               "embedding_model": "ignored", "openai_api_key": OPENAI_KEY}
    body = results(client.post(URL, headers=auth, json=payload))
    assert body["embedding"]["message"] == "openai test-embed returned a 3-dimension vector."
    # No embedding key anywhere: falls back to the chat key, as before.
    assert embedding_api.calls[0]["headers"]["Authorization"] == f"Bearer {OPENAI_KEY}"


def test_blank_embedding_model_uses_provider_default(client, auth, chat_models, image_api,
                                                     embedding_api):
    body = results(client.post(URL, headers=auth, json={**EMBED, "embedding_model": ""}))
    assert body["embedding"]["status"] == "pass"
    assert embedding_api.calls[0]["json"]["model"] == "test-embed"


def test_stored_embedding_key_is_used_when_blank(client, auth, chat_models, image_api,
                                                 embedding_api):
    saved = client.post("/api/v1/admin/config", headers=auth,
                        json={"llm_provider": "openai", "llm_model": "m",
                              "openai_embedding_api_key": EMBED_KEY})
    assert saved.status_code == 200
    body = results(client.post(URL, headers=auth, json={**EMBED, "openai_embedding_api_key": " "}))
    assert body["embedding"]["status"] == "pass"
    assert embedding_api.calls[0]["headers"]["Authorization"] == f"Bearer {EMBED_KEY}"


@pytest.mark.parametrize("status, expected", [
    (401, "Embedding (OpenAI) rejected the API key (HTTP 401)."),
    (404, "Embedding (OpenAI): the model was not found or is not available (HTTP 404)."),
    (429, "Embedding (OpenAI) is rate-limited or out of quota (HTTP 429)."),
])
def test_embedding_http_failure_is_safe(client, auth, chat_models, image_api, embedding_api,
                                        caplog, status, expected):
    caplog.set_level(logging.INFO)
    embedding_api.response = FakeEmbeddingResponse(
        status, text=f'{{"error": "bad key {EMBED_KEY}"}}')
    response = client.post(URL, headers=auth, json=EMBED)
    body = results(response)
    assert body["embedding"]["status"] == "fail"
    assert body["embedding"]["message"].startswith(expected)
    assert body["llm"]["status"] == "pass"       # other checks are unaffected
    assert EMBED_KEY not in response.text
    # The provider body (echoing the key) is logged server-side, redacted.
    assert "config_check.fail check=embedding" in caplog.text
    assert EMBED_KEY not in caplog.text


def test_embedding_unreachable(client, auth, chat_models, image_api, embedding_api):
    embedding_api.error = requests.ConnectionError("connection refused")
    body = results(client.post(URL, headers=auth, json=EMBED))
    assert body["embedding"] == {"status": "fail", "message": "Embedding (OpenAI) could not be reached."}


def test_embedding_empty_vector_fails(client, auth, chat_models, image_api, embedding_api):
    embedding_api.response = FakeEmbeddingResponse(vectors=[[]])
    body = results(client.post(URL, headers=auth, json=EMBED))
    assert body["embedding"] == {"status": "fail", "message": "Embedding (OpenAI) returned no usable vector."}


def test_embedding_wrong_vector_count_fails(client, auth, chat_models, image_api, embedding_api):
    embedding_api.response = FakeEmbeddingResponse(vectors=[[0.1], [0.2]])
    body = results(client.post(URL, headers=auth, json=EMBED))
    assert body["embedding"] == {
        "status": "fail", "message": "Embedding (OpenAI): OpenAI returned an unexpected embedding count."}


def test_embedding_missing_key_fails_without_network(client, auth, chat_models, image_api,
                                                     embedding_api):
    payload = {**FULL, "test_embedding": True, "embedding_provider": "gemini"}
    body = results(client.post(URL, headers=auth, json=payload))
    assert body["embedding"] == {"status": "fail",
                                 "message": "No Gemini API key is configured for embeddings."}
    assert embedding_api.calls == []


def test_embedding_timeout_fails(client, auth, chat_models, image_api, monkeypatch):
    def slow(*args, **kwargs):
        import time
        time.sleep(0.5)
        return [[0.1]]

    monkeypatch.setattr(checks, "EMBEDDING_CHECK_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(embeddings, "embed_texts", slow)
    body = results(client.post(URL, headers=auth, json=EMBED))
    assert body["embedding"] == {"status": "fail", "message": "Embedding (OpenAI) did not respond in time."}


def test_short_embedding_key_is_rejected_without_echo(client, auth, chat_models, image_api):
    response = client.post(URL, headers=auth, json={**EMBED, "gemini_embedding_api_key": "short-secret"})
    assert response.status_code == 422
    assert "short-secret" not in response.text


def test_embedding_reuse_passes_with_the_chat_key(client, auth, chat_models, image_api, embedding_api):
    saved = client.post("/api/v1/admin/config", headers=auth,
                        json={"llm_provider": "openai", "llm_model": "m",
                              "openai_embedding_api_key": EMBED_KEY})
    assert saved.status_code == 200      # a stored embedding key that reuse must ignore
    payload = {**FULL, "test_embedding": True, "embedding_provider": "openai",
               "embedding_reuse_chat_api_key": True}
    body = results(client.post(URL, headers=auth, json=payload))
    assert body["embedding"]["status"] == "pass"
    assert embedding_api.calls[0]["headers"]["Authorization"] == f"Bearer {OPENAI_KEY}"


def test_embedding_reuse_mismatch_fails_only_the_embedding_check(client, auth, chat_models,
                                                                image_api, embedding_api):
    payload = {**FULL, "test_embedding": True, "embedding_provider": "gemini",
               "embedding_reuse_chat_api_key": True}
    response = client.post(URL, headers=auth, json=payload)
    body = results(response)
    assert body["embedding"] == {"status": "fail", "message": (
        "The chat API key can only be reused when chat and embedding providers are the same.")}
    assert body["llm"]["status"] == "pass" and body["image_generation"]["status"] == "pass"
    assert embedding_api.calls == []


def test_embedding_reuse_with_ollama_fails(client, auth, chat_models, image_api, embedding_api):
    payload = {**FULL, "llm_provider": "ollama", "test_embedding": True,
               "embedding_provider": "ollama", "embedding_reuse_chat_api_key": True}
    body = results(client.post(URL, headers=auth, json=payload))
    assert body["embedding"] == {"status": "fail", "message": (
        "The chat API key cannot be reused with Ollama, which uses no API key.")}


def test_embedding_reuse_missing_chat_key_fails_without_network(client, auth, chat_models,
                                                               image_api, embedding_api):
    payload = {**FULL, "llm_provider": "gemini", "test_embedding": True,
               "embedding_provider": "gemini", "embedding_reuse_chat_api_key": True,
               "gemini_embedding_api_key": GEMINI_KEY}   # ignored under reuse
    response = client.post(URL, headers=auth, json=payload)
    body = results(response)
    assert body["embedding"] == {"status": "fail", "message": (
        "No Gemini chat API key is configured to reuse for embeddings.")}
    assert embedding_api.calls == []
    assert GEMINI_KEY not in response.text


def test_embedding_reuse_flag_is_ignored_when_env_is_in_charge(client, auth, chat_models,
                                                               image_api, embedding_api):
    payload = {**FULL, "test_embedding": True, "embedding_provider": "",
               "embedding_reuse_chat_api_key": True, "openai_embedding_api_key": EMBED_KEY}
    body = results(client.post(URL, headers=auth, json=payload))
    assert body["embedding"]["status"] == "pass"
    assert embedding_api.calls[0]["headers"]["Authorization"] == f"Bearer {EMBED_KEY}"
