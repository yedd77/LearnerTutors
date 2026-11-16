import json

import pytest
from cryptography.fernet import Fernet

import app.config as config_module
from conftest import reset_settings_cache

OPENAI_KEY = "sk-test-openai-" + "a" * 30
GEMINI_KEY = "gemini-test-key-" + "b" * 30
OPENAI_IMAGE_KEY = "sk-test-image-" + "c" * 30
BASE = {"llm_provider": "openai", "llm_model": "chat-model-x"}


def post(client, auth, **fields):
    return client.post("/api/v1/admin/config", headers=auth, json={**BASE, **fields})


def stored(runtime_path):
    return json.loads(runtime_path.read_text(encoding="utf-8"))


def decrypt(token: str) -> str:
    return Fernet(__import__("os").environ["RUNTIME_CONFIG_ENCRYPTION_KEY"].encode()).decrypt(token.encode()).decode()


def test_update_applies_provider_and_model_immediately(client, auth):
    response = post(client, auth, image_provider="gemini", image_model="img-model")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "updated"
    assert (body["llm_provider"], body["llm_model"]) == ("openai", "chat-model-x")
    assert (body["image_provider"], body["image_model"]) == ("gemini", "img-model")
    reset_settings_cache()
    settings = config_module.get_settings()
    assert settings.llm_provider == "openai" and settings.llm_model == "chat-model-x"


def test_keys_are_encrypted_on_disk_never_plaintext(client, auth, runtime_path):
    post(client, auth, openai_api_key=OPENAI_KEY, gemini_api_key=GEMINI_KEY,
         openai_image_api_key=OPENAI_IMAGE_KEY)
    raw = runtime_path.read_text(encoding="utf-8")
    for secret in (OPENAI_KEY, GEMINI_KEY, OPENAI_IMAGE_KEY):
        assert secret not in raw
    data = stored(runtime_path)
    assert decrypt(data["provider_api_keys"]["openai"]) == OPENAI_KEY
    assert decrypt(data["provider_api_keys"]["gemini"]) == GEMINI_KEY
    assert decrypt(data["image_provider_api_keys"]["openai"]) == OPENAI_IMAGE_KEY


def test_stored_keys_override_environment_values(client, auth, configure):
    configure(OPENAI_API_KEY="env-fallback-key-" + "z" * 20)
    assert config_module.get_settings().openai_api_key.startswith("env-fallback")
    post(client, auth, openai_api_key=OPENAI_KEY)
    reset_settings_cache()
    assert config_module.get_settings().openai_api_key == OPENAI_KEY


def test_chat_and_image_keys_are_stored_separately(client, auth):
    post(client, auth, openai_api_key=OPENAI_KEY, openai_image_api_key=OPENAI_IMAGE_KEY)
    reset_settings_cache()
    settings = config_module.get_settings()
    assert settings.openai_api_key == OPENAI_KEY
    assert settings.image_openai_api_key == OPENAI_IMAGE_KEY


def test_get_config_never_returns_key_values(client, auth):
    post(client, auth, openai_api_key=OPENAI_KEY, gemini_api_key=GEMINI_KEY,
         openai_image_api_key=OPENAI_IMAGE_KEY)
    reset_settings_cache()
    response = client.get("/api/v1/admin/config", headers=auth)
    assert response.status_code == 200
    for secret in (OPENAI_KEY, GEMINI_KEY, OPENAI_IMAGE_KEY):
        assert secret not in response.text
    body = response.json()
    assert body["openai_api_key_configured"] is True
    assert body["gemini_api_key_configured"] is True
    assert body["openai_image_api_key_configured"] is True
    assert body["gemini_image_api_key_configured"] is False


def test_update_response_never_returns_key_values(client, auth):
    response = post(client, auth, openai_api_key=OPENAI_KEY)
    assert OPENAI_KEY not in response.text
    assert response.json()["openai_api_key_configured"] is True


def test_omitted_or_blank_key_preserves_existing_key(client, auth, runtime_path):
    post(client, auth, openai_api_key=OPENAI_KEY)
    before = stored(runtime_path)["provider_api_keys"]["openai"]
    post(client, auth, llm_model="another-model")  # key omitted
    post(client, auth, openai_api_key="   ")  # blank
    assert stored(runtime_path)["provider_api_keys"]["openai"] == before


def test_clear_flag_removes_only_that_key(client, auth, runtime_path):
    post(client, auth, openai_api_key=OPENAI_KEY, gemini_api_key=GEMINI_KEY)
    post(client, auth, clear_openai_api_key=True)
    keys = stored(runtime_path)["provider_api_keys"]
    assert "openai" not in keys and "gemini" in keys


def test_clear_flag_wins_over_a_supplied_key(client, auth, runtime_path):
    post(client, auth, openai_api_key=OPENAI_KEY)
    post(client, auth, openai_api_key=OPENAI_KEY, clear_openai_api_key=True)
    assert "openai" not in stored(runtime_path)["provider_api_keys"]


@pytest.mark.parametrize("bad_key", ["short", "x" * 19, "x" * 513])
def test_key_length_is_validated_without_echoing_the_key(client, auth, bad_key):
    response = post(client, auth, openai_api_key=bad_key)
    assert response.status_code == 422
    assert bad_key not in response.text  # the rejected value must never be echoed back


def test_reusing_chat_key_requires_the_same_provider(client, auth):
    mismatch = post(client, auth, llm_provider="openai", image_provider="gemini", reuse_chat_api_key=True)
    assert mismatch.status_code == 422
    assert "same" in mismatch.json()["detail"]
    ok = post(client, auth, llm_provider="gemini", image_provider="gemini", reuse_chat_api_key=True)
    assert ok.status_code == 200
    assert ok.json()["reuse_chat_api_key"] is True


@pytest.mark.parametrize("fields", [
    {"llm_provider": "mistral"},
    {"llm_model": ""},
    {"llm_model": "   "},
    {"image_provider": "ollama"},
])
def test_invalid_provider_or_model_is_rejected(client, auth, fields):
    assert post(client, auth, **fields).status_code == 422


def test_missing_encryption_key_returns_503_and_writes_nothing(client, auth, monkeypatch, runtime_path):
    monkeypatch.delenv("RUNTIME_CONFIG_ENCRYPTION_KEY")
    response = post(client, auth, openai_api_key=OPENAI_KEY)
    assert response.status_code == 503
    assert "RUNTIME_CONFIG_ENCRYPTION_KEY" in response.json()["detail"]
    assert OPENAI_KEY not in response.text
    assert not runtime_path.exists()


def test_invalid_encryption_key_returns_503(client, auth, monkeypatch):
    monkeypatch.setenv("RUNTIME_CONFIG_ENCRYPTION_KEY", "not-a-valid-fernet-key")
    assert post(client, auth, openai_api_key=OPENAI_KEY).status_code == 503


def test_update_without_keys_needs_no_encryption_key(client, auth, monkeypatch):
    monkeypatch.delenv("RUNTIME_CONFIG_ENCRYPTION_KEY")
    assert post(client, auth).status_code == 200


def test_unreadable_stored_keys_do_not_crash_settings(client, auth, runtime_path, monkeypatch):
    post(client, auth, openai_api_key=OPENAI_KEY)
    monkeypatch.setenv("RUNTIME_CONFIG_ENCRYPTION_KEY", Fernet.generate_key().decode())  # rotated key
    reset_settings_cache()
    settings = config_module.get_settings()  # logs the problem, keeps the service alive
    assert settings.openai_api_key == ""
    assert client.get("/health").status_code == 200


def test_runtime_file_is_written_atomically(client, auth, runtime_path):
    post(client, auth)
    assert runtime_path.is_file()
    assert not runtime_path.with_suffix(".tmp").exists()
    assert set(stored(runtime_path)) >= {"llm_provider", "llm_model", "provider_api_keys"}


def test_settings_cache_refreshes_when_the_file_changes(runtime_path):
    reset_settings_cache()
    assert config_module.get_settings().llm_model == ""
    runtime_path.write_text(json.dumps({"llm_provider": "ollama", "llm_model": "m1"}), encoding="utf-8")
    assert config_module.get_settings().llm_model == "m1"
    stat = runtime_path.stat()
    runtime_path.write_text(json.dumps({"llm_provider": "ollama", "llm_model": "m2"}), encoding="utf-8")
    # Guarantee a distinct mtime even on coarse clocks.
    __import__("os").utime(runtime_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000))
    assert config_module.get_settings().llm_model == "m2"


def test_only_allowed_runtime_fields_are_applied(runtime_path):
    runtime_path.write_text(json.dumps({
        "llm_model": "ok-model", "rag_api_key": "attacker-controlled-" + "q" * 20, "chroma_path": "C:/evil",
    }), encoding="utf-8")
    reset_settings_cache()
    settings = config_module.get_settings()
    assert settings.llm_model == "ok-model"
    assert settings.rag_api_key.startswith("test-api-key-")
    assert "evil" not in settings.chroma_path
