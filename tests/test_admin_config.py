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


@pytest.mark.parametrize("content", ["[1, 2, 3]", "\"just a string\"", "null"])
def test_non_object_runtime_file_does_not_crash_settings(client, auth, runtime_path, content):
    runtime_path.write_text(content, encoding="utf-8")
    reset_settings_cache()
    settings = config_module.get_settings()  # logged, falls back to .env values
    assert settings.llm_model == ""
    # Moodle can still repair the file through the admin endpoint.
    assert post(client, auth).status_code == 200
    assert config_module.get_settings().llm_model == "chat-model-x"


# --------------------------------------------------------------------------- embedding settings
OPENAI_EMBED_KEY = "sk-test-embed-openai-" + "e" * 30
GEMINI_EMBED_KEY = "gemini-test-embed-key-" + "f" * 30


def test_old_client_payload_keeps_env_embedding_and_no_reindex(client, auth, runtime_path):
    body = post(client, auth).json()
    assert (body["embedding_provider"], body["embedding_model"]) == ("openai", "test-embed")
    assert body["embedding_changed"] is False and body["reindex_required"] is False
    assert body["openai_embedding_api_key_configured"] is False
    assert body["gemini_embedding_api_key_configured"] is False
    data = stored(runtime_path)
    assert "embedding_provider" not in data and "embedding_model" not in data
    assert client.get("/api/v1/admin/config", headers=auth).json()["embedding_source"] == "env"


def test_embedding_override_round_trip(client, auth):
    response = post(client, auth, embedding_provider="gemini", embedding_model="  gem-embed-x  ")
    body = response.json()
    assert response.status_code == 200
    assert (body["embedding_provider"], body["embedding_model"]) == ("gemini", "gem-embed-x")
    reset_settings_cache()
    settings = config_module.get_settings()
    assert (settings.embedding_provider, settings.resolved_embedding_model) == ("gemini", "gem-embed-x")
    read = client.get("/api/v1/admin/config", headers=auth).json()
    assert read["embedding_source"] == "moodle"
    assert (read["embedding_provider"], read["resolved_embedding_model"]) == ("gemini", "gem-embed-x")


def test_blank_embedding_model_uses_the_provider_default(client, auth, configure):
    configure(GEMINI_EMBED_MODEL="gemini-default-embed")
    body = post(client, auth, embedding_provider="gemini", embedding_model="").json()
    assert body["embedding_model"] == "gemini-default-embed"


def test_blank_embedding_provider_restores_env_values(client, auth, runtime_path):
    post(client, auth, embedding_provider="ollama", embedding_model="nomic-x")
    body = post(client, auth, embedding_provider="", embedding_model="ignored").json()
    assert (body["embedding_provider"], body["embedding_model"]) == ("openai", "test-embed")
    assert body["reindex_required"] is True          # ollama:nomic-x -> back to .env identity
    assert "embedding_provider" not in stored(runtime_path)
    reset_settings_cache()
    assert config_module.get_settings().embedding_provider == "openai"
    assert client.get("/api/v1/admin/config", headers=auth).json()["embedding_source"] == "env"


@pytest.mark.parametrize("change", [
    {"embedding_provider": "gemini"},
    {"embedding_provider": "openai", "embedding_model": "text-embedding-3-large"},
])
def test_identity_change_requires_reindex(client, auth, change):
    body = post(client, auth, **change).json()
    assert body["embedding_changed"] is True and body["reindex_required"] is True


def test_same_identity_does_not_require_reindex(client, auth):
    # The explicit selection equals the .env identity (openai + its default model).
    body = post(client, auth, embedding_provider="openai", embedding_model="test-embed").json()
    assert body["reindex_required"] is False
    again = post(client, auth, embedding_provider="openai", embedding_model="").json()
    assert again["reindex_required"] is False


def test_key_only_change_does_not_require_reindex(client, auth):
    post(client, auth, embedding_provider="gemini", embedding_model="gem-embed")
    body = post(client, auth, embedding_provider="gemini", embedding_model="gem-embed",
                gemini_embedding_api_key=GEMINI_EMBED_KEY).json()
    assert body["reindex_required"] is False and body["embedding_changed"] is False
    assert body["gemini_embedding_api_key_configured"] is True


def test_chat_or_image_change_does_not_require_reindex(client, auth):
    post(client, auth, embedding_provider="gemini")
    body = post(client, auth, embedding_provider="gemini", llm_model="other-chat",
                image_model="img-2").json()
    assert body["reindex_required"] is False


def test_embedding_identity_tags_only_gemini_dimensions():
    base = config_module.get_settings()
    gemini = base.model_copy(update={"embedding_provider": "gemini", "embedding_model": "g"})
    assert gemini.embedding_identity == f"gemini:g:{base.gemini_embed_dimensions}"
    assert gemini.model_copy(update={"gemini_embed_dimensions": 1536}).embedding_identity == "gemini:g:1536"
    openai = base.model_copy(update={"gemini_embed_dimensions": 1536})
    assert openai.embedding_identity == "openai:test-embed:default"


def test_embedding_keys_are_encrypted_and_separate(client, auth, runtime_path):
    post(client, auth, openai_api_key=OPENAI_KEY, openai_embedding_api_key=OPENAI_EMBED_KEY,
         gemini_embedding_api_key=GEMINI_EMBED_KEY)
    raw = runtime_path.read_text(encoding="utf-8")
    assert OPENAI_EMBED_KEY not in raw and GEMINI_EMBED_KEY not in raw
    data = stored(runtime_path)
    assert decrypt(data["embedding_provider_api_keys"]["openai"]) == OPENAI_EMBED_KEY
    assert decrypt(data["embedding_provider_api_keys"]["gemini"]) == GEMINI_EMBED_KEY
    reset_settings_cache()
    settings = config_module.get_settings()
    assert settings.embedding_openai_api_key == OPENAI_EMBED_KEY
    assert settings.openai_api_key == OPENAI_KEY
    read = client.get("/api/v1/admin/config", headers=auth)
    assert OPENAI_EMBED_KEY not in read.text and GEMINI_EMBED_KEY not in read.text
    assert read.json()["openai_embedding_api_key_configured"] is True


def test_embedding_key_blank_keeps_and_clear_flag_removes(client, auth, runtime_path):
    post(client, auth, openai_embedding_api_key=OPENAI_EMBED_KEY, gemini_embedding_api_key=GEMINI_EMBED_KEY)
    before = stored(runtime_path)["embedding_provider_api_keys"]["openai"]
    post(client, auth, openai_embedding_api_key="   ")
    assert stored(runtime_path)["embedding_provider_api_keys"]["openai"] == before
    body = post(client, auth, clear_openai_embedding_api_key=True).json()
    keys = stored(runtime_path)["embedding_provider_api_keys"]
    assert "openai" not in keys and "gemini" in keys
    assert body["openai_embedding_api_key_configured"] is False
    post(client, auth, gemini_embedding_api_key=GEMINI_EMBED_KEY, clear_gemini_embedding_api_key=True)
    assert stored(runtime_path)["embedding_provider_api_keys"] == {}


@pytest.mark.parametrize("field", ["openai_embedding_api_key", "gemini_embedding_api_key"])
@pytest.mark.parametrize("bad_key", ["short", "x" * 513])
def test_embedding_key_length_is_validated_without_echo(client, auth, field, bad_key):
    response = post(client, auth, **{field: bad_key})
    assert response.status_code == 422
    assert bad_key not in response.text


@pytest.mark.parametrize("fields", [
    {"embedding_provider": "mistral"},
    {"embedding_model": "m" * 161},
])
def test_invalid_embedding_selection_is_rejected(client, auth, fields):
    assert post(client, auth, **fields).status_code == 422


def test_unknown_stored_embedding_provider_is_ignored(runtime_path):
    runtime_path.write_text(json.dumps({"embedding_provider": "evil", "embedding_model": "x"}),
                            encoding="utf-8")
    reset_settings_cache()
    settings = config_module.get_settings()
    assert (settings.embedding_provider, settings.resolved_embedding_model) == ("openai", "test-embed")


# --------------------------------------------------------------------------- embedding chat-key reuse
from app.services.embedding_provider import embedding_api_key  # noqa: E402

MISMATCH = "The chat API key can only be reused when chat and embedding providers are the same."
OLLAMA = "The chat API key cannot be reused with Ollama, which uses no API key."


def test_embedding_reuse_round_trip(client, auth, runtime_path):
    body = post(client, auth, embedding_provider="openai", embedding_reuse_chat_api_key=True).json()
    assert body["embedding_reuse_chat_api_key"] is True
    assert stored(runtime_path)["embedding_reuse_chat_api_key"] is True
    reset_settings_cache()
    assert config_module.get_settings().embedding_reuse_chat_api_key is True
    read = client.get("/api/v1/admin/config", headers=auth).json()
    assert read["embedding_reuse_chat_api_key"] is True


def test_embedding_reuse_uses_chat_key_even_with_a_stored_embedding_key(client, auth):
    post(client, auth, embedding_provider="openai", openai_api_key=OPENAI_KEY,
         openai_embedding_api_key=OPENAI_EMBED_KEY, embedding_reuse_chat_api_key=True)
    reset_settings_cache()
    settings = config_module.get_settings()
    assert settings.embedding_openai_api_key == OPENAI_EMBED_KEY      # kept, but not used
    assert embedding_api_key("openai", settings) == OPENAI_KEY


def test_without_embedding_reuse_the_stored_embedding_key_is_used(client, auth):
    post(client, auth, embedding_provider="openai", openai_api_key=OPENAI_KEY,
         openai_embedding_api_key=OPENAI_EMBED_KEY)
    reset_settings_cache()
    assert embedding_api_key("openai", config_module.get_settings()) == OPENAI_EMBED_KEY


@pytest.mark.parametrize("fields, message", [
    ({"llm_provider": "openai", "embedding_provider": "gemini"}, MISMATCH),
    ({"llm_provider": "ollama", "embedding_provider": "openai"}, OLLAMA),
    ({"llm_provider": "ollama", "embedding_provider": "ollama"}, OLLAMA),
    ({"llm_provider": "openai", "embedding_provider": "ollama"}, OLLAMA),
])
def test_invalid_embedding_reuse_is_422_and_writes_nothing(client, auth, runtime_path, fields, message):
    response = post(client, auth, embedding_reuse_chat_api_key=True, **fields)
    assert response.status_code == 422
    assert response.json()["detail"] == message
    assert not runtime_path.exists()


def test_embedding_reuse_is_ignored_when_env_is_in_charge(client, auth, runtime_path):
    # Mismatched providers would be a 422, but with embedding_provider="" the flag is moot.
    response = post(client, auth, llm_provider="ollama", embedding_provider="",
                    embedding_reuse_chat_api_key=True)
    assert response.status_code == 200
    assert response.json()["embedding_reuse_chat_api_key"] is False
    assert stored(runtime_path)["embedding_reuse_chat_api_key"] is False


def test_toggling_only_embedding_reuse_does_not_require_reindex(client, auth):
    post(client, auth, embedding_provider="openai", embedding_model="e1")
    on = post(client, auth, embedding_provider="openai", embedding_model="e1",
              embedding_reuse_chat_api_key=True).json()
    off = post(client, auth, embedding_provider="openai", embedding_model="e1").json()
    for body in (on, off):
        assert body["reindex_required"] is False and body["embedding_changed"] is False
    assert (on["embedding_reuse_chat_api_key"], off["embedding_reuse_chat_api_key"]) == (True, False)


def test_old_client_payload_reports_embedding_reuse_off(client, auth, runtime_path):
    body = post(client, auth).json()
    assert body["embedding_reuse_chat_api_key"] is False
    assert stored(runtime_path)["embedding_reuse_chat_api_key"] is False
    reset_settings_cache()
    assert config_module.get_settings().embedding_reuse_chat_api_key is False
