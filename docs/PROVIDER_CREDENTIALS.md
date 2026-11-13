# Dynamic AI provider credentials

Moodle administrators can manage OpenAI and Google Gemini API keys from:

**Site administration → Plugins → Local plugins → Course Learning Assistant**

The keys are separate from `RAG_API_KEY`, which remains the shared Moodle-to-FastAPI authentication secret.

## One-time backend setup

1. Install the backend dependencies:
   `pip install -r requirements.txt`
2. Generate a persistent encryption key:
   `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
3. Add it to the backend `.env`:
   `RUNTIME_CONFIG_ENCRYPTION_KEY=<generated-value>`
4. Keep this key outside source control and back it up securely. Do not regenerate it after provider credentials have been saved.

Existing `OPENAI_API_KEY` and `GEMINI_API_KEY` environment variables remain supported as fallback credentials. A key managed through Moodle takes precedence.

## How updates work

- Moodle stores each provider key as AES-256-GCM ciphertext in `config_plugins`. The settings page always renders a blank password field; blank means keep the existing value.
- On the next chat or document-indexing request, Moodle decrypts the provider keys inside PHP and sends them to the authenticated FastAPI admin configuration endpoint over the configured backend connection.
- FastAPI encrypts the provider keys with Fernet before writing them to `storage/runtime_config.json`.
- Runtime configuration is refreshed when the file changes, including across multiple FastAPI worker processes. No FastAPI restart is needed.
- The same OpenAI key is used for OpenAI chat, embeddings, and image generation. The Gemini key is used for Gemini chat and embeddings. Ollama does not require a provider API key.
- The admin config read endpoint returns only boolean configured indicators. It never returns provider key values.
- Rotating a provider key does not require document reindexing. If you change the embedding model/provider itself, vector compatibility is a separate concern.

If `RUNTIME_CONFIG_ENCRYPTION_KEY` is lost or changed, the encrypted runtime provider keys cannot be decrypted. Restore the original key or re-enter the provider keys in Moodle.
