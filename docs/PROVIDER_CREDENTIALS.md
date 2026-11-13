# Dynamic AI provider credentials

Moodle administrators configure chat and image generation independently from:

**Site administration → Plugins → Local plugins → Course Learning Assistant**

The settings page has separate sections for **Text generation** and **Image generation**. Chat and image models can be selected independently.

The keys are separate from `RAG_API_KEY`, which remains the shared Moodle-to-FastAPI authentication secret. `RUNTIME_CONFIG_ENCRYPTION_KEY` remains a backend-only Fernet master key; it is not entered in Moodle settings.

## One-time backend setup

1. Install the backend dependencies:
   `pip install -r requirements.txt`
2. Generate a persistent encryption key:
   `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
3. Add it to the backend `.env`:
   `RUNTIME_CONFIG_ENCRYPTION_KEY=<generated-value>`
4. Keep this key outside source control and back it up securely. Do not regenerate it after provider credentials have been saved.

Existing `OPENAI_API_KEY` and `GEMINI_API_KEY` environment variables remain supported as fallback chat credentials. A key managed through Moodle takes precedence.

## Moodle settings

- **Chat provider / Chat model / Provider API key** control text generation. The single chat key field stores the credential for the selected OpenAI or Gemini provider. Ollama does not require a key.
- **Image provider / Image generation model / Image provider API key** control `/explain` illustrations independently. OpenAI and Gemini are supported. Example image models are `gpt-image-1` and `gemini-3.1-flash-image`.
- **Reuse chat provider API key** lets image generation use the chat credential without entering it a second time. Enable it only when the selected chat and image providers are identical. FastAPI validates this condition and rejects mismatched reuse.
- If reuse is disabled, the image key is stored separately from the chat key, even when both providers are the same.
- Blank password fields preserve existing credentials. Saved keys are never rendered back into the browser.

## How updates work

- Moodle stores chat and image provider credentials as separate AES-256-GCM ciphertext values in `config_plugins`. The encryption key is `LOCAL_RAGASSISTANT_ENCRYPTION_KEY`, configured outside the database.
- On the next chat or document-indexing request, Moodle decrypts the keys inside PHP and sends them to the authenticated FastAPI admin configuration endpoint over the configured backend connection.
- FastAPI encrypts the chat and image credentials separately with Fernet before writing them to `storage/runtime_config.json`.
- Runtime configuration is refreshed when the file changes, including across multiple FastAPI worker processes. No FastAPI restart or document reindex is required for provider/model/key changes.
- The admin config read endpoint returns only boolean configured indicators. It never returns provider key values.
- `RAG_API_KEY`, `LOCAL_RAGASSISTANT_ENCRYPTION_KEY`, and `RUNTIME_CONFIG_ENCRYPTION_KEY` have different purposes and must not be reused interchangeably.

If `RUNTIME_CONFIG_ENCRYPTION_KEY` is lost or changed, the encrypted runtime provider keys cannot be decrypted. Restore the original key or re-enter the provider keys in Moodle.
