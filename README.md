# LearnerTutors backend

FastAPI backend for **LearnerTutors**, a course-aware AI learning assistant inside Moodle. It indexes lecturers' course documents and answers student questions grounded in those documents, with source citations.

The Moodle plugin ([LearnerTutors-FE](https://github.com/yedd77/LearnerTutors-FE), installed as `local_ragassistant`) is the only client. It authenticates every request with a shared secret, and it decides which chat provider and model to use. This backend never picks a generation model by itself.

- **Chat providers:** OpenAI, Gemini or local Ollama. Moodle sends the provider and model with every request and pushes the API keys through `/api/v1/admin/config`. Switching model in Moodle applies to the next request, with no restart.
- **Retrieval:** ChromaDB, filtered by course and the per-document retrieval switch. A calibrated relevance threshold means off-topic questions get a "couldn't find relevant text" reply instead of an invented answer.
- **Chat workflow:** a LangGraph flow. Greetings get a short reply without retrieval, follow-ups are rewritten using the chat history, and weak results get one retry with a rewritten query. See [structure.md](structure.md).

## Prerequisites

- Python 3.12 (developed on 3.12.8).
- One embedding provider: an OpenAI or Gemini key, or [Ollama](https://ollama.com) running locally with an embedding model such as `nomic-embed-text`.
- Ollama, only if you want local chat models (e.g. `ollama pull qwen2.5:7b-instruct`) or the default evaluation setup.

## Install

```bash
python -m venv .venv
# Windows (Git Bash): source .venv/Scripts/activate   PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux:        source .venv/bin/activate

pip install -r requirements.txt          # run the API
pip install -r requirements-dev.txt      # + tests (pytest)
pip install -r requirements-eval.txt     # + RAGAS evaluation tooling (optional)
```

## Configure `.env`

Create `.env` in the repository root. Only the first three settings are required, and everything else has a working default.

```ini
# Shared Moodle-to-backend secret, at least 32 characters. Put the same value in Moodle's
# "Application API key" setting.
RAG_API_KEY=

# Fernet key that encrypts the provider keys Moodle sends (storage/runtime_config.json).
# Generate it once and keep it: changing it makes the saved keys unreadable.
RUNTIME_CONFIG_ENCRYPTION_KEY=

# Embedding model used for indexing and retrieval (openai | gemini | ollama).
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
```

Generate the two secrets:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"                               # RAG_API_KEY
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"  # RUNTIME_CONFIG_ENCRYPTION_KEY
```

**Chat provider, model and keys come from Moodle** (Site administration → Plugins → Local plugins → Course Learning Assistant). See [docs/PROVIDER_CREDENTIALS.md](docs/PROVIDER_CREDENTIALS.md). `OPENAI_API_KEY` and `GEMINI_API_KEY` in `.env` are only fallbacks; a key saved in Moodle takes precedence. Embeddings use the same key for their provider, so an OpenAI embedding setup needs an OpenAI key saved in Moodle or in `.env`.

**Changing the embedding provider or model is a migration, not a setting.** Each embedding model has its own Chroma collection, because vectors from different models can't be compared. After a change, every document must be re-indexed before it can be found again. The old collection stays on disk, so switching back makes it usable again. The embedding model is deliberately configurable only in `.env`, not from Moodle.

### Optional settings (defaults shown)

| Setting | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server for chat and/or embeddings |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `OPENAI_TIMEOUT_SECONDS` / `GEMINI_TIMEOUT_SECONDS` / `OLLAMA_TIMEOUT_SECONDS` | 120 / 180 / 180 | Provider request timeouts |
| `TOP_K` | 5 | Chunks retrieved per question (some tools use 10) |
| `RETRIEVAL_MAX_DISTANCE` | calibrated per embedding model | Relevance cut-off. `text-embedding-3-small` uses 0.70; uncalibrated models are not filtered. `0` disables it. See DECISION-005 in `conductor/decision-log.md` |
| `RAG_MAX_RETRIEVAL_ATTEMPTS` | 2 (max 3) | First search plus rewritten retries |
| `RAG_AUX_TIMEOUT_SECONDS` | 20 | Timeout for the classify/rewrite LLM calls. On timeout, the request carries on without them |
| `CHROMA_PATH` / `CHROMA_COLLECTION` | `./storage/chroma` / `ragassistant_chunks` | Vector store location and name prefix |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 900 / 150 | Characters per chunk. Changing these needs a re-index |
| `MAX_UPLOAD_MB` / `MAX_PAGES` / `MAX_CHUNKS` | 20 / 300 / 2000 | Upload limits |
| `RETRIEVAL_WORKERS` | 64 | Thread pool for synchronous Chroma and embedding calls |
| `LANGSMITH_TRACING` | `false` | Keep this off unless you're allowed to send student questions to LangSmith |
| `LOG_LEVEL` / `APP_ENV` | `INFO` / `development` | `APP_ENV=production` hides `/docs` |

## Run

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- `GET /health`: liveness check, no authentication.
- `GET /ready`: opens Chroma and returns the indexed chunk count, or `503` if the vector store is unavailable.
- `GET /docs`: interactive API reference (not available when `APP_ENV=production`).

All `/api/v1/*` routes require the `X-RAG-API-Key: <RAG_API_KEY>` header. Supported upload formats are PDF, DOCX, PPTX, TXT and Markdown. Scanned PDFs need OCR first.

## Test

```bash
python -m pytest -q
```

The suite runs fully offline in about 15–25 s. Provider calls are stubbed, a network guard fails any test that tries to reach the internet, and Chroma and the runtime config use temporary directories. Your `.env` and `storage/` are never touched.

## Evaluate (RAGAS)

A three-step pipeline. Install `requirements-eval.txt` first. Steps 1 and 3 use local Ollama by default.

```bash
# 1. Build a question/reference dataset from eval/test_docs/ → eval/output/evaluation_dataset.csv
RAGAS_LLM_MODEL=qwen2.5:7b-instruct python eval/generate_dataset.py

# 2. Ask the running backend every question → eval/output/rag_responses.csv
#    Index eval/test_docs/ into the course first. --provider/--model default to the model selected in Moodle.
python eval/run_rag_response.py --course-id 2 --user-id 10 --endpoint /api/v1/chat --limit 5

# 3. Score faithfulness, answer relevancy, context precision and context recall → eval/results/
RAGAS_LLM_MODEL=qwen2.5:7b-instruct python eval/run_evaluation.py
```

**Cost note:** step 2 sends real chat requests. Each question also costs one query embedding with your embedding provider, plus up to 3 extra LLM calls for classifying and rewriting. With OpenAI or Gemini selected, that uses paid API credit. Use `--limit` for a trial run. Steps 1 and 3 call whatever `RAGAS_BASE_URL` points to (default: local Ollama, free).

In PowerShell, set the variable first: `$env:RAGAS_LLM_MODEL="qwen2.5:7b-instruct"`.

## More documentation

- [structure.md](structure.md): module layout, request flows and the chat graph.
- [docs/PROVIDER_CREDENTIALS.md](docs/PROVIDER_CREDENTIALS.md): how Moodle stores and syncs provider keys.
- `conductor/decision-log.md`: architecture decisions (LangChain/LangGraph, relevance threshold, personal uploads).
