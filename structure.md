# LearnerTutors Backend Structure

FastAPI backend for a Moodle RAG (Retrieval-Augmented Generation) assistant. It uses:

- **FastAPI** for the HTTP API, authenticated with a shared key from the Moodle plugin.
- **OpenAI, Gemini or Ollama** for chat generation through LangChain chat models. Moodle chooses the provider and model on every request.
- **LangGraph** to orchestrate each chat request: classify, rewrite, retrieve, grade, then generate.
- A separately configured **embedding provider/model** for indexing and retrieval (direct HTTP/SDK calls).
- **ChromaDB** for persistent vector storage, with one collection per embedding model.
- **Pydantic Settings** for `.env` configuration, plus an encrypted runtime config that Moodle writes.

Setup, run, test and evaluation commands are in [README.md](README.md).

## Project tree

```text
LearnerTutors/
|
|-- app/
|   |-- config.py               Settings, runtime config loading, tracing policy, calibrated thresholds
|   |-- main.py                 FastAPI app, lifespan (Chroma warm-up / pool shutdown), /health, /ready
|   |-- prompt_builder.py       Builds chat messages from core + tool prompts
|   |-- prompt_registry.py      Prompt tool definitions (slash commands, top_k)
|   |-- prompt_router.py        Picks a tool from tool_id, slash command or patterns
|   |-- schemas.py              Pydantic request/response models
|   |-- security.py             X-RAG-API-Key authentication dependency
|   |
|   |-- api/
|   |   |-- admin.py            /admin: connection test, runtime provider/model/key config
|   |   |-- chat.py             /chat: tool menu and the chat endpoint (runs the RAG graph)
|   |   |-- documents.py        /documents: index, suggestions, delete, retrieval toggle
|   |   `-- illustrations.py    /illustrations: signed-URL image download
|   |
|   |-- rag_graph/
|   |   |-- context.py          RagContext: immutable per-request access scope + model
|   |   |-- state.py            RagState: graph state (no access-scope fields)
|   |   |-- nodes.py            classify, conversational_reply, rewrite_query, retrieve,
|   |   |                       grade, generate, validate, no_context, finalize
|   |   `-- graph.py            StateGraph wiring and run_rag_graph()
|   |
|   |-- prompts/
|   |   |-- core/               instructions.md, grounding.md, answer_format.md
|   |   |-- tools/              ask, concept_explanation, exact_text_extraction, exam_generation,
|   |   |                       flashcards, note_making, practice_quiz, study_guide, summarization
|   |   `-- graph/              classify.md, rewrite.md, conversational.md
|   |
|   `-- services/
|       |-- chunker.py          Overlapping character-window chunking
|       |-- document_parser.py  PDF/DOCX/PPTX/TXT/MD text extraction
|       |-- embedding_provider.py  Embeddings via OpenAI, Gemini or Ollama
|       |-- illustrations.py    /explain image generation (OpenAI or Gemini) and signed URLs
|       |-- llm_provider.py     The only chat-generation abstraction (LangChain models)
|       |-- retrieval.py        Retrieval thread pool, relevance filter, context/source building
|       |-- suggestion_generator.py  Starter-question suggestions for documents
|       `-- vector_store.py     ChromaDB collection, indexing, course-scoped search
|
|-- docs/PROVIDER_CREDENTIALS.md  How Moodle stores and syncs provider keys
|-- eval/                       RAGAS pipeline (generate_dataset, run_rag_response, run_evaluation)
|-- tests/                      Offline pytest suite (see README)
|-- conductor/                  Track specs, plans and the architecture decision log
|-- storage/                    Runtime data, git-ignored
|   |-- chroma/                 ChromaDB files
|   |-- illustrations/          Generated /explain images
|   `-- runtime_config.json     Moodle-synced provider/model + Fernet-encrypted keys
|
|-- requirements.txt            Runtime dependencies
|-- requirements-dev.txt        + pytest
|-- requirements-eval.txt       + RAGAS evaluation tooling
|-- pytest.ini
|-- README.md
`-- structure.md                This file
```

## Application entry point

### `app/main.py`

- Applies the LangSmith tracing policy (forced off unless `LANGSMITH_TRACING=true`) before LangChain is imported.
- Configures logging and registers the admin, documents, chat and illustrations routers under `/api/v1`.
- **Lifespan:** on startup, opens Chroma and runs its one-time metadata migration. A failure here is not fatal, so `/health` keeps working and `/ready` reports 503. On shutdown, it stops the retrieval thread pool.
- `GET /health` is a liveness check. `GET /ready` opens Chroma and reports the indexed chunk count.
- Middleware gives each request an `X-Request-ID` and logs timings, without logging headers or bodies.

### `app/config.py`

- `Settings` reads infrastructure settings from `.env`: secrets, provider URLs and timeouts, embedding config, Chroma, limits and the RAG graph limits.
- `get_settings()` overlays `storage/runtime_config.json`, the chat/image provider and model plus encrypted keys that Moodle writes. It reloads when the file's modification time changes, which keeps every worker process in sync without a restart.
- `CALIBRATED_MAX_DISTANCE` holds the measured relevance cut-off for each embedding model. `resolved_retrieval_max_distance` applies it unless `RETRIEVAL_MAX_DISTANCE` overrides it.

## API layer

All routers except `/illustrations` require the `X-RAG-API-Key` header.

### `app/api/chat.py`

- `GET /api/v1/chat/tools`: the tool menu for the UI.
- `POST /api/v1/chat`:
  - routes the message to a prompt tool;
  - rejects an empty question after the slash command is removed (422);
  - builds a `RagContext` from the request and runs the RAG graph;
  - for an explicit `/explain`, first asks the chat model whether an image would help (`prompts/graph/illustrate.md`), and generates one only if it says yes; a failed check means no image;
  - returns `ChatResponse`.
  - Provider failures return 502.

### `app/api/documents.py`

- `POST /documents/index` uploads, extracts, chunks, embeds and stores a document. Only course documents are accepted; `visibility` must be `course` (see DECISION-003). It can also generate starter suggestions.
- `POST /documents/suggestions` and `POST /documents/{id}/suggestions/{slot}` regenerate suggestions without re-indexing.
- `DELETE /documents/{id}?course_id=` removes a document's vectors.
- `PATCH /documents/{id}/retrieval` enables or disables retrieval for all of a document's chunks.

### `app/api/admin.py`

- `POST /admin/connection-test` confirms that Moodle can authenticate.
- `GET /admin/config` reports the current provider and model, and whether keys are configured. It returns booleans only, never the keys.
- `POST /admin/config` is how Moodle pushes the chat/image provider, model and keys. The keys are Fernet-encrypted into `storage/runtime_config.json`. Embedding settings are read-only here.

### `app/api/illustrations.py`

- `GET /illustrations/{image_id}?expires=&signature=` serves a generated image. Authorisation comes from the HMAC-signed, expiring URL.

## Security model

- `X-RAG-API-Key` authenticates the Moodle server, not individual users. Moodle remains responsible for enrolment and capability checks.
- Retrieval is always filtered by `course_id`, `retrieval_enabled=true` and `visibility="course"`. Legacy personal chunks are never returned.
- In the chat graph, the access scope (`course_id`, `user_id`, `include_course_documents`) is held only in the immutable `RagContext`. LLM output can change the search text, but never the scope.
- Uploaded and retrieved text is treated as untrusted data in every prompt.

## Chat generation: `app/services/llm_provider.py`

This is the single provider abstraction. Every call builds a new LangChain model (`ChatOpenAI`, `ChatGoogleGenerativeAI` or `ChatOllama`) from the provider and model in the request and the current settings. That's why switching model or key in Moodle applies to the next request. Retries are disabled. Every provider gets an explicit temperature (default 0.5; a negative value sends none). If a model rejects it, as OpenAI reasoning models do, the call is retried once without it and that model uses its provider default for the rest of the process.

- `generate_answer_async(route, context, history, provider, model)` produces chat answers.
- `generate_messages` / `generate_messages_async` take prebuilt message lists (suggestions, conversational replies).
- `generate_structured` / `generate_structured_sync` return typed Pydantic output, using structured output first and falling back to JSON mode.
- Errors are raised as `ProviderError("<provider> returned HTTP <status>: <body>")`, and cancellation propagates.

## Chat workflow: `app/rag_graph/`

```text
START → classify ─conversational─► conversational_reply ───────────────────► finalize → END
            │ course_question
            ▼
        rewrite_query ─► retrieve ─► grade ─relevant─► generate ─► validate ─► finalize
            ▲    │ same query as before   │ poor, retry allowed
            │    └───────────────────────────────────► no_context ──────────► finalize
            └─────────────────────────┘ poor, exhausted ─► no_context
```

- **classify:**
  - Explicit tools and slash commands always take the course path.
  - A regex check spots greetings, thanks and small talk, and clear questions (a trailing "?", a question or study verb, or 6+ words).
  - Only short, ambiguous messages go to an LLM classifier. If classification fails, the message is treated as a course question.
- **conversational_reply:** a short reply from `prompts/graph/conversational.md`, which refuses academic content. There is no retrieval and no sources.
- **rewrite_query:**
  - First turns are not rewritten. Follow-ups with history are condensed into a standalone query, and retries are rewritten with different wording.
  - Exact-text extraction is never rewritten.
  - The rewritten text is used only for search. Generation always uses the original question.
- **retrieve:** calls `vector_store.search_course` on the retrieval thread pool, using the scope from `RagContext`.
- **grade:**
  - Drops chunks above the relevance threshold.
  - Retries only if chunks existed but none were close enough, up to `RAG_MAX_RETRIEVAL_ATTEMPTS` (max 3).
- **generate:** builds the numbered `[S1..Sn]` passages and sources (`services/retrieval.py`), then calls `generate_answer_async`.
- **validate:** a pass-through hook for a future grounding validator.
- **no_context:** returns the standard "I couldn't find relevant text…" message.

Each node logs `rag_graph.node.finish`. Every request ends with `rag_graph.finish … outcome= path= attempts= llm_calls=`. A clear first-turn question makes no extra LLM calls; the worst case is 1 classify plus 2 rewrites.

## Prompt system

- **`prompt_registry.py`** defines each tool: its ID, label, slash command, prompt file and optional `top_k`.
- **`prompt_router.py`** picks a tool in this order:
  1. explicit `tool_id`;
  2. slash command (`/exam`, `/summarize`, …);
  3. rule-based patterns;
  4. the `ask` default.
- **`prompt_builder.py`** combines the core prompts with the selected tool prompt, the context, the history and the question. Prompt files are cached by modification time, so edits apply without a restart. Known bug ERR-001: `_fill` re-substitutes placeholders inside inserted values.
- **`prompts/core/`** holds the shared role, grounding and answer-format instructions.
- **`prompts/tools/`** holds one task prompt per tool.
- **`prompts/graph/`** holds the classifier, query-rewrite and conversational prompts used by the graph.

## Document, embedding and retrieval services

- **`document_parser.py`** extracts page- or slide-aware text:
  - PDF via PyMuPDF;
  - DOCX via python-docx;
  - PPTX via python-pptx;
  - TXT and MD as UTF-8.

  Scanned PDFs need OCR, which isn't included.
- **`chunker.py`** creates overlapping character windows (`CHUNK_SIZE`/`CHUNK_OVERLAP`) and keeps each chunk's page number.
- **`embedding_provider.py`** produces embeddings through OpenAI, Gemini or Ollama, configured only in `.env` with `EMBEDDING_PROVIDER`/`EMBEDDING_MODEL`.
- **`vector_store.py`:**
  - opens the Chroma collection for the active embedding model, named `<prefix>_<provider>_<model>_<hash>`;
  - indexes and replaces documents;
  - searches with course scoping;
  - deletes documents and toggles retrieval;
  - migrates legacy metadata;
  - reports retrieval status.
- **`retrieval.py`:**
  - the lazily created retrieval thread pool;
  - `enabled_only`;
  - `filter_by_relevance` (logs `chat.retrieval.filtered`);
  - `build_context_and_sources`;
  - `NO_CONTEXT_ANSWER`.
- **`suggestion_generator.py`** produces three document-grounded starter questions via `generate_structured_sync`, with a deterministic fallback.
- **`illustrations.py`** generates `/explain` images with the Moodle-selected image provider. Images are stored under `storage/illustrations/` and served via signed URLs.

## Request flows

### Indexing a document

```text
Moodle upload (POST /api/v1/documents/index)
    -> documents.py            validate type/size, visibility=course
    -> document_parser.py      text per page/slide
    -> chunker.py              overlapping chunks
    -> vector_store.py         embedding_provider.embed_texts() → Chroma collection for the
                               active embedding model (replaces any previous chunks of the document)
    -> suggestion_generator.py (optional) three starter questions
```

### Answering a question

```text
Moodle chat (POST /api/v1/chat, provider + model from Moodle)
    -> chat.py                 prompt_router → RagContext(course, user, include flag, provider, model)
    -> rag_graph               classify → [conversational_reply]
                               or rewrite_query → retrieve (vector_store.search_course,
                               embedding_provider) → grade (relevance threshold, ≤ 3 attempts)
                               → generate (prompt_builder + llm_provider) | no_context
    -> chat.py                 optional /explain illustration
    -> ChatResponse            answer, sources, retrieved_chunks, retrieved_contexts
```

### Runtime configuration (provider/model/keys)

```text
Moodle admin settings
    -> POST /api/v1/admin/config          provider, model, keys (authenticated with X-RAG-API-Key)
    -> admin.py                           Fernet-encrypt keys with RUNTIME_CONFIG_ENCRYPTION_KEY
    -> storage/runtime_config.json        atomic replace
    -> config.get_settings()              every worker reloads when the file's mtime changes
    -> llm_provider._chat_model()         next request builds a new model with the new settings
```

Changing the **chat** model needs nothing else. Changing the **embedding** provider or model (in `.env`, then restart) switches to a different Chroma collection, so every document must be re-indexed. The previous collection stays on disk.

## Important limitations

- Scanned PDFs need OCR, which isn't implemented. Tables and complex layouts may not extract perfectly.
- Chunking uses character windows rather than semantic boundaries.
- Indexing runs synchronously inside the request.
- The relevance threshold is calibrated only for `openai/text-embedding-3-small`. Other embedding models are not filtered until they're calibrated (see DECISION-005).
- Answers are not yet checked for grounding. The `validate` node is a placeholder for the deferred adaptive-retrieval-eval track.
- The shared API key does not replace Moodle authorisation.
- Past papers are not separated from other course documents.
