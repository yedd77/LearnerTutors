# LearnerTutors Backend Structure

FastAPI backend for a Moodle RAG (Retrieval-Augmented Generation) assistant. It uses:

- **FastAPI** for the HTTP API, authenticated with a shared key from the Moodle plugin.
- **OpenAI, Gemini or Ollama** for chat generation through LangChain chat models. Moodle chooses the provider and model on every request.
- **LangGraph** to orchestrate each chat request: classify, rewrite, retrieve (hybrid dense + keyword), grade, refine (rerank, neighbours), then generate.
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
|   |   |                       grade, refine, generate, validate, no_context, finalize
|   |   `-- graph.py            StateGraph wiring and run_rag_graph()
|   |
|   |-- prompts/
|   |   |-- core/               instructions.md, grounding.md, answer_format.md
|   |   |-- tools/              ask, concept_explanation, exam_generation, flashcards,
|   |   |                       note_making, practice_quiz, study_guide, summarization
|   |   `-- graph/              classify.md, rewrite.md, conversational.md
|   |
|   `-- services/
|       |-- chunker.py          Sentence-aligned overlapping chunks with heading and position
|       |-- configuration_check.py  Live provider checks for POST /admin/test-configuration
|       |-- document_parser.py  PDF/DOCX/PPTX/TXT/MD text extraction, with headings
|       |-- keyword_index.py    Per-course BM25 keyword ranking for hybrid search
|       |-- document_images.py  Figures from documents: filter, store, [IMGn] markers, access-checked serving
|       |-- embedding_provider.py  Embeddings via OpenAI, Gemini or Ollama
|       |-- illustrations.py    /explain image generation (OpenAI or Gemini) and signed URLs
|       |-- llm_provider.py     The only chat-generation abstraction (LangChain models)
|       |-- retrieval.py        Retrieval thread pool, relevance filter, context/source building
|       |-- suggestion_generator.py  Starter-question suggestions for documents
|       `-- vector_store.py     ChromaDB collection, indexing, course-scoped hybrid search
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

- `Settings` reads infrastructure settings from `.env`: secrets, provider URLs and timeouts, the default embedding config, Chroma, limits and the RAG graph limits. A Moodle embedding selection in `runtime_config.json` overrides `EMBEDDING_PROVIDER`/`EMBEDDING_MODEL`.
- `Settings.embedding_identity` (provider, resolved model, Gemini dimensions) names the embedding space. It selects the Chroma collection and decides when a reindex is required.
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
- `GET /admin/config` reports the current provider and model, whether keys are configured, and `embedding_source` (`moodle` or `env`). It returns booleans only, never the keys.
- `POST /admin/config` is how Moodle pushes the chat/image/embedding provider, model and keys. The keys are Fernet-encrypted into `storage/runtime_config.json`.
  - `embedding_provider=""` (the default, and what older clients imply) means no Moodle override: the `.env` embedding settings apply, and any previous override is removed. A blank `embedding_model` means the provider default.
  - Embedding keys (`openai_embedding_api_key`, `gemini_embedding_api_key`) are stored separately. Without one, embeddings use the chat key for that provider.
  - `embedding_reuse_chat_api_key` (like `reuse_chat_api_key` for images) makes embeddings use the chat key and ignore any stored embedding key. It needs matching, non-Ollama chat and embedding providers, otherwise 422 (the configuration test reports a `fail` instead). With `embedding_provider=""` it is ignored and stored as false. It is not part of the embedding identity, so it never triggers a reindex.
  - When the embedding identity changes, the response has `embedding_changed` and `reindex_required` set to true, and Moodle queues a full reindex. A key-only change is not an identity change.
- `POST /admin/test-configuration` tests the settings form as sent, including unsaved values, without saving anything. It takes the `/admin/config` fields plus `fallback_model` and `test_embedding`; blank keys use the stored ones. It makes small real calls to the main LLM, an Ollama fallback model, the image provider and (only when `test_embedding` is true) the embedding provider, at the same time, and returns `results.{authentication, llm, fallback_model, image_generation, embedding}` with `pass`, `fail` or `skipped` and a fixed safe message. One failure does not stop the others; the response is always 200 unless the API key or input is invalid. Provider error text goes only to the server log, with keys redacted.

### `app/api/illustrations.py`

- `GET /illustrations/{image_id}?expires=&signature=` serves a generated image. Authorisation comes from the HMAC-signed, expiring URL.

## Security model

- `X-RAG-API-Key` authenticates the Moodle server, not individual users. Moodle remains responsible for enrolment and capability checks.
- Retrieval is always filtered by `course_id`, `retrieval_enabled=true` and `visibility="course"`. Legacy personal chunks are never returned.
- In the chat graph, the access scope (`course_id`, `user_id`, `include_course_documents`) is held only in the immutable `RagContext`. LLM output can change the search text, but never the scope.
- Uploaded and retrieved text is treated as untrusted data in every prompt.

## Chat generation: `app/services/llm_provider.py`

This is the single provider abstraction. Every call builds a new LangChain model (`ChatOpenAI`, `ChatGoogleGenerativeAI` or `ChatOllama`) from the provider and model in the request and the current settings. That's why switching model or key in Moodle applies to the next request. Retries are disabled. Every provider gets an explicit temperature (default 0.5; a negative value sends none). If a model rejects it, as OpenAI reasoning models do, the call is retried once without it and that model uses its provider default for the rest of the process.

A chat request may also carry an optional `fallback_provider` + `fallback_model` (both or neither). If a chat-path call (classify, rewrite, conversational reply, answer, `/explain` image check) fails with a temporary error (429, 5xx, timeout or connection error), it is tried once on that model and logged as `llm.fallback`. Configuration errors (400/401/403/404) never fall back, and if the fallback also fails the original error is returned.

- `generate_answer_async(route, context, history, provider, model)` produces chat answers.
- `generate_messages` / `generate_messages_async` take prebuilt message lists (suggestions, conversational replies).
- `generate_structured` / `generate_structured_sync` return typed Pydantic output, using structured output first and falling back to JSON mode.
- Errors are raised as `ProviderError("<provider> returned HTTP <status>: <body>")`, and cancellation propagates.

## Chat workflow: `app/rag_graph/`

```text
START → classify ─conversational─► conversational_reply ───────────────────► finalize → END
            │ course_question
            ▼
        rewrite_query ─► retrieve ─► grade ─relevant─► refine ─► generate ─► validate ─► finalize
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
  - Simple first turns are not rewritten. Follow-ups with history are condensed into a standalone query, and retries are rewritten with different wording.
  - Multi-part questions (`is_compound`: "…, and how …", or two question marks) are rewritten too, into a main query plus up to 2 `sub_queries`.
  - The rewritten text is used only for search. Generation always uses the original question.
- **retrieve:** calls `vector_store.hybrid_search` on the retrieval thread pool, using the scope from `RagContext`. Each query runs a dense search and a BM25 keyword search (`keyword_index.py`) for a candidate pool (`RETRIEVAL_CANDIDATES`, at least 2 × top_k, max 50). The lists are fused with reciprocal rank fusion, near-duplicates are dropped, and every candidate gets its smallest cosine distance to any query, so the relevance threshold also applies to keyword hits. Keyword ranking only orders chunk IDs: every chunk is read back from Chroma with the course, retrieval and visibility filter.
- **grade:**
  - Drops chunks above the relevance threshold.
  - Retries only if chunks existed but none were close enough, up to `RAG_MAX_RETRIEVAL_ATTEMPTS` (max 3).
- **refine:**
  - When more than top_k candidates pass the threshold, one LLM call (`prompts/graph/rerank.md`) orders them (`RAG_RERANK`, on by default). The model can only reorder the passages it was shown. Any failure keeps the fused order.
  - Keeps the best top_k.
  - `/summarize` and `/studyguide`: when at least half the hits come from one document (and heading), that document or section is read in order, one passage per page, if it fits in `SUMMARY_CONTEXT_CHARS`.
  - Otherwise the best `RETRIEVAL_NEIGHBOUR_HITS` hits are merged with their neighbouring chunks, and adjacent hits are merged into one passage. A merged passage carries `chunk_count`; `retrieved_chunks` and the `/notes` sufficiency check count chunks, not passages.
- **generate:** attaches document figures (`document_images.attach_images`), builds the numbered `[S1..Sn]` passages (with file, page, section and `[IMGn: caption]` figure markers) and sources (`services/retrieval.py`), calls `generate_answer_async`, then keeps only valid `[IMGn]` markers. The response lists the figures in `images` (omitted when none).
- **validate:** a pass-through hook for a future grounding validator.
- **no_context:** returns the standard "I couldn't find relevant text…" message.

Each node logs `rag_graph.node.finish`. Every request ends with `rag_graph.finish … outcome= path= attempts= llm_calls=`. A clear, single-part first-turn question with at most top_k relevant candidates makes no extra LLM calls; the worst case is 1 classify, 3 rewrites and 1 rerank.

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

  Sections carry `images` (PDF, DOCX, PPTX): each embedded image with its character position in the section text, a caption (a nearby "Figure …" line plus up to 3 continuation lines of its text block, the alt text, or the slide title) and a content hash. PDF figures drawn as vector paths (charts, diagrams) have no picture: when a figure caption (not "Table") has no picture, the drawings next to it (above first, then below; same text column; never past a body-text paragraph; at least 3 path items and 100×60 pt; page rules ignored) plus nearby short labels are rendered to PNG at 2× (max 1600 px) and inserted just before the caption, like an embedded picture. A picture inside such a region is replaced by the rendering. Rotated pages are skipped.

  Sections carry a `heading` when the document has one: PDF lines at least 1.2 × the body font size, DOCX Heading/Title styles, the PPTX slide title, Markdown `#` lines. Scanned PDFs need OCR, which isn't included.
- **`chunker.py`** packs paragraphs and sentences into chunks of up to `CHUNK_SIZE` characters. Consecutive chunks repeat whole sentences of up to `CHUNK_OVERLAP` characters. Each chunk keeps its page and heading, and gets a document-wide `chunk_index`. Images never enter the chunk text: `image_refs` lists the images that follow a chunk (an image inside a chunk's text is anchored to that chunk), and `image_refs_before` on the first chunk lists images before any text.
- **`document_images.py`** (figures in answers):
  - at indexing, drops icons (< 100 px), images used 3+ times (logos, backgrounds) and formats Pillow cannot read (e.g. EMF), converts the rest to PNG (max 1600 px, max 200 per document) and stores them with a `manifest.json` (image ID, anchor chunk, caption, page) under `storage/document_images/<hash of course+document>/`; re-indexing replaces them, deleting a document or a failed re-index removes them;
  - in `generate`, `attach_images` takes up to 6 images anchored from just before to just after each passage (`chunk_range`), at most 2 per passage, preferring captions that share words with the question, `build_context_and_sources` shows them as `[IMGn: caption]` where they sit, and `resolve_markers` keeps only supplied markers, once each (`placed`); when the context has figures, `prompt_builder` also adds `prompts/core/figure_reminder.md` to the task prompt, before its closing cue;
  - `GET /api/v1/document-images/{image_id}?course_id=&document_id=` (API key) serves the PNG only if it is in that course document's manifest and the document passes the retrieval access filter. Moodle must proxy it after checking the user's course access (`docs/frontend-document-images-handoff.md`).
- **`embedding_provider.py`** produces embeddings through OpenAI, Gemini or Ollama. The provider/model comes from Moodle when it sets one, otherwise from `.env` (`EMBEDDING_PROVIDER`/`EMBEDDING_MODEL`). The key is the chat key when `embedding_reuse_chat_api_key` is on; otherwise the Moodle embedding key, then the chat key. `embed_texts(..., settings=)` accepts unsaved settings for the configuration test.
- **`vector_store.py`:**
  - opens the Chroma collection for the active embedding model, named `<prefix>_<provider>_<model>_<hash>`, and reopens it in every worker when the embedding identity changes;
  - indexes and replaces documents, storing `heading` and `chunk_index` and embedding each chunk as `Document: <title>\nSection: <heading>\n\n<text>` (the stored text stays plain);
  - searches with course scoping: `search_course` (dense only) and `hybrid_search` (dense + keyword);
  - `expand_with_neighbours` and `ordered_document_chunks`, both through the same access filter;
  - deletes documents and toggles retrieval;
  - migrates legacy metadata;
  - reports retrieval status.
- **`retrieval.py`:**
  - the lazily created retrieval thread pool;
  - `enabled_only`;
  - `filter_by_relevance` (logs `chat.retrieval.filtered`);
  - `build_context_and_sources`;
  - `document_scope` / `document_passages` (whole-document reading for summaries) and `chunk_count`;
  - `NO_CONTEXT_ANSWER`.
- **`suggestion_generator.py`** produces three document-grounded starter questions via `generate_structured_sync`, with a deterministic fallback.
- **`illustrations.py`** generates `/explain` images with the Moodle-selected image provider. Images are stored under `storage/illustrations/` and served via signed URLs.

## Request flows

### Indexing a document

```text
Moodle upload (POST /api/v1/documents/index)
    -> documents.py            validate type/size, visibility=course
    -> document_parser.py      text per page/slide, with headings
    -> chunker.py              sentence-aligned overlapping chunks (heading, chunk_index)
    -> vector_store.py         embedding_provider.embed_texts() → Chroma collection for the
                               active embedding model (replaces any previous chunks of the document)
    -> suggestion_generator.py (optional) three starter questions
```

### Answering a question

```text
Moodle chat (POST /api/v1/chat, provider + model from Moodle)
    -> chat.py                 prompt_router → RagContext(course, user, include flag, provider, model)
    -> rag_graph               classify → [conversational_reply]
                               or rewrite_query (+ sub-queries) → retrieve (vector_store.hybrid_search:
                               dense + BM25, RRF) → grade (relevance threshold, ≤ 3 attempts)
                               → refine (rerank, top_k, neighbours / whole document)
                               → generate (prompt_builder + llm_provider) | no_context
    -> chat.py                 optional /explain illustration
    -> ChatResponse            answer, sources, retrieved_chunks, retrieved_contexts
```

### Runtime configuration (provider/model/keys)

```text
Moodle admin settings
    -> POST /api/v1/admin/config          chat/image/embedding provider, model, keys (X-RAG-API-Key)
    -> admin.py                           Fernet-encrypt keys with RUNTIME_CONFIG_ENCRYPTION_KEY
    -> storage/runtime_config.json        atomic replace
    -> config.get_settings()              every worker reloads when the file's mtime changes
    -> llm_provider._chat_model()         next request builds a new model with the new settings
```

Changing the **chat** model needs nothing else. Changing the **embedding** provider or model (in Moodle, or in `.env` and then restart) switches to a different Chroma collection, so every document must be re-indexed. For a Moodle change, `POST /admin/config` returns `reindex_required: true` and Moodle queues the reindex. Until it finishes, retrieval finds nothing in the new collection. The previous collection stays on disk.

## Important limitations

- Scanned PDFs need OCR, which isn't implemented. Tables and complex layouts may not extract perfectly.
- Chunking uses character windows rather than semantic boundaries.
- Indexing runs synchronously inside the request.
- The relevance threshold is calibrated only for `openai/text-embedding-3-small`. Other embedding models are not filtered until they're calibrated (see DECISION-005).
- Answers are not yet checked for grounding. The `validate` node is a placeholder for the deferred adaptive-retrieval-eval track.
- The shared API key does not replace Moodle authorisation.
- Past papers are not separated from other course documents.
