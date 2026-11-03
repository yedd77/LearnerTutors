# RAG Assistant Backend Structure

This project is a local FastAPI backend for a Moodle RAG (Retrieval-Augmented
Generation) assistant. It uses:

- **FastAPI** for the HTTP API.
- **Ollama** for text embeddings and chat responses.
- **ChromaDB** for persistent vector storage.
- **Pydantic Settings** for configuration through environment variables.

## Project tree

```text
ragassistant-backend/
|
|-- app/
|   |-- __init__.py
|   |-- config.py
|   |-- main.py
|   |-- prompt_builder.py
|   |-- prompt_registry.py
|   |-- prompt_router.py
|   |-- schemas.py
|   |-- security.py
|   |
|   |-- api/
|   |   |-- __init__.py
|   |   |-- chat.py
|   |   `-- documents.py
|   |
|   |-- prompts/
|   |   |-- core/
|   |   |   |-- answer_format.md
|   |   |   |-- grounding.md
|   |   |   `-- instructions.md
|   |   |
|   |   `-- tools/
|   |       |-- ask.md
|   |       |-- concept_explanation.md
|   |       |-- exact_text_extraction.md
|   |       |-- exam_generation.md
|   |       |-- flashcards.md
|   |       |-- note_making.md
|   |       |-- practice_quiz.md
|   |       |-- study_guide.md
|   |       `-- summarization.md
|   |
|   `-- services/
|       |-- __init__.py
|       |-- chunker.py
|       |-- document_parser.py
|       |-- ollama_client.py
|       `-- vector_store.py
|
|-- storage/
|   `-- chroma/
|       `-- ...                 Persistent ChromaDB files
|
|-- .env.example                Example environment configuration
|-- README.md                   Setup, run, and API usage instructions
|-- requirements.txt            Python dependencies
`-- structure.md                This project structure reference
```

## Application entry point

### `app/main.py`

Creates and configures the FastAPI application.

- Configures application logging.
- Registers the document and chat routers under `/api/v1`.
- Provides `GET /health` for a basic health check.
- Provides `GET /ready` to verify that ChromaDB can be opened and report the
  number of indexed chunks.
- Adds request and response timing logs without logging request headers or
  bodies.

### `app/config.py`

Defines the Pydantic Settings configuration used throughout the application.
It reads values from `.env` and provides defaults for:

- FastAPI host, port, environment, and log level.
- The shared RAG API key.
- Ollama URL, chat model, embedding model, and timeout.
- ChromaDB path and collection name.
- Retrieval, chunking, upload, page, and context limits.

The `chroma_absolute_path` property converts a relative ChromaDB path to an
absolute path and creates the directory when required.

## API layer

### `app/api/chat.py`

Defines the chat endpoints under `/api/v1/chat`.

- `GET /api/v1/chat/tools` returns the available prompt tools for a UI menu.
- `POST /api/v1/chat`:
  1. Routes the request to a prompt tool.
  2. Searches course-scoped document chunks in ChromaDB.
  3. Removes chunks whose `retrieval_enabled` flag is false.
  4. Builds source markers and source metadata.
  5. Generates a grounded answer through Ollama.
  6. Returns the answer, sources, and retrieved chunk count.

The endpoint accepts optional conversation history and an explicit `tool_id`.

### `app/api/documents.py`

Defines document-management endpoints under `/api/v1/documents`.

- `POST /api/v1/documents/index` uploads and indexes PDF, DOCX, PPTX, TXT, or
  Markdown content.
- `DELETE /api/v1/documents/{document_id}` removes a document's vectors.
- `PATCH /api/v1/documents/{document_id}/retrieval` enables or disables
  retrieval for all chunks belonging to a document.

Uploads are validated for file type, size, document metadata, page limits, and
maximum generated chunks. All endpoints require the configured API key.

## Request and security modules

### `app/schemas.py`

Contains the Pydantic request and response models used by the API, including
chat requests, chat responses, source items, indexing responses, and deletion
responses.

### `app/security.py`

Implements the `X-RAG-API-Key` authentication dependency used by the document
and chat routers.

The key authenticates the Moodle server to this backend. It does not itself
verify Moodle user enrollment or permissions; that authorization remains the
responsibility of the Moodle integration.

## Prompt system

### `app/prompt_registry.py`

Defines the supported prompt tools and their metadata:

- Tool ID and display label.
- Description and slash command.
- Prompt filename.
- Retrieval mode and optional `top_k` value.
- Natural-language aliases.

It also provides the tool menu returned by the chat API.

### `app/prompt_router.py`

Selects a tool from:

1. An explicitly supplied `tool_id`.
2. A slash command such as `/exam` or `/summarize`.
3. Rule-based natural-language patterns.
4. The default `ask` tool.

Routing is local and rule-based; it does not make a second Ollama classifier
request.

### `app/prompt_builder.py`

Loads the shared core prompts and the selected tool prompt, then combines them
with the retrieved course context and conversation history into the final
message bundle sent to Ollama.

Prompt files are cached per process, so FastAPI must be restarted after editing
them.

### `app/prompts/core/`

Shared instructions used by every tool:

- `instructions.md` — defines the assistant's role and general behavior.
- `grounding.md` — requires answers to stay grounded in retrieved material.
- `answer_format.md` — defines the expected answer and source-marker format.

### `app/prompts/tools/`

Task-specific instructions selected by the prompt router:

- `ask.md` — ordinary course questions.
- `concept_explanation.md` — step-by-step explanations.
- `exact_text_extraction.md` — exact or verbatim source extraction.
- `exam_generation.md` — original exam-question generation.
- `flashcards.md` — question-and-answer revision cards.
- `note_making.md` — structured revision notes.
- `practice_quiz.md` — practice quiz generation.
- `study_guide.md` — study-guide organization.
- `summarization.md` — summaries of retrieved material.

All tools currently use the same course-scoped retrieval path. Document-type
metadata is not yet available to separate lecture slides from past papers.

## Document and retrieval services

### `app/services/document_parser.py`

Extracts text into page- or slide-aware sections:

- PDF pages through PyMuPDF.
- DOCX paragraphs and tables through `python-docx`.
- PPTX slides and tables through `python-pptx`.
- UTF-8 text and Markdown files.

Scanned-PDF OCR is not included. The parser raises a clear error when no
extractable text is found.

### `app/services/chunker.py`

Normalizes extracted text and divides it into overlapping character windows.
Each chunk retains its original page or slide number for source references.
Chunk size and overlap are controlled by application settings.

### `app/services/ollama_client.py`

Provides the Ollama integration:

- `embed_texts()` sends document text to `/api/embed` and validates the
  returned embeddings.
- `generate_answer()` sends the assembled prompt to `/api/chat` and validates
  the generated answer.
- Handles Ollama URLs, timeouts, optional prompt logging, conversation-history
  formatting, and API/network errors through `OllamaError`.

### `app/services/vector_store.py`

Manages the persistent ChromaDB collection:

- Creates or opens the configured collection.
- Generates embeddings and stores document chunks.
- Searches chunks by course, owner, visibility, and query.
- Replaces document chunks safely during re-indexing.
- Deletes all vectors for a document.
- Enables or disables retrieval per document.
- Migrates older chunks so missing `retrieval_enabled` metadata defaults to
  enabled.
- Reports indexed documents and enabled/disabled chunk counts.

## Persistent storage

### `storage/chroma/`

Contains the local ChromaDB database and vector index files. These files hold
embeddings and metadata for indexed document chunks; they are not the original
uploaded documents.

The storage path and collection name are configurable through `.env`.

## Typical request flow

### Indexing a document

```text
Upload file
    -> documents.py
    -> document_parser.py
    -> chunker.py
    -> ollama_client.embed_texts()
    -> vector_store.py
    -> storage/chroma/
```

### Answering a question

```text
Chat request
    -> chat.py
    -> prompt_router.py
    -> vector_store.py
    -> ollama_client.embed_texts()
    -> prompt_builder.py
    -> ollama_client.generate_answer()
    -> ChatResponse with answer and sources
```

## Important limitations

- Scanned PDFs require OCR, which is not implemented.
- Tables and complex layouts may not extract perfectly.
- Chunking uses character windows rather than semantic boundaries.
- Indexing and answer generation are synchronous.
- Retrieval has no calibrated similarity-confidence threshold.
- The shared API key does not replace Moodle authorization.
- Past-paper retrieval is not separated from other course documents.


MAIN
|
|_____App/config.py
|     Defines application configuration using Pydantic Settings
|
|_____App/main.py
|     Creates and configures the FastAPI application
|
|_____App/security.py
|     Implements API-key authentication


App/schemas.py
Handles Pydantic request and response models

App/prompt_router.py
Routes a user message to a `Tool Definition`

App/prompt_registry.py
Defines the supporting tools

App/prompt_builder.py
Builds the final prompt bundle sent to Ollama

App/api/chat.py
Defines the chatbot API

App/api/documents.py
Defines document upload and manager API

App/prompts/core/answer_format.md
Framework of instruction on how the model should response

App/prompts/core/grounding.md
Framework of instruction on how the model should verifies itself and increase the accuracy of the response

App/prompts/core/instructions
Defines the LLM role

App/prompts/tools/ask.md
Framework to prompt model on how to response to simple question

App/prompts/tools/concept_explanation.md
Framework to prompt model on how to explain specific topic or concept

App/prompts/tools/exact_text_extraction.md
Framework to prompt model on how to give the text on specific page

App/prompts/tools/exam_generations.md
Framework to prompt model on how to generate exam question based on specific topic user ask

App/prompts/tools/flashcards.md
Framework to prompt model on how to create flashcard based on specific topic user ask

App/prompts/tools/note_making.md
Framework to prompt model on how to generate note based on specific topic user ask

App/prompts/tools/practice_quiz.md
Framework to prompt model on how to generate quiz for practice on specific topic user ask

App/prompts/tools/study_guide.md
Framework to prompt model on how to generate study guide on specific topic user ask

App/prompts/tools/summarization.md
Framework to prompt model on how to create a summary on specific topic user ask

App/services/chunker.py
Implement chunker to chunk text into smaller size

App/services/ollama_client.py
Crate document embedding and generate chat answer

App/services/vector_store.py
	