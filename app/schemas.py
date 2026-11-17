"""Pydantic request and response models for the public API."""
from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator

class ChatHistoryMessage(BaseModel):
    """One earlier turn of the conversation, as sent by Moodle."""
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    """Body of ``POST /api/v1/chat``.

    ``course_id``/``user_id`` come from Moodle, which has already checked the
    user's access. ``tool_id`` selects a prompt tool explicitly; otherwise the
    question text is routed (see prompt_router).
    """
    question: str = Field(min_length=2, max_length=4000)
    course_id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    include_course_documents: bool = True
    history: list[ChatHistoryMessage] = Field(default_factory=list, max_length=12)
    tool_id: str | None = Field(default=None, max_length=80)
    # Moodle must explicitly supply the administrator-selected generation model.
    provider: Literal["openai", "gemini", "ollama"]
    model: str = Field(min_length=1, max_length=160)
    # Optional model Moodle wants tried once when the primary one is temporarily
    # unavailable (429, 5xx, timeout, connection error). Send both or neither.
    fallback_provider: Literal["openai", "gemini", "ollama"] | None = None
    fallback_model: str | None = Field(default=None, max_length=160)

    @field_validator("question")
    @classmethod
    def nonblank(cls, value: str) -> str:
        """Trim the question and reject whitespace-only input."""
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be blank.")
        return value

    @field_validator("model")
    @classmethod
    def clean_model(cls, value: str) -> str:
        """Trim the model name and reject whitespace-only input."""
        value = value.strip()
        if not value:
            raise ValueError("A generation model must be selected in Moodle settings.")
        return value

    @model_validator(mode="after")
    def fallback_pair(self) -> "ChatRequest":
        """Require fallback_provider and fallback_model together; blank model means none."""
        if self.fallback_model is not None:
            self.fallback_model = self.fallback_model.strip() or None
        if (self.fallback_provider is None) != (self.fallback_model is None):
            raise ValueError("Send both fallback_provider and fallback_model, or neither.")
        return self


class SourceItem(BaseModel):
    """A cited document page. ``markers`` lists the [S1..Sn] labels used in the answer."""
    document_id: str
    filename: str
    page: int | None = None
    chunk_id: str
    excerpt: str
    markers: list[str]


class ChatResponse(BaseModel):
    """Chat answer with its sources; ``retrieved_contexts`` are the chunk texts used."""
    answer: str
    sources: list[SourceItem]
    retrieved_chunks: int = 0
    retrieved_contexts: list[str] = Field(default_factory=list)
    illustration_url: str | None = None


class IndexResponse(BaseModel):
    """Result of indexing a document, including any generated starter suggestions."""
    document_id: str
    filename: str
    course_id: int
    indexed_chunks: int
    status: str
    suggestions: list[str] = Field(default_factory=list)


class DeleteResponse(BaseModel):
    """Result of deleting a document's vectors."""
    document_id: str
    deleted_chunks: int
    status: str
