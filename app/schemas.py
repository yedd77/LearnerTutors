from typing import Literal
from pydantic import BaseModel, Field, field_validator

class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    course_id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    include_course_documents: bool = True
    history: list[ChatHistoryMessage] = Field(default_factory=list, max_length=12)
    tool_id: str | None = Field(default=None, max_length=80)
    # Optional until the Moodle frontend starts sending these fields.
    provider: Literal["openai", "gemini", "ollama"] | None = None
    model: str | None = Field(default=None, max_length=160)
    request_id: str | None = Field(default=None, min_length=8, max_length=80)

    @field_validator("question")
    @classmethod
    def nonblank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be blank.")
        return value

    @field_validator("model")
    @classmethod
    def clean_model(cls, value: str | None) -> str | None:
        return (value.strip() or None) if value is not None else None


class SourceItem(BaseModel):
    document_id: str
    filename: str
    page: int | None = None
    chunk_id: str
    excerpt: str
    markers: list[str]


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    retrieved_chunks: int = 0
    retrieved_contexts: list[str] = Field(default_factory=list)
    illustration_url: str | None = None


class IndexResponse(BaseModel):
    document_id: str
    filename: str
    course_id: int
    indexed_chunks: int
    status: str
    suggestions: list[str] = Field(default_factory=list)


class DeleteResponse(BaseModel):
    document_id: str
    deleted_chunks: int
    status: str
