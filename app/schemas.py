from typing import Literal
from pydantic import BaseModel, Field, field_validator

# Pydantic models for request and response payloads in the RAG Assistant API.
class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)

# Model for the chat request.
class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    course_id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    include_course_documents: bool = True
    history: list[ChatHistoryMessage] = Field(default_factory=list, max_length=12)
    # Optional explicit tool selected by Moodle's slash-command UI.
    # When absent, the backend routes plain-language requests automatically.
    tool_id: str | None = Field(default=None, max_length=80)

    # Validate question arent blank
    @field_validator("question")
    @classmethod
    def nonblank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be blank.")
        return value

# Model for a source item in the response.
class SourceItem(BaseModel):
    document_id: str
    filename: str
    page: int | None = None
    chunk_id: str
    excerpt: str
    markers: list[str]

# Model for the chat response.
class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    retrieved_chunks: int

# Model for the index response.
class IndexResponse(BaseModel):
    document_id: str
    filename: str
    course_id: int
    indexed_chunks: int
    status: str

# Model for the delete response.
class DeleteResponse(BaseModel):
    document_id: str
    deleted_chunks: int
    status: str