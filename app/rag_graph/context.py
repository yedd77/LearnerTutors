from dataclasses import dataclass


@dataclass(frozen=True)
class RagContext:
    """Per-request access scope and model selection.

    Passed as LangGraph runtime context, never as state: nodes and model output
    can change what text is searched for, but never who or which course the
    search is for.
    """
    course_id: int
    user_id: int
    include_course_documents: bool
    request_id: str
    provider: str
    model: str
