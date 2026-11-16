import operator
from typing import Annotated, Any, Literal, TypedDict

from app.prompt_router import RouteResult
from app.schemas import SourceItem


class RagState(TypedDict, total=False):
    """Mutable graph state. Deliberately has no course, user or visibility fields:
    the access scope lives only in RagContext."""

    question: str                 # original question, after command stripping
    history: list[dict[str, str]]
    route: RouteResult
    intent: Literal["conversational", "course_question"]
    classified_by: str            # router | heuristic | llm | fallback
    search_query: str             # text used for retrieval only, never for generation
    attempts: int                 # retrieval attempts made so far
    retrieved_raw: int            # chunks returned by the last search, before grading
    matches: list[dict[str, Any]]
    grade: Literal["relevant", "poor"]
    next_step: str
    context: str
    sources: list[SourceItem]
    answer: str
    validation: dict[str, Any] | None   # filled by the grounding validator (later track)
    outcome: Literal["answered", "no_context", "conversational"]
    trace: Annotated[list[str], operator.add]
    llm_calls: Annotated[int, operator.add]
