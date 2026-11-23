"""Typed state shared between the graph nodes."""
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
    expanded: bool                # /notes: student approved a wider search (larger top_k, same course)
    intent: Literal["conversational", "course_question"]
    classified_by: str            # router | heuristic | llm | fallback
    search_query: str             # text used for retrieval only, never for generation
    search_queries: list[str]     # search_query plus up to 2 sub-queries for multi-part questions
    attempts: int                 # retrieval attempts made so far
    retrieved_raw: int            # chunks returned by the last search, before grading
    matches: list[dict[str, Any]]
    grade: Literal["relevant", "poor"]
    next_step: str
    context: str
    sources: list[SourceItem]
    answer: str
    validation: dict[str, Any] | None   # filled by the grounding validator (later track)
    study_sheet: Any              # /notes: StudySheet content; the endpoint renders the PDF
    outcome: Literal["answered", "no_context", "conversational",
                     "study_sheet", "needs_expansion", "insufficient"]
    trace: Annotated[list[str], operator.add]
    llm_calls: Annotated[int, operator.add]
