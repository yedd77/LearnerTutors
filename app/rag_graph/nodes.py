"""Graph nodes. Every node reads the access scope from ``runtime.context`` only."""
import asyncio
import logging
import re
import time
from typing import Literal

from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.prompt_builder import _fill, _read_prompt
from app.prompt_registry import PROMPT_DIR
from app.rag_graph.context import RagContext
from app.rag_graph.state import RagState
from app.services import llm_provider, retrieval, vector_store
from app.services import study_sheet as sheets

logger = logging.getLogger("app.rag_graph")
chat_logger = logging.getLogger("app.api.chat")  # keep the existing chat.* log events
GRAPH_PROMPTS = PROMPT_DIR / "graph"
MAX_ATTEMPTS_CAP = 3

CONVERSATIONAL_FALLBACK = (
    "Hi! I'm your course assistant. Ask me anything about your course materials, "
    "and I can also summarise, explain concepts, or make notes, flashcards and practice quizzes."
)


class Intent(BaseModel):
    """Classifier output schema."""
    intent: Literal["conversational", "course_question"]


class SearchQuery(BaseModel):
    """Query-rewrite output schema."""
    query: str = Field(min_length=1, max_length=500)


def max_attempts(settings: Settings) -> int:
    """RAG_MAX_RETRIEVAL_ATTEMPTS clamped to 1..MAX_ATTEMPTS_CAP."""
    return min(MAX_ATTEMPTS_CAP, max(1, settings.rag_max_retrieval_attempts))


def _aux_messages(prompt_file: str, **values: str) -> list[dict[str, str]]:
    """Build the messages for a classify/rewrite call from a prompt file in prompts/graph."""
    prompt = _fill(_read_prompt(str(GRAPH_PROMPTS / prompt_file)), **values)
    return [
        {"role": "system", "content": "You are a precise routing component. Return only valid JSON."},
        {"role": "user", "content": prompt},
    ]


# --------------------------------------------------------------------------- classify
_SOCIAL = (
    r"hi+|hello+|hey+|hiya|yo|greetings|good (?:morning|afternoon|evening|night|day)"
    r"|thanks|thank you|thank u|thx|ty|tq|cheers|much appreciated|appreciate it"
    r"|bye|goodbye|good bye|see you|see ya|later|ok|okay|k|cool|nice|great|awesome|perfect"
    r"|got it|understood|i see|alright|sure|noted"
    r"|who are you|what are you|what can you do|what do you do|how are you|how are you doing"
    r"|hows it going|whats up|sup|help|help me|can you help me|can you help"
)
_FILLER = r"(?: (?:there|so much|a lot|very much|again|bot|assistant|everyone|all|sir|miss|maam|mate|buddy))*"
_SOCIAL_ONLY = re.compile(rf"(?:(?:{_SOCIAL}){_FILLER})(?: (?:{_SOCIAL}){_FILLER})*")
_QUESTION_START = {
    "what", "why", "how", "when", "where", "which", "who", "whose", "whom", "is", "are", "was",
    "were", "does", "do", "did", "can", "could", "should", "would", "will", "explain", "describe",
    "define", "list", "compare", "contrast", "summarise", "summarize", "give", "tell", "show",
    "calculate", "solve", "find", "derive", "prove", "outline", "discuss", "identify", "state",
    "whats", "whys", "hows", "wheres", "whens", "whos", "isnt", "arent", "doesnt", "cant",
}


def heuristic_intent(message: str) -> str | None:
    """Cheap classification. None means ambiguous: ask the LLM."""
    normalised = " ".join(re.sub(r"[^\w\s]", " ", message.lower().replace("'", "")).split())
    if not normalised or _SOCIAL_ONLY.fullmatch(normalised):
        return "conversational"
    words = normalised.split()
    if message.rstrip().endswith("?") or words[0] in _QUESTION_START or len(words) >= 6:
        return "course_question"
    return None


async def classify(state: RagState, runtime: Runtime[RagContext]) -> dict:
    """Decide between a conversational reply and the course-question path.

    Explicit tools and slash commands always take the course path, then the
    regex heuristic decides clear cases, and only ambiguous messages use the
    LLM classifier. Any classifier failure falls back to the course path.
    """
    route = state["route"]
    if route.explicit_command or route.tool.id != "ask":
        return {"intent": "course_question", "classified_by": "router"}
    intent = heuristic_intent(state["question"])
    if intent is not None:
        return {"intent": intent, "classified_by": "heuristic"}

    ctx = runtime.context
    try:
        result = await llm_provider.generate_structured(
            _aux_messages("classify.md", history=llm_provider._history_text(state.get("history", [])),
                          question=state["question"]),
            Intent, ctx.provider, ctx.model, timeout=get_settings().rag_aux_timeout_seconds,
            fallback=ctx.fallback,
        )
        return {"intent": result.intent, "classified_by": "llm", "llm_calls": 1}
    except Exception as exc:  # fail open to the safe default; CancelledError is not an Exception
        logger.warning("rag_graph.classify.fallback id=%s error=%s", ctx.request_id, exc)
        return {"intent": "course_question", "classified_by": "fallback", "llm_calls": 1}


def after_classify(state: RagState) -> str:
    """Edge: conversational_reply or rewrite_query."""
    return "conversational_reply" if state["intent"] == "conversational" else "rewrite_query"


# --------------------------------------------------------------------------- conversational
async def conversational_reply(state: RagState, runtime: Runtime[RagContext]) -> dict:
    """Short friendly reply with no retrieval; a static reply is used if the LLM fails."""
    ctx = runtime.context
    messages = [{"role": "system", "content": _read_prompt(str(GRAPH_PROMPTS / "conversational.md"))}]
    messages += [
        {"role": item["role"], "content": str(item["content"])[:2000]}
        for item in state.get("history", [])[-6:]
        if item.get("role") in ("user", "assistant")
    ]
    messages.append({"role": "user", "content": state["question"]})
    try:
        answer = await llm_provider.generate_messages_async(messages, ctx.provider, ctx.model,
                                                            fallback=ctx.fallback)
    except Exception as exc:
        logger.warning("rag_graph.conversational.fallback id=%s error=%s", ctx.request_id, exc)
        answer = CONVERSATIONAL_FALLBACK
    return {"answer": answer, "matches": [], "sources": [], "context": "",
            "outcome": "conversational", "llm_calls": 1}


# --------------------------------------------------------------------------- course path
async def rewrite_query(state: RagState, runtime: Runtime[RagContext]) -> dict:
    """Choose the search text for this attempt.

    The first turn uses the question as-is; follow-ups are condensed with the
    history and retries are reworded by the LLM. Exact-text extraction is never
    rewritten. The result is used only for search, never for generation.
    """
    question = state["question"]
    attempts = state.get("attempts", 0)
    history = state.get("history", [])
    # Exact extraction must search the user's own wording; first turns need no rewrite.
    if state["route"].tool.id == "exact_text_extraction" or (attempts == 0 and not history):
        return {"search_query": question, "next_step": "retrieve"}

    ctx = runtime.context
    retry_note = ""
    if attempts:
        retry_note = (
            "- An earlier search found nothing relevant. Use different wording: broader key "
            f"terms or likely textbook vocabulary. Earlier search: {state.get('search_query', '')[:300]}"
        )
    try:
        result = await llm_provider.generate_structured(
            _aux_messages("rewrite.md", retry_note=retry_note,
                          history=llm_provider._history_text(history), question=question),
            SearchQuery, ctx.provider, ctx.model, timeout=get_settings().rag_aux_timeout_seconds,
            fallback=ctx.fallback,
        )
        query = " ".join(result.query.split())[:500] or question
    except Exception as exc:
        logger.warning("rag_graph.rewrite.fallback id=%s error=%s", ctx.request_id, exc)
        query = question
    # A retry with the same text would return the same chunks: stop instead.
    repeated = attempts > 0 and query.casefold() == state.get("search_query", "").casefold()
    return {"search_query": query, "next_step": "no_context" if repeated else "retrieve",
            "llm_calls": 1}


def after_rewrite(state: RagState) -> str:
    """Edge: retrieve, or no_context when a retry would repeat the same query."""
    return state["next_step"]


async def retrieve(state: RagState, runtime: Runtime[RagContext]) -> dict:
    """Search the course with the current query; the scope comes only from runtime.context."""
    ctx = runtime.context
    settings = get_settings()
    top_k = state["route"].tool.top_k or settings.top_k
    if state.get("expanded"):
        # Approved wider /notes search: more chunks, still only this course.
        top_k = sheets.expanded_top_k(settings)
    started = time.perf_counter()
    chat_logger.info("chat.retrieval.start id=%s course=%s top_k=%s attempt=%s expanded=%s",
                     ctx.request_id, ctx.course_id, top_k, state.get("attempts", 0) + 1,
                     bool(state.get("expanded")))
    loop = asyncio.get_running_loop()
    matches = await loop.run_in_executor(
        retrieval.get_retrieval_executor(),
        vector_store.search_course,
        state["search_query"],          # the only model-influenced argument
        ctx.course_id,
        ctx.include_course_documents,
        top_k,
    )
    chat_logger.info("chat.retrieval.finish id=%s chunks=%s duration_ms=%.2f",
                     ctx.request_id, len(matches), (time.perf_counter() - started) * 1000)
    matches = retrieval.enabled_only(matches)
    return {"matches": matches, "retrieved_raw": len(matches),
            "attempts": state.get("attempts", 0) + 1}


def grade_matches(matches: list[dict], settings: Settings, request_id: str) -> list[dict]:
    """Deterministic relevance grading. Extension point for the validator track."""
    return retrieval.filter_by_relevance(matches, settings, request_id)


async def grade(state: RagState, runtime: Runtime[RagContext]) -> dict:
    """Keep relevant chunks and choose the next step: generate, retry, or no_context.

    A retry only happens when chunks existed but none were close enough, the
    attempt budget allows it, and the tool is not exact-text extraction.
    """
    ctx = runtime.context
    settings = get_settings()
    matches = state.get("matches", [])
    try:
        relevant = grade_matches(matches, settings, ctx.request_id)
    except Exception as exc:
        logger.warning("rag_graph.grade.fallback id=%s error=%s", ctx.request_id, exc)
        relevant = matches
    if relevant:
        next_step = "study_sheet" if sheets.is_notes(state["route"].tool.id) else "generate"
    elif (
        state.get("retrieved_raw", 0) > 0                      # something exists, just not close enough
        and state["route"].tool.id != "exact_text_extraction"
        and state.get("attempts", 0) < max_attempts(settings)
    ):
        next_step = "rewrite_query"
    else:
        next_step = "no_context"
    return {"matches": relevant, "grade": "relevant" if relevant else "poor", "next_step": next_step}


def after_grade(state: RagState) -> str:
    """Edge: the next_step chosen by grade."""
    return state["next_step"]


async def study_sheet(state: RagState, runtime: Runtime[RagContext]) -> dict:
    """/notes: check the material is sufficient, then generate the study sheet content.

    Too little material on a normal search asks the student before widening it
    (needs_expansion); too little after a widened search is reported honestly
    (insufficient). Either way no sheet is made. The PDF itself is rendered by
    the chat endpoint, not here.
    """
    ctx = runtime.context
    settings = get_settings()
    matches = state["matches"]
    question = state["route"].question
    expanded = bool(state.get("expanded"))
    limited = {"matches": [], "sources": [], "context": "", "study_sheet": None,
               "answer": (sheets.still_limited_message(question) if expanded
                          else sheets.ask_to_expand_message(question)),
               "outcome": "insufficient" if expanded else "needs_expansion"}
    if not sheets.enough_material(matches, settings):
        chat_logger.info("chat.study_sheet.limited id=%s chunks=%s expanded=%s reason=too_little",
                         ctx.request_id, len(matches), expanded)
        return limited

    context, sources = retrieval.build_context_and_sources(matches)
    started = time.perf_counter()
    chat_logger.info("chat.llm.start id=%s provider=%s model=%s task=study_sheet",
                     ctx.request_id, ctx.provider, ctx.model)
    # Uses the original question (route.question), never the rewritten search query.
    sheet = await sheets.generate_study_sheet(question, context, state.get("history", []),
                                              ctx.provider, ctx.model, ctx.fallback)
    chat_logger.info("chat.llm.finish id=%s task=study_sheet coverage=%s duration_ms=%.2f",
                     ctx.request_id, sheet.coverage, (time.perf_counter() - started) * 1000)
    if sheet.coverage == "limited" or not sheet.has_content():
        chat_logger.info("chat.study_sheet.limited id=%s chunks=%s expanded=%s reason=model",
                         ctx.request_id, len(matches), expanded)
        return {**limited, "llm_calls": 1}
    return {"study_sheet": sheet, "context": context, "sources": sources, "answer": "",
            "outcome": "study_sheet", "llm_calls": 1}


async def generate(state: RagState, runtime: Runtime[RagContext]) -> dict:
    """Build the [S1..Sn] context and sources, then generate the grounded answer."""
    ctx = runtime.context
    started = time.perf_counter()
    context, sources = retrieval.build_context_and_sources(state["matches"])
    history = state.get("history", [])
    chat_logger.info(
        "chat.context.ready id=%s chunks=%s sources=%s context_chars=%s history_messages=%s duration_ms=%.2f",
        ctx.request_id, len(state["matches"]), len(sources), len(context), len(history),
        (time.perf_counter() - started) * 1000,
    )
    started = time.perf_counter()
    chat_logger.info("chat.llm.start id=%s provider=%s model=%s", ctx.request_id, ctx.provider, ctx.model)
    # Generation always uses the original question (route.question), never the search query.
    answer = await llm_provider.generate_answer_async(
        route=state["route"], context=context, history=history,
        provider=ctx.provider, model=ctx.model, fallback=ctx.fallback,
    )
    chat_logger.info("chat.llm.finish id=%s answer_chars=%s duration_ms=%.2f",
                     ctx.request_id, len(answer), (time.perf_counter() - started) * 1000)
    return {"answer": answer, "context": context, "sources": sources,
            "outcome": "answered", "llm_calls": 1}


async def validate(state: RagState, runtime: Runtime[RagContext]) -> dict:
    """Grounding validation hook. Pass-through until the validator track."""
    return {"validation": None}


async def no_context(state: RagState, runtime: Runtime[RagContext]) -> dict:
    """Return the standard no-relevant-text answer; the model is never asked to guess."""
    chat_logger.info("chat.no_context id=%s attempts=%s", runtime.context.request_id, state.get("attempts", 0))
    return {"answer": retrieval.NO_CONTEXT_ANSWER, "matches": [], "sources": [], "context": "",
            "outcome": "no_context"}


async def finalize(state: RagState, runtime: Runtime[RagContext]) -> dict:
    """Terminal node; the response is built from the final state by the chat endpoint."""
    return {}
