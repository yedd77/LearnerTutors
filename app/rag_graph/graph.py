"""Chat workflow as a LangGraph StateGraph.

START → classify ─conversational─► conversational_reply ─────────────────────────────► finalize → END
            │ course_question
            ▼
        rewrite_query → retrieve → grade ─relevant─► refine → generate → validate ─────► finalize
            ▲    │ same query as before   │ poor, retry allowed
            │    └──────────────────────────────────────► no_context ──────────► finalize
            └────────────────────────┘ poor, exhausted ─► no_context

retrieve: hybrid dense + keyword search for the query and any sub-queries.
refine: LLM rerank (when more candidates than top_k), cut to top_k, then
neighbour expansion or, for /summarize and /studyguide, the whole document in order.

/notes (note_making): grade ─relevant─► refine ─► study_sheet ─► finalize. study_sheet either
produces StudySheet content (the endpoint renders the PDF) or, when the material is
too thin, asks to widen the search (needs_expansion) or reports it (insufficient).

/quiz (practice_quiz): grade ─relevant─► refine ─► practice_quiz ─► finalize. practice_quiz
produces a validated PracticeQuiz (one retry on an invalid quiz, else quiz_failed) or, when
the material is too thin, offers a broader topic (needs_expansion) or reports it (insufficient).

/flashcards (flashcards): grade ─relevant─► refine ─► flashcards ─► finalize. flashcards
produces a validated FlashcardDeck of 3 to the requested number of cards (one retry on an
invalid deck, else flashcards_failed); too little material is reported (insufficient), with
no pending action.
"""
import functools
import logging
import time

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from app.prompt_router import RouteResult
from app.rag_graph import nodes
from app.rag_graph.context import RagContext
from app.rag_graph.state import RagState

logger = logging.getLogger("app.rag_graph")
RECURSION_LIMIT = 25


def _timed(name: str, node):
    """Wrap a node to log its duration and append its name to ``state["trace"]``."""
    @functools.wraps(node)
    async def wrapper(state: RagState, runtime: Runtime[RagContext]) -> dict:
        """Run the wrapped node, then record timing and trace."""
        started = time.perf_counter()
        update = dict(await node(state, runtime))
        logger.info("rag_graph.node.finish id=%s node=%s duration_ms=%.2f",
                    runtime.context.request_id, name, (time.perf_counter() - started) * 1000)
        update["trace"] = [name]
        return update
    return wrapper


def build_graph():
    """Wire the nodes and conditional edges shown in the module docstring and compile the graph."""
    graph = StateGraph(RagState, context_schema=RagContext)
    for name in ("classify", "conversational_reply", "rewrite_query", "retrieve", "grade",
                 "refine", "generate", "validate", "study_sheet", "practice_quiz", "flashcards",
                 "no_context", "finalize"):
        graph.add_node(name, _timed(name, getattr(nodes, name)))
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", nodes.after_classify,
                                ["conversational_reply", "rewrite_query"])
    graph.add_edge("conversational_reply", "finalize")
    graph.add_conditional_edges("rewrite_query", nodes.after_rewrite, ["retrieve", "no_context"])
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", nodes.after_grade,
                                ["refine", "rewrite_query", "no_context"])
    graph.add_conditional_edges("refine", nodes.after_refine,
                                ["generate", "study_sheet", "practice_quiz", "flashcards"])
    graph.add_edge("generate", "validate")
    graph.add_edge("validate", "finalize")
    graph.add_edge("no_context", "finalize")
    graph.add_edge("study_sheet", "finalize")
    graph.add_edge("practice_quiz", "finalize")
    graph.add_edge("flashcards", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


RAG_GRAPH = build_graph()


async def run_rag_graph(route: RouteResult, history: list[dict], context: RagContext,
                        expanded: bool = False) -> RagState:
    """Run the workflow for one chat request and return the final state.

    ``expanded`` is set only after the student approved a wider /notes search or a
    broader /quiz topic.
    """
    started = time.perf_counter()
    result = await RAG_GRAPH.ainvoke(
        {"question": route.question, "history": history, "route": route,
         "attempts": 0, "trace": [], "llm_calls": 0, "expanded": expanded},
        context=context,
        config={"recursion_limit": RECURSION_LIMIT},
    )
    logger.info(
        "rag_graph.finish id=%s outcome=%s path=%s attempts=%s llm_calls=%s classified_by=%s duration_ms=%.2f",
        context.request_id, result.get("outcome"), ">".join(result.get("trace", [])),
        result.get("attempts", 0), result.get("llm_calls", 0), result.get("classified_by"),
        (time.perf_counter() - started) * 1000,
    )
    return result
