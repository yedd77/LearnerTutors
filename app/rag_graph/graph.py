"""Chat workflow as a LangGraph StateGraph.

START → classify ─conversational─► conversational_reply ─────────────────────► finalize → END
            │ course_question
            ▼
        rewrite_query → retrieve → grade ─relevant─► generate → validate ─────► finalize
            ▲    │ same query as before   │ poor, retry allowed
            │    └──────────────────────────────────────► no_context ──────────► finalize
            └────────────────────────┘ poor, exhausted ─► no_context
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
    @functools.wraps(node)
    async def wrapper(state: RagState, runtime: Runtime[RagContext]) -> dict:
        started = time.perf_counter()
        update = dict(await node(state, runtime))
        logger.info("rag_graph.node.finish id=%s node=%s duration_ms=%.2f",
                    runtime.context.request_id, name, (time.perf_counter() - started) * 1000)
        update["trace"] = [name]
        return update
    return wrapper


def build_graph():
    graph = StateGraph(RagState, context_schema=RagContext)
    for name in ("classify", "conversational_reply", "rewrite_query", "retrieve", "grade",
                 "generate", "validate", "no_context", "finalize"):
        graph.add_node(name, _timed(name, getattr(nodes, name)))
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", nodes.after_classify,
                                ["conversational_reply", "rewrite_query"])
    graph.add_edge("conversational_reply", "finalize")
    graph.add_conditional_edges("rewrite_query", nodes.after_rewrite, ["retrieve", "no_context"])
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", nodes.after_grade,
                                ["generate", "rewrite_query", "no_context"])
    graph.add_edge("generate", "validate")
    graph.add_edge("validate", "finalize")
    graph.add_edge("no_context", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


RAG_GRAPH = build_graph()


async def run_rag_graph(route: RouteResult, history: list[dict], context: RagContext) -> RagState:
    """Run the workflow for one chat request and return the final state."""
    started = time.perf_counter()
    result = await RAG_GRAPH.ainvoke(
        {"question": route.question, "history": history, "route": route,
         "attempts": 0, "trace": [], "llm_calls": 0},
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
