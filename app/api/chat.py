"""Chat endpoints: the prompt-tool menu and the course-grounded chat request."""
import asyncio
import logging
import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.config import get_settings
from app.schemas import ChatRequest, ChatResponse
from app.security import RequireApiKey
from app.services import llm_provider
from app.services.llm_provider import ProviderError
from app.services.illustrations import generate_illustration_async
from app.prompt_registry import get_tool_menu
from app.prompt_router import RouteResult, route_message
from app.rag_graph.context import RagContext
from app.rag_graph.graph import run_rag_graph
from app.rag_graph.nodes import _aux_messages

router = APIRouter(prefix="/chat", tags=["chat"], dependencies=[RequireApiKey])
logger = logging.getLogger(__name__)


@router.get("/tools")
def list_tools():
    """Return the prompt tools (slash commands) Moodle can show in its chat menu."""
    return {"tools": get_tool_menu()}


class IllustrationDecision(BaseModel):
    """Output schema of the "would an image help?" check."""
    needs_image: bool
    reason: str = Field(default="", max_length=300)


async def _illustration_needed(question: str, answer: str, provider: str, model: str,
                               request_id: str) -> bool:
    """Ask the chat model whether an illustration would help. Fails closed.

    Uses the Moodle-selected chat provider/model. Any error or timeout means
    no image, so a broken check never spends on image generation.
    """
    try:
        decision = await llm_provider.generate_structured(
            _aux_messages("illustrate.md", question=question, answer=answer[:4000]),
            IllustrationDecision, provider, model,
            timeout=get_settings().rag_aux_timeout_seconds,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("chat.illustration.check_failed id=%s error=%s", request_id, exc)
        return False
    logger.info("chat.illustration.check id=%s needs_image=%s reason=%s",
                request_id, decision.needs_image, decision.reason[:200])
    return decision.needs_image


async def _illustrate(route: RouteResult, question: str, answer: str, context: str,
                      request_id: str, provider: str, model: str) -> str | None:
    """Return a signed image URL for an explicit ``/explain`` request, else None.

    The chat model first decides whether an image would help; if it says no or
    the check fails, no image is generated. Illustration is best effort: any
    failure is logged and the text answer is still returned. Cancellation propagates.
    """
    if route.tool.id != "concept_explanation" or not route.explicit_command:
        return None
    if not await _illustration_needed(question, answer, provider, model, request_id):
        return None
    started = time.perf_counter()
    logger.info("chat.illustration.start id=%s", request_id)
    try:
        url = await generate_illustration_async(question=question, answer=answer, context=context)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("chat.illustration.error id=%s", request_id)
        return None
    logger.info(
        "chat.illustration.finish id=%s duration_ms=%.2f",
        request_id,
        (time.perf_counter() - started) * 1000,
    )
    return url


@router.post("", response_model=ChatResponse)
async def ask(request: ChatRequest, http_request: Request):
    """Answer a student's message from the course documents via the RAG graph.

    Steps: route to a prompt tool (422 if a slash command has no text), run the
    graph with the request's access scope and Moodle-selected model, and add an
    illustration for ``/explain`` when the model judges one helpful. Conversational and no-context replies return
    no sources. Errors: 502 provider failure, 422 invalid input, 500 otherwise.
    """
    settings = get_settings()
    request_id = getattr(http_request.state, "request_id", "unknown")
    request_started = time.perf_counter()

    logger.info(
        "chat.start id=%s course=%s user=%s history_turns=%s provider=%s model=%s",
        request_id,
        request.course_id,
        request.user_id,
        len(request.history),
        request.provider or settings.llm_provider,
        request.model or settings.llm_model,
    )

    try:
        route_started = time.perf_counter()
        route = route_message(
            request.question,
            requested_tool_id=request.tool_id,
        )
        question = route.question.strip()
        logger.info(
            "chat.route id=%s tool=%s explicit=%s duration_ms=%.2f",
            request_id,
            route.tool.id,
            route.explicit_command,
            (time.perf_counter() - route_started) * 1000,
        )
        if not question:
            raise HTTPException(
                status_code=422,
                detail="Enter a request after selecting the command, e.g. /exam Create 10 questions.",
            )

        history = [item.model_dump() for item in request.history]
        # Access scope comes from the request only and is passed as runtime
        # context; the graph can change the search text, never the scope.
        result = await run_rag_graph(
            route=route,
            history=history,
            context=RagContext(
                course_id=request.course_id,
                user_id=request.user_id,
                include_course_documents=request.include_course_documents,
                request_id=request_id,
                provider=request.provider or settings.llm_provider,
                model=request.model or settings.llm_model,
            ),
        )
        answer = result["answer"]
        matches = result.get("matches", [])
        context = result.get("context", "")
        if result.get("outcome") != "answered":
            logger.info(
                "chat.finish id=%s outcome=%s total_ms=%.2f",
                request_id,
                result.get("outcome"),
                (time.perf_counter() - request_started) * 1000,
            )
            return ChatResponse(answer=answer, sources=[], retrieved_chunks=0, retrieved_contexts=[])

        illustration_url = await _illustrate(
            route, question, answer, context, request_id,
            provider=request.provider or settings.llm_provider,
            model=request.model or settings.llm_model,
        )

        logger.info(
            "chat.finish id=%s total_ms=%.2f",
            request_id,
            (time.perf_counter() - request_started) * 1000,
        )
        return ChatResponse(
            answer=answer,
            sources=result["sources"],
            retrieved_chunks=len(matches),
            retrieved_contexts=[match["text"] for match in matches],
            illustration_url=illustration_url,
        )

    except HTTPException:
        raise
    except ProviderError as exc:
        logger.exception("chat.provider_error id=%s", request_id)
        raise HTTPException(502, str(exc)) from exc
    except ValueError as exc:
        logger.warning("chat.validation_error id=%s error=%s", request_id, exc)
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("chat.error id=%s", request_id)
        raise HTTPException(500, f"RAG request failed: {exc}") from exc
