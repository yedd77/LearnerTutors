"""Chat endpoints: the prompt-tool menu and the course-grounded chat request."""
import asyncio
import logging
import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.config import get_settings
from app.schemas import ChatRequest, ChatResponse, DownloadItem, PendingAction
from app.security import RequireApiKey
from app.services import llm_provider, pending_actions, retrieval, temp_files
from app.services import study_sheet as sheets
from app.services.llm_provider import ProviderError
from app.services.illustrations import generate_illustration_async
from app.prompt_registry import get_tool, get_tool_menu
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
                               request_id: str,
                               fallback: tuple[str, str] | None = None) -> bool:
    """Ask the chat model whether an illustration would help. Fails closed.

    Uses the Moodle-selected chat provider/model. Any error or timeout means
    no image, so a broken check never spends on image generation.
    """
    try:
        decision = await llm_provider.generate_structured(
            _aux_messages("illustrate.md", question=question, answer=answer[:4000]),
            IllustrationDecision, provider, model,
            timeout=get_settings().rag_aux_timeout_seconds, fallback=fallback,
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
                      request_id: str, provider: str, model: str,
                      fallback: tuple[str, str] | None = None) -> str | None:
    """Return a signed image URL for an explicit ``/explain`` request, else None.

    The chat model first decides whether an image would help; if it says no or
    the check fails, no image is generated. Illustration is best effort: any
    failure is logged and the text answer is still returned. Cancellation propagates.
    """
    if route.tool.id != "concept_explanation" or not route.explicit_command:
        return None
    if not await _illustration_needed(question, answer, provider, model, request_id, fallback):
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


def _resolve_pending(request: ChatRequest, request_id: str) -> RouteResult | str:
    """Answer to a widen-search question: the route to re-run, or the reply to send.

    Approval re-runs the question stored in the signed token, never the request
    text, so only the original /notes request can be repeated, and only for its
    own course and user. Decline, or an invalid/expired token, returns a reply.
    """
    try:
        pending = pending_actions.verify(request.pending_action_token or "", sheets.EXPAND_ACTION,
                                         course_id=request.course_id, user_id=request.user_id)
    except pending_actions.PendingActionError as exc:
        logger.info("chat.pending.invalid id=%s reason=%s", request_id, exc)
        return sheets.expired_request_message()
    if not sheets.is_notes(pending.tool_id):
        logger.warning("chat.pending.invalid id=%s reason=unexpected_tool", request_id)
        return sheets.expired_request_message()
    if request.pending_decision != "approve":
        logger.info("chat.pending.declined id=%s", request_id)
        return sheets.declined_message()
    logger.info("chat.pending.approved id=%s", request_id)
    return RouteResult(tool=get_tool(pending.tool_id), question=pending.question,
                       explicit_command=True)


async def _study_sheet_response(result: dict, request: ChatRequest, request_id: str,
                                request_started: float) -> ChatResponse:
    """Render the study sheet PDF, store it temporarily and describe the download.

    Rendering is CPU-bound, so it runs in a worker thread. A rendering or
    storage failure returns a polite message instead of an error.
    """
    from app.pdf.study_sheet import render_study_sheet

    settings = get_settings()
    sheet, sources = result["study_sheet"], result["sources"]
    matches = result.get("matches", [])
    ttl_minutes = max(1, settings.study_sheet_ttl_minutes)
    started = time.perf_counter()
    try:
        rendered = await asyncio.to_thread(render_study_sheet, sheet, sources)
        stored = await asyncio.to_thread(
            temp_files.store, rendered.pdf, course_id=request.course_id,
            user_id=request.user_id, filename=f"{sheet.title} study sheet",
            ttl_seconds=ttl_minutes * 60,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("chat.study_sheet.render_failed id=%s", request_id)
        return ChatResponse(answer=sheets.render_failed_message(), sources=[])
    logger.info("chat.study_sheet.ready id=%s pages=%s scale=%.2f render_ms=%.2f total_ms=%.2f",
                request_id, rendered.pages, rendered.scale,
                (time.perf_counter() - started) * 1000,
                (time.perf_counter() - request_started) * 1000)
    return ChatResponse(
        answer=sheets.ready_message(ttl_minutes, rendered.pages),
        sources=sources,
        retrieved_chunks=retrieval.chunk_count(matches),
        retrieved_contexts=[match["text"] for match in matches],
        download=DownloadItem(file_id=stored.file_id, filename=stored.filename,
                              content_type=stored.content_type, expires_at=stored.expires_at,
                              ttl_minutes=ttl_minutes, pages=rendered.pages),
    )


@router.post("", response_model=ChatResponse)
async def ask(request: ChatRequest, http_request: Request):
    """Answer a student's message from the course documents via the RAG graph.

    Steps: route to a prompt tool (422 if a slash command has no text), run the
    graph with the request's access scope and Moodle-selected model, and add an
    illustration for ``/explain`` when the model judges one helpful. Conversational and no-context replies return
    no sources. ``/notes`` returns a study sheet PDF as ``download``, or a
    ``pending_action`` asking to widen a thin search; the reply to that action
    arrives as ``pending_action_token`` + ``pending_decision``.
    Errors: 502 provider failure, 422 invalid input, 500 otherwise.
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
        expanded = False
        if request.pending_decision is not None:
            # The student answered an earlier "widen the /notes search?" question.
            route = _resolve_pending(request, request_id)
            if isinstance(route, str):
                return ChatResponse(answer=route, sources=[])
            expanded = True
        else:
            route_started = time.perf_counter()
            route = route_message(
                request.question,
                requested_tool_id=request.tool_id,
            )
            logger.info(
                "chat.route id=%s tool=%s explicit=%s duration_ms=%.2f",
                request_id,
                route.tool.id,
                route.explicit_command,
                (time.perf_counter() - route_started) * 1000,
            )
        question = route.question.strip()
        if not question:
            raise HTTPException(
                status_code=422,
                detail="Enter a request after selecting the command, e.g. /exam Create 10 questions.",
            )

        history = [item.model_dump() for item in request.history]
        # Access scope comes from the request only and is passed as runtime
        # context; the graph can change the search text, never the scope.
        rag_context = RagContext(
            course_id=request.course_id,
            user_id=request.user_id,
            include_course_documents=request.include_course_documents,
            request_id=request_id,
            provider=request.provider or settings.llm_provider,
            model=request.model or settings.llm_model,
            fallback_provider=request.fallback_provider,
            fallback_model=request.fallback_model,
        )
        result = await run_rag_graph(route=route, history=history, context=rag_context,
                                     expanded=expanded)
        answer = result["answer"]
        matches = result.get("matches", [])
        context = result.get("context", "")
        if result.get("outcome") == "study_sheet":
            return await _study_sheet_response(result, request, request_id, request_started)
        if result.get("outcome") == "needs_expansion":
            token, expires_at = pending_actions.issue(
                sheets.EXPAND_ACTION, course_id=request.course_id, user_id=request.user_id,
                question=question, tool_id=route.tool.id,
            )
            logger.info("chat.finish id=%s outcome=needs_expansion total_ms=%.2f", request_id,
                        (time.perf_counter() - request_started) * 1000)
            return ChatResponse(
                answer=answer, sources=[], retrieved_chunks=0, retrieved_contexts=[],
                pending_action=PendingAction(
                    type=sheets.EXPAND_ACTION, token=token, expires_at=expires_at,
                    approve_label="Yes, search more widely", decline_label="No, cancel",
                ),
            )
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
            fallback=rag_context.fallback,
        )

        logger.info(
            "chat.finish id=%s total_ms=%.2f",
            request_id,
            (time.perf_counter() - request_started) * 1000,
        )
        return ChatResponse(
            answer=answer,
            sources=result["sources"],
            retrieved_chunks=retrieval.chunk_count(matches),
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
