"""Chat endpoints: the prompt-tool menu and the course-grounded chat request."""
import asyncio
import logging
import time
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.config import get_settings
from app.schemas import ChatRequest, ChatResponse, DownloadItem, ImageItem, PendingAction
from app.security import RequireApiKey
from app.services import llm_provider, pending_actions, retrieval, temp_files
from app.services import exam_paper as exams
from app.services import practice_quiz as quizzes
from app.services import study_sheet as sheets
from app.services.llm_provider import ProviderError
from app.services.illustrations import generate_illustration_async
from app.prompt_registry import get_tool, get_tool_menu
from app.prompt_router import RouteResult, disabled_message, route_message
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
    """Answer to a widen-search / broaden-topic question: the route to re-run, or the reply.

    Approval re-runs the request stored in the signed token, never the request
    text, so only the original /notes or /quiz request can be repeated (a /quiz
    approval searches the broader topic signed into the token), and only for
    its own course and user. Each action type is accepted only for its own
    tool. Decline, or an invalid/expired token, returns a reply.
    """
    token = request.pending_action_token or ""
    pending, reason = None, None
    expired_message = sheets.expired_request_message()
    for action in (sheets.EXPAND_ACTION, quizzes.EXPAND_ACTION):
        try:
            pending = pending_actions.verify(token, action, course_id=request.course_id,
                                             user_id=request.user_id)
            break
        except pending_actions.PendingActionExpired as exc:
            # Genuine token of this action, just too old: word the reply for its command.
            reason = exc
            if action == quizzes.EXPAND_ACTION:
                expired_message = quizzes.expired_request_message()
            break
        except pending_actions.PendingActionError as exc:
            reason = exc
    if pending is None:
        logger.info("chat.pending.invalid id=%s reason=%s", request_id, reason)
        return expired_message

    quiz_action = pending.action == quizzes.EXPAND_ACTION
    allowed = quizzes.is_quiz if quiz_action else sheets.is_notes
    if not allowed(pending.tool_id):
        logger.warning("chat.pending.invalid id=%s reason=unexpected_tool", request_id)
        return quizzes.expired_request_message() if quiz_action else sheets.expired_request_message()
    if request.pending_decision != "approve":
        logger.info("chat.pending.declined id=%s action=%s", request_id, pending.action)
        return quizzes.declined_message(pending.topic) if quiz_action else sheets.declined_message()
    logger.info("chat.pending.approved id=%s action=%s", request_id, pending.action)
    question = (pending.topic or pending.question) if quiz_action else pending.question
    return RouteResult(tool=get_tool(pending.tool_id), question=question, explicit_command=True)


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


async def _exam_paper_response(result: dict, request: ChatRequest, request_id: str,
                               request_started: float) -> ChatResponse:
    """Render the /exam practice paper PDF, store it temporarily and describe the download.

    Same pattern as the study sheet: rendering runs in a worker thread, the
    file is stored for this course and user only, and a rendering or storage
    failure returns a polite message instead of an error. Only counts are logged.
    """
    from app.pdf.exam_paper import render_exam_paper

    settings = get_settings()
    paper, sources = result["exam_paper"], result["sources"]
    matches = result.get("matches", [])
    ttl_minutes = max(1, settings.exam_paper_ttl_minutes)
    started = time.perf_counter()
    try:
        rendered = await asyncio.to_thread(render_exam_paper, paper, sources)
        stored = await asyncio.to_thread(
            temp_files.store, rendered.pdf, course_id=request.course_id,
            user_id=request.user_id, filename=f"{paper.title} exam paper",
            ttl_seconds=ttl_minutes * 60, kind="exam_paper",
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("chat.exam.render_failed id=%s", request_id)
        return ChatResponse(answer=exams.render_failed_message(), sources=[])
    logger.info("chat.exam.ready id=%s pages=%s questions=%s render_ms=%.2f total_ms=%.2f",
                request_id, rendered.pages, paper.question_count(),
                (time.perf_counter() - started) * 1000,
                (time.perf_counter() - request_started) * 1000)
    return ChatResponse(
        answer=exams.ready_message(ttl_minutes, paper),
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
    ``pending_action`` asking to widen a thin search; ``/quiz`` returns a
    10-question ``quiz``, or a ``pending_action`` offering a broader topic. The
    reply to an action arrives as ``pending_action_token`` + ``pending_decision``.
    ``/flashcards`` returns a 3-15 card ``flashcards`` deck, or a plain-text reply
    when the material is too thin (no pending action, no illustration).
    ``/exam`` in a course with past-paper blueprints returns a practice paper PDF
    as ``download`` (no answer key); without blueprints it is a normal answer.
    ``disabled_tools`` (e.g. ``["exam_generation"]``) are never run: an explicit
    tool_id or slash command of one gets a plain "turned off" answer, and a
    wording match falls through to the next route.
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
            # The student answered an earlier "widen the /notes search?" or
            # "broaden the /quiz topic?" question.
            route = _resolve_pending(request, request_id)
            if isinstance(route, str):
                return ChatResponse(answer=route, sources=[])
            expanded = True
        else:
            route_started = time.perf_counter()
            route = route_message(
                request.question,
                requested_tool_id=request.tool_id,
                disabled_tools=request.disabled_tools,
            )
            logger.info(
                "chat.route id=%s tool=%s explicit=%s duration_ms=%.2f",
                request_id,
                route.tool.id,
                route.explicit_command,
                (time.perf_counter() - route_started) * 1000,
            )
        # A tool Moodle turned off for this request never runs, including one a
        # pending action would resume.
        if route.blocked or route.tool.id in request.disabled_tools:
            logger.info("chat.route.disabled id=%s tool=%s", request_id, route.tool.id)
            return ChatResponse(answer=disabled_message(route.tool), sources=[])
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
        if result.get("outcome") == "exam_paper":
            return await _exam_paper_response(result, request, request_id, request_started)
        if result.get("outcome") == "quiz":
            logger.info("chat.finish id=%s outcome=quiz total_ms=%.2f", request_id,
                        (time.perf_counter() - request_started) * 1000)
            return ChatResponse(
                answer=answer,
                sources=result["sources"],
                retrieved_chunks=retrieval.chunk_count(matches),
                retrieved_contexts=[match["text"] for match in matches],
                quiz=result["quiz"],
            )
        if result.get("outcome") == "flashcards":
            logger.info("chat.finish id=%s outcome=flashcards total_ms=%.2f", request_id,
                        (time.perf_counter() - request_started) * 1000)
            return ChatResponse(
                answer=answer,
                sources=result["sources"],
                retrieved_chunks=retrieval.chunk_count(matches),
                retrieved_contexts=[match["text"] for match in matches],
                flashcards=result["flashcards"],
            )
        if result.get("outcome") == "needs_expansion":
            if quizzes.is_quiz(route.tool.id):
                action, topic = quizzes.EXPAND_ACTION, result.get("broader_topic", "")
                labels = ("Yes, broaden the topic", "No, don't create it")
            else:
                action, topic = sheets.EXPAND_ACTION, ""
                labels = ("Yes, search more widely", "No, cancel")
            token, expires_at = pending_actions.issue(
                action, course_id=request.course_id, user_id=request.user_id,
                question=question, tool_id=route.tool.id, topic=topic,
            )
            logger.info("chat.finish id=%s outcome=needs_expansion action=%s total_ms=%.2f",
                        request_id, action, (time.perf_counter() - request_started) * 1000)
            return ChatResponse(
                answer=answer, sources=[], retrieved_chunks=0, retrieved_contexts=[],
                pending_action=PendingAction(
                    type=action, token=token, expires_at=expires_at,
                    approve_label=labels[0], decline_label=labels[1],
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
            images=[
                ImageItem(
                    marker=image["marker"], image_id=image["image_id"],
                    document_id=image["document_id"], filename=image["filename"],
                    page=image["page"], caption=image["caption"], placed=image["placed"],
                    path=(f"/api/v1/document-images/{image['image_id']}"
                          f"?course_id={request.course_id}&document_id={quote(image['document_id'], safe='')}"),
                )
                for image in result.get("images") or []
            ],
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
