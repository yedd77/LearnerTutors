import asyncio
import logging
import time

from fastapi import APIRouter, HTTPException, Request

from app.config import get_settings
from app.schemas import ChatRequest, ChatResponse
from app.security import RequireApiKey
from app.services.llm_provider import ProviderError
from app.services.illustrations import generate_illustration_async
from app.prompt_registry import get_tool_menu
from app.prompt_router import route_message
from app.rag_graph.context import RagContext
from app.rag_graph.graph import run_rag_graph

router = APIRouter(prefix="/chat", tags=["chat"], dependencies=[RequireApiKey])
logger = logging.getLogger(__name__)


@router.get("/tools")
def list_tools():
    return {"tools": get_tool_menu()}


@router.post("", response_model=ChatResponse)
async def ask(request: ChatRequest, http_request: Request):
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

        illustration_url = None
        if route.tool.id == "concept_explanation" and route.explicit_command:
            illustration_started = time.perf_counter()
            logger.info("chat.illustration.start id=%s", request_id)
            try:
                illustration_url = await generate_illustration_async(
                    question=question,
                    answer=answer,
                    context=context,
                )
                logger.info(
                    "chat.illustration.finish id=%s duration_ms=%.2f",
                    request_id,
                    (time.perf_counter() - illustration_started) * 1000,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("chat.illustration.error id=%s", request_id)

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
