import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException, Request

from app.config import get_settings
from app.schemas import ChatRequest, ChatResponse, SourceItem
from app.security import RequireApiKey
from app.services.llm_provider import ProviderError, generate_answer_async
from app.services.illustrations import generate_illustration_async
from app.services.vector_store import search_course
from app.prompt_registry import get_tool_menu
from app.prompt_router import route_message

router = APIRouter(prefix="/chat", tags=["chat"], dependencies=[RequireApiKey])
logger = logging.getLogger(__name__)

# Retrieval uses Chroma's synchronous client and synchronous embedding SDKs.
# Keep this I/O-heavy work off the event loop and independent of Starlette's
# smaller default worker pool. Tune with RAG_RETRIEVAL_WORKERS for the host.
_settings = get_settings()
_retrieval_executor = ThreadPoolExecutor(
    max_workers=max(1, _settings.retrieval_workers),
    thread_name_prefix="rag-retrieval",
)


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

        retrieval_started = time.perf_counter()
        logger.info(
            "chat.retrieval.start id=%s course=%s top_k=%s",
            request_id,
            request.course_id,
            route.tool.top_k or settings.top_k,
        )
        loop = asyncio.get_running_loop()
        matches = await loop.run_in_executor(
            _retrieval_executor,
            search_course,
            question,
            request.course_id,
            request.user_id,
            request.include_course_documents,
            route.tool.top_k or settings.top_k,
        )
        logger.info(
            "chat.retrieval.finish id=%s chunks=%s duration_ms=%.2f",
            request_id,
            len(matches),
            (time.perf_counter() - retrieval_started) * 1000,
        )
        matches = [
            match for match in matches
            if match["metadata"].get("retrieval_enabled", True)
        ]
        if not matches:
            logger.info(
                "chat.no_context id=%s total_ms=%.2f",
                request_id,
                (time.perf_counter() - request_started) * 1000,
            )
            return ChatResponse(
                answer=(
                    "I couldn't find relevant text in the documents available to you "
                    "for this course. Upload or index course materials first, or try a "
                    "more specific question."
                ),
                sources=[],
                retrieved_chunks=0,
                retrieved_contexts=[],
            )

        context_started = time.perf_counter()
        passages, sources, source_indexes = [], [], {}
        for i, match in enumerate(matches, 1):
            metadata = match["metadata"]
            page = metadata.get("page", -1)
            page = page if isinstance(page, int) and page > 0 else None
            document_id = str(metadata.get("document_id", ""))
            passages.append(
                f"[S{i}] File: {metadata.get('filename', 'Unknown')}; "
                f"page/slide: {page or 'not available'}\n{match['text']}"
            )
            key = (document_id, page)
            source_index = source_indexes.get(key)
            if source_index is None:
                sources.append(
                    SourceItem(
                        document_id=document_id,
                        filename=str(metadata.get("filename", "Unknown")),
                        page=page,
                        chunk_id=match["chunk_id"],
                        excerpt=match["text"][:350],
                        markers=[f"S{i}"],
                    )
                )
                source_indexes[key] = len(sources) - 1
            else:
                sources[source_index].markers.append(f"S{i}")

        history = [item.model_dump() for item in request.history]
        context = "\n\n".join(passages)
        logger.info(
            "chat.context.ready id=%s chunks=%s sources=%s context_chars=%s history_messages=%s duration_ms=%.2f",
            request_id,
            len(matches),
            len(sources),
            len(context),
            len(history),
            (time.perf_counter() - context_started) * 1000,
        )

        generation_started = time.perf_counter()
        logger.info(
            "chat.llm.start id=%s provider=%s model=%s",
            request_id,
            request.provider or settings.llm_provider,
            request.model or settings.llm_model,
        )
        answer = await generate_answer_async(
            route=route,
            context=context,
            history=history,
            provider=request.provider,
            model=request.model,
        )
        logger.info(
            "chat.llm.finish id=%s answer_chars=%s duration_ms=%.2f",
            request_id,
            len(answer),
            (time.perf_counter() - generation_started) * 1000,
        )

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
            sources=sources,
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
