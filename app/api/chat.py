import logging
from fastapi import APIRouter, HTTPException
from app.config import get_settings
from app.schemas import ChatRequest, ChatResponse, SourceItem
from app.security import RequireApiKey
from app.services.openai_client import OpenAIError, generate_answer
from app.services.illustrations import generate_illustration
from app.services.vector_store import search_course
from app.prompt_registry import get_tool_menu
from app.prompt_router import route_message

router = APIRouter(prefix="/chat", tags=["chat"], dependencies=[RequireApiKey])
logger = logging.getLogger(__name__)

# return a list of supported tools for UI command menu
@router.get("/tools")
def list_tools():
    return {"tools": get_tool_menu()}

# route to handle chat request with retrieval and tool selection
@router.post("", response_model=ChatResponse)
def ask(request: ChatRequest):
    s = get_settings()
    try:
        route = route_message(request.question, requested_tool_id=request.tool_id)
        question = route.question.strip()
        if not question:
            raise HTTPException(
                status_code=422,
                detail="Enter a request after selecting the command, e.g. /exam Create 10 questions.",
            )

        # Retrieval is still course-scoped and uses the existing access filters.
        # Task-specific retrieval (e.g. separate lecture slides vs past papers)
        # is not available until document type metadata is added to indexing.
        matches = search_course(
            question,
            request.course_id,
            request.user_id,
            request.include_course_documents,
            route.tool.top_k or s.top_k,
        )
        
        # filter any matches that have retrieval disabled
        matches = [m for m in matches if m["metadata"].get("retrieval_enabled", True)]
        if not matches:
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

        # build context and source list for the answer generation
        passages, sources, source_indexes = [], [], {}
        for i, match in enumerate(matches, 1):
            meta = match["metadata"]
            page = meta.get("page", -1)
            page = page if isinstance(page, int) and page > 0 else None
            document_id = str(meta.get("document_id", ""))
            passages.append(
                f"[S{i}] File: {meta.get('filename', 'Unknown')}; "
                f"page/slide: {page or 'not available'}\n{match['text']}"
            )
            key = (document_id, page)
            source_index = source_indexes.get(key)
            if source_index is None:
                sources.append(
                    SourceItem(
                        document_id=document_id,
                        filename=str(meta.get("filename", "Unknown")),
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
        answer = generate_answer(
            route=route,
            context=context,
            history=history,
        )

        # Only an explicitly selected /explain command produces an illustration.
        # Image failures are non-fatal: preserve and return the text explanation.
        illustration_url = None
        if route.tool.id == "concept_explanation" and route.explicit_command:
            try:
                illustration_url = generate_illustration(
                    question=question,
                    answer=answer,
                    context=context,
                )
            except Exception:
                logger.exception("Could not generate the /explain illustration.")

        return ChatResponse(
            answer=answer,
            sources=sources,
            retrieved_chunks=len(matches),
            retrieved_contexts=[m["text"] for m in matches],
            illustration_url=illustration_url,
        )
    
    except HTTPException:
        raise
    except OpenAIError as exc:
        raise HTTPException(502, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"RAG request failed: {exc}") from exc