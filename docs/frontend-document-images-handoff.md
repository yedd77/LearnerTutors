# Document figures in chat answers: Moodle handoff

The backend now extracts images from uploaded PDF, DOCX and PPTX files and can
show them in answers. This note describes what the Moodle plugin
(`local_ragassistant`) needs to do. Nothing changes for answers without figures.

## Response field

`POST /api/v1/chat` may include an `images` list. It is omitted when there are
no figures, so existing replies are unchanged.

```json
{
  "answer": "The cycle regenerates oxaloacetate [S1].\n[IMG1]\nEach turn ...",
  "images": [
    {
      "marker": "IMG1",
      "image_id": "3f9c…32 hex…",
      "document_id": "42",
      "filename": "krebs.pdf",
      "page": 4,
      "caption": "Figure 1: The Krebs cycle",
      "placed": true,
      "path": "/api/v1/document-images/3f9c…?course_id=7&document_id=42"
    }
  ]
}
```

- `placed: true`: the answer contains `[IMG1]` on its own line where the figure belongs.
- `placed: false`: the figure comes from the same passages but the model did not place it.
  Show these after the answer (for example under "Figures"), or hide them.
- Markers in the answer are always `[IMGn]`, only for listed images, each at most once.

## Rendering

1. Replace each `[IMGn]` line in the answer with an `<img>` for the matching item
   (`alt` = `caption`, with the file name and page as a small caption line).
2. Render unplaced images after the answer, if you show them.
3. Do this after Markdown rendering, or protect the markers from the Markdown renderer.

## Fetching an image (must be proxied)

`GET {backend}{path}` with the `X-RAG-API-Key` header returns `image/png`.
Course figures are course material, so the browser must never call the backend
directly and the API key must never reach the browser:

1. Add a plugin endpoint (like the study-sheet download proxy) that takes
   `image_id`, `course_id` and `document_id`.
2. Check `require_login`, course access and the `useassistant` capability for
   that course.
3. Fetch from the backend with the API key and stream the PNG back, with
   `Cache-Control: private`.

The backend returns 404 when the image does not belong to that course document,
or the document was deleted, disabled for retrieval or re-indexed without the image.

## Existing documents

Images are extracted at indexing time. Documents indexed before this change
have no figures until they are re-indexed.
