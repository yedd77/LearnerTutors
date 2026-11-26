# Frontend handoff: /exam papers in the lecturer's past-paper format (Moodle plugin `local_ragassistant`)

You are working in the Moodle plugin repository (`local_ragassistant`, repo
`LearnerTutors-FE`). The backend (FastAPI, repo `LearnerTutors`) is implemented on
branch `feat/exam-gen`. Your task is the plugin side only.
**Do not change the backend.** Do not commit or push unless the developer asks.

The feature in one paragraph: a lecturer uploads one or more past exam papers. The
backend reads each one and keeps only its **structure** (sections, question types,
number of questions, marks, sub-parts, time allowed when printed). It never makes the
paper searchable and never stores its questions. When a student sends `/exam`, the
backend writes **new** questions from the course materials in that structure and
returns a PDF practice paper, with **no answer key**. With several past papers it
combines them into one typical structure. A course without past papers keeps today's
`/exam` behaviour (a normal chat answer).

Before editing, inspect how the plugin currently:
- uploads a course document (the form, the task or endpoint that posts to backend
  `POST /api/v1/documents/index`, and where document records, status and suggestions are stored);
- deletes a document and toggles `retrieval_enabled`;
- reindexes documents (for example when the embedding configuration changes);
- renders the `/notes` study sheet `download` and proxies
  `GET /api/v1/generated-files/{file_id}` (see `docs/frontend-study-sheet-handoff.md`);
- checks capabilities (`local/ragassistant:managedocuments`, `useassistant`) and course settings.

Reuse those mechanisms. Don't create a second HTTP client or a second way to read settings.

---

## 1. Lecturer uploads a past paper

### 1a. Request: one new optional form field

Same multipart `POST {backend}/api/v1/documents/index` as today, plus:

| Field | Values | Meaning |
|---|---|---|
| `document_role` | `course_material` (default when omitted) or `past_exam` | `past_exam` stores the paper's structure for /exam instead of indexing it |

Anything else is a 400. Ordinary uploads don't need to send the field; their response is unchanged.

For `past_exam`:
- `visibility` must still be `course`.
- `retrieval_enabled` means "use this paper for /exam" (`false` keeps the structure but ignores it).
- `generate_suggestions` is ignored: no starter questions are made from a past paper, because
  they could reveal its questions.

### 1b. Response (201)

```json
{
  "document_id": "42",
  "filename": "BIO101 Final 2025.pdf",
  "course_id": 7,
  "indexed_chunks": 0,
  "status": "ready",
  "suggestions": [],
  "document_role": "past_exam",
  "exam_blueprint": {
    "description": "Section A: 20 multiple-choice questions, 1 mark each, 4 options each. Instructions: Answer ALL questions.\nSection B: 5 structured questions, candidates answer 3, 20 marks each, 3 sub-parts each, numbered (a), (b), ...\nTotal: 80 marks\nDuration: 120 minutes\nDifficulty mix: about 30% easy, 50% medium, 20% hard",
    "question_count": 25,
    "total_marks": 80,
    "duration_minutes": 120,
    "sections": [ { "title": "Section A", "question_type": "mcq", "question_count": 20, "...": "..." } ]
  }
}
```

- `indexed_chunks` is always 0: the paper is not searchable in chat.
- Show `exam_blueprint.description` to the lecturer (for example under the document in the
  document list, labelled "Detected exam structure"), so they can see what /exam will follow.
  Store it with the document record; the backend has no endpoint to fetch it later.
- `document_role` and `exam_blueprint` are **absent** for ordinary course material.

### 1c. Errors

| Status | Meaning | Suggested lecturer message |
|---|---|---|
| 422 | not an exam paper, or the file has no extractable text (scanned PDFs need OCR) | "This file doesn't look like an exam paper, so no exam structure was saved." (or the parser message) |
| 502 | analysis failed (no model configured in Moodle, provider error). Details are only in backend logs | "The exam structure couldn't be analysed. Please try again later." Mark the document failed |
| 409 | the `document_id` belongs to another course | existing handling |
| 400 / 413 / 415 | as for any upload | existing handling |

Analysis is one LLM call during the upload, using the generation model configured in
Moodle. Run it the same way ordinary indexing runs today (background task if that's
how uploads work), and keep the pending/processing/ready/failed states.

### 1d. Plugin changes

1. Upload form for users with `local/ragassistant:managedocuments`: a choice
   **Document type: Course material / Past exam paper** (language strings). Students
   must not be able to upload past papers.
2. Store the role with the document record (new column, e.g. `documentrole`, with an
   `db/upgrade.php` step and a version bump), plus the `exam_blueprint.description` text.
3. **Reindexing must resend `document_role=past_exam`** for past papers. If it is left
   out, the paper is indexed as normal course material and its questions become
   searchable in chat. An unchanged file is not analysed again (the backend compares
   file hashes), so reindexing costs no LLM calls.
4. Delete: same `DELETE /api/v1/documents/{id}?course_id=..` (200 with `deleted_chunks: 0`
   for a past paper).
5. Enable/disable: same `PATCH /api/v1/documents/{id}/retrieval?course_id=..`. For a past
   paper, label the toggle "Use for /exam".
6. Don't offer "regenerate suggestions" for past papers.
7. Keep past-paper files out of any student-facing document list (the decision is
   "structure only": students never see the past questions).

---

## 2. Student sends /exam

No request changes. In a course with at least one enabled past paper, a successful
`/exam` reply has the same `download` field as the `/notes` study sheet:

```json
{
  "answer": "Your practice exam paper is ready!\n\nIt follows the format of your lecturer's past papers, with new questions written from your course materials. It has no answer key.\n\nClick the button below to download your PDF. Please save it to your device: generated papers are only available for 1 hour and are then deleted automatically.",
  "sources": [ "...same shape as today..." ],
  "retrieved_chunks": 8,
  "retrieved_contexts": [ "..." ],
  "illustration_url": null,
  "download": {
    "file_id": "3f9c0d2e8b7a4c1f9e6d5b4a3c2b1a09",
    "filename": "Practice_Exam_Photosynthesis_exam_paper.pdf",
    "content_type": "application/pdf",
    "expires_at": 1795190400,
    "ttl_minutes": 60,
    "pages": 3
  }
}
```

- Reuse the study sheet's download button and the existing PHP download proxy. The file
  is fetched from the same `GET /api/v1/generated-files/{file_id}?course_id=..&user_id=..`,
  with `user_id = $USER->id` (never a request parameter).
- Show `answer` as the message text; it already contains the "save it, it's temporary" notice.
- `pages` is often more than 1 for an exam paper; no extra message is needed.

Other replies are plain text with no `download` and no `pending_action`:
- not enough course material for a full paper;
- the paper could not be written in the required format after one retry;
- the PDF could not be created.

If the student asks for a specific structure (for example "/exam 10 MCQs on chapter 3"),
the backend follows the request instead of the past-paper format and says so in `answer`.

A course with no enabled past paper: `/exam` replies exactly as before (Markdown answer, no `download`).

---

## 3. Test checklist (plugin side)

1. Lecturer uploads a real past paper as "Past exam paper": status becomes ready and the
   detected structure is shown.
2. Uploading lecture notes as "Past exam paper" shows the "doesn't look like an exam paper"
   message and marks the document failed.
3. A student asks the chat about a question that appears only in the past paper: it is not
   found (the paper is not searchable).
4. `/exam` in that course returns a PDF download; the PDF opens and follows the past
   paper's sections and marks, with new questions and no answers.
5. Another student cannot download the first student's paper (proxy uses `$USER->id`).
6. Disable the past paper ("Use for /exam" off): `/exam` goes back to a normal chat answer.
7. Delete the past paper: same result.
8. Trigger a reindex: the past paper stays a past paper (still not searchable).
9. A course without past papers: `/exam` is unchanged.
10. A student without `managedocuments` cannot see or use the document-type choice.
