# Frontend handoff: /notes study sheet PDF (Moodle plugin `local_ragassistant`)

You are working in the Moodle plugin repository (`local_ragassistant`, repo
`LearnerTutors-FE`). The backend (FastAPI, repo `LearnerTutors`) is already
implemented on branch `feat/notes-gen`. Your task is the plugin side only.
**Do not change the backend.** Do not commit or push unless the developer asks.

Before editing, inspect how the plugin currently:
- sends a chat message (the `fetch(form.action, ...)` call in `chat.js` and the PHP
  endpoint it posts to, which builds the backend `POST /api/v1/chat` body and adds
  the `X-RAG-API-Key` header);
- renders an assistant reply (`createMessage`, `appendIllustration`,
  `appendSources` in `chat.js`);
- checks course access and capabilities (e.g. `local/ragassistant:useassistant`,
  the `allowstudentchat` course setting);
- reads the backend URL and API key from plugin settings.

Reuse those mechanisms. Don't create a second HTTP client or a second way to read settings.

---

## 1. What the backend now returns

`POST /api/v1/chat` is unchanged for every existing command. Two **optional**
response fields were added. They are **omitted from the JSON** when unused, so
existing replies are byte-for-byte the same.

### 1a. `download`: a study sheet PDF is ready

Returned when a `/notes` request succeeds:

```json
{
  "answer": "Your study sheet is ready!\n\nClick the button below to download your PDF.\n\nPlease download and save it to your device. Generated study sheets are only available for 1 hour and are then deleted automatically. They are not stored permanently in LearnerTutors.",
  "sources": [ ...same shape as today... ],
  "retrieved_chunks": 6,
  "retrieved_contexts": [ "..." ],
  "illustration_url": null,
  "download": {
    "file_id": "3f9c0d2e8b7a4c1f9e6d5b4a3c2b1a09",
    "filename": "Photosynthesis_study_sheet.pdf",
    "content_type": "application/pdf",
    "expires_at": 1795190400,
    "ttl_minutes": 60,
    "pages": 1
  }
}
```

- `file_id` is 32 lowercase hex characters. It is **not** a URL and **not** a path.
- `expires_at` is Unix time (seconds). After it, the file is deleted.
- `pages` is usually 1. When it is greater than 1, `answer` already tells the student why.
- Show `answer` as the message text. It already contains the "download and save it,
  it's temporary" notice, so don't duplicate it.

### 1b. `pending_action`: the student must explicitly approve or decline

Returned when `/notes` found too little course material:

```json
{
  "answer": "I found only limited information about \"photosynthesis\" in the course materials I searched, so a study sheet made from it would be incomplete.\n\nWould you like me to search more widely across this course's documents to create more complete notes?",
  "sources": [],
  "retrieved_chunks": 0,
  "retrieved_contexts": [],
  "illustration_url": null,
  "pending_action": {
    "type": "expand_notes_search",
    "token": "eyJhIjoi...<opaque>...",
    "expires_at": 1795188600,
    "approve_label": "Yes, search more widely",
    "decline_label": "No, cancel"
  }
}
```

The `token` is opaque and signed. Never parse, modify or build it. It is tied to
this course and this user and expires after about 30 minutes.

### 1c. New optional request fields on `POST /api/v1/chat`

| Field | Type | Meaning |
|---|---|---|
| `pending_action_token` | string (up to 4096) | the `token` from `pending_action` |
| `pending_decision` | `"approve"` or `"decline"` | the student's choice |

Send **both or neither**; sending only one is a 422. `question` is still required
(at least 2 characters), so send the button label (e.g. "Yes, search more widely").
The backend ignores that text and re-runs the original request stored in the
token. `course_id`, `user_id`, `provider`, `model`, etc. are sent exactly as for
any chat request.

Possible replies to a decision:
- approve and enough material → a normal `download` reply (1a);
- approve but still too little → `answer` explains honestly, with no `download` and no new `pending_action`;
- decline → a short "cancelled" `answer`, with no `download`;
- expired, invalid or another user's token → `answer` says the request expired and asks the student to send `/notes` again.

### 1d. New backend download endpoint (server-to-server only)

```
GET {backend}/api/v1/generated-files/{file_id}?course_id={courseid}&user_id={userid}
Header: X-RAG-API-Key: <plugin's configured key>
```

| Status | Meaning |
|---|---|
| 200 | PDF bytes, `Content-Type: application/pdf`, `Content-Disposition: attachment; filename="..."`, `Cache-Control: private, no-store` |
| 404 | unknown ID, or the file belongs to another course or user (deliberately the same response) |
| 410 | the file expired and has been deleted |
| 401 / 503 | API key problem (backend configuration) |

The browser must never call this directly, because it needs the API key.

---

## 2. Task A: PHP download proxy (security-critical)

Create a plugin endpoint, for example `download_sheet.php` (pick a name that fits
the plugin's conventions). The browser calls it; it calls the backend endpoint above.

Requirements:
1. `require_login($course)` with the course from `required_param('courseid', PARAM_INT)`.
2. Check the same capability and course settings the chat endpoint checks (e.g.
   `local/ragassistant:useassistant` in the course context, `allowstudentchat`).
   Reuse the existing check; don't re-implement it differently.
3. `file_id = required_param('id', PARAM_ALPHANUM)` and also validate it against
   `/^[0-9a-f]{32}$/`. Reject anything else.
4. **`user_id` must be `$USER->id`**, never a request parameter. This is what stops
   students from downloading each other's PDFs.
5. Call the backend with the existing HTTP client and settings, sending the API key header.
6. On 200: stream the bytes to the browser with `Content-Type: application/pdf`,
   `Content-Disposition: attachment; filename="<backend filename>"` (sanitise it),
   and `Cache-Control: private, no-store`. Don't cache it in Moodle file storage;
   PDFs must not be stored permanently.
7. On 404 or 410: return a clear error the JavaScript can show (e.g. JSON
   `{"error": "expired"}` or `{"error": "notfound"}` with the same HTTP status).
   Use language strings.
8. Never expose the backend URL, API key or backend error bodies to the browser.
9. Don't log the PDF contents.

## 3. Task B: forward the new chat request fields

In the PHP endpoint that builds the backend chat request body, add
`pending_action_token` and `pending_decision` **only when both are present** in
the incoming request:
- `pending_decision` must be exactly `approve` or `decline`; drop anything else;
- `pending_action_token` is a string up to 4096 characters (use `PARAM_RAW_TRIMMED`
  or similar, then length-check it).

Keep `course_id` and `user_id` from the server side (course context and `$USER->id`)
as today. Pass through the new response fields (`download`, `pending_action`) to
the JavaScript unchanged, if the PHP endpoint filters response fields.

## 4. Task C: Download PDF button in the chat (`chat.js` + `styles.css` + lang)

Add `appendDownload(content, result.download)` next to the existing
`appendIllustration(...)` / `appendSources(...)` calls. Render it only when
`result.download` is an object with a valid `file_id`.

Button requirements:
- A prominent button containing a PDF/download icon and the text **"Download PDF"**
  (language string). Use Moodle's icon system (e.g. the `f/pdf` file icon or the
  Font Awesome download icon used elsewhere in Moodle) and the theme's button
  classes (`btn btn-primary`), following the plugin's existing class prefix
  (`ragassistant-...`).
- Show a small line under it such as "Available until 14:35" (from `expires_at`,
  in the user's local time) and the file name.
- On click, fetch the proxy URL (`download_sheet.php?courseid=..&id=..`):
  - 200 → turn the response into a Blob and trigger a download with
    `download.filename` (object URL plus a temporary `<a download>`; revoke the URL afterwards);
  - 404 or 410 → don't navigate. Replace the button with an inline notice
    ("This study sheet has expired. Send /notes again to create a new one.") and
    disable it.
  - Network error → inline retry message; keep the button usable.
- Before fetching, if `Date.now()/1000 > expires_at`, show the expired notice straight away.
- Show a busy state while downloading (disable the button and set `aria-busy`).
- Accessibility: a real `<button>`, a visible focus style and an `aria-label`
  including the file name.
- Works on mobile: a full-width button on narrow screens and a touch target of at least 44px.
  On iOS Safari the PDF may open in a viewer instead of saving; that is acceptable.
- If chat history is re-rendered from saved messages, render the button again
  only when the saved message kept its `download` data, showing the expired state
  when appropriate. If the plugin doesn't save it, skip this.

## 5. Task D: Approve/Decline buttons for `pending_action`

When `result.pending_action` is present and `type === "expand_notes_search"`:
- Under the answer, render two buttons using `approve_label` and `decline_label`
  (primary and secondary styles).
- On click, disable **both** buttons immediately (no double submit), then send a
  normal chat request with `question` set to the clicked label,
  `pending_action_token` set to the token, and `pending_decision` set to
  `"approve"` or `"decline"`. Show the clicked label as the student's message,
  then render the reply like any other (it may contain `download`).
- If `expires_at` has passed, render the buttons disabled with a note that the
  request expired.
- **Never** turn typed text such as "yes" into an approval. Only these buttons
  send `pending_decision`. A typed message is sent as a normal chat message without
  the token.

## 6. Acceptance checks

1. `/notes <topic with enough material>` → message with the temporary-file notice
   plus a working Download PDF button. The file downloads and opens as a one-page sheet.
2. `/notes <topic with little material>` → message with approve and decline buttons.
   Approve → widened search, then the button appears or there's an honest
   "still not enough" reply. Decline → cancelled message and no PDF.
3. Typing "yes" after the permission question does **not** trigger the widened search.
4. Student B pasting student A's proxy URL (same course) → not found, no PDF.
5. A file past `expires_at` (or after the backend deleted it) → the inline
   "expired" notice, no broken download and no raw error page.
6. `/ask`, `/exam`, `/summarize`, `/flashcards`, `/explain` (with illustration),
   `/quiz` and `/studyguide` render exactly as before.
7. Mobile width (about 375px): the buttons are full-width and usable.
8. No backend URL, API key or `user_id` parameter appears in the browser's network tab.

Report back with the files you created or changed, how you tested each check above,
and anything you couldn't verify.
