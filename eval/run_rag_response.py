import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime, timezone
from pathlib import Path
from app.config import get_settings

import pandas as pd
import requests

s = get_settings()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "eval" / "output"

DEFAULT_BASE_URL = s.tunnel_url
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}

OUTPUT_COLUMNS = [
    "row_id", "user_input", "response", "retrieved_contexts", "reference",
    "persona_name", "query_style", "query_length", "synthesizer_name",
    "status", "http_status", "error", "retrieved_chunks", "empty_contexts",
    "latency_s", "attempts", "endpoint", "collected_at",
]


def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_input = OUTPUT_DIR / "evaluation_dataset_filtered.csv"
    if not default_input.exists():
        default_input = OUTPUT_DIR / "evaluation_dataset.csv"
    p.add_argument("--input", type=Path, default=default_input, help="Question dataset CSV")
    p.add_argument("--output", type=Path, default=OUTPUT_DIR / "rag_responses.csv")
    p.add_argument("--base-url", default=os.getenv("RAG_BASE_URL", DEFAULT_BASE_URL))
    p.add_argument("--endpoint", default=os.getenv("RAG_ENDPOINT"),
                   help="Chat path, e.g. /chat")
    p.add_argument("--course-id", type=int, default=os.getenv("RAG_COURSE_ID"))
    p.add_argument("--user-id", type=int, default=os.getenv("RAG_USER_ID"))
    p.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout, seconds")
    p.add_argument("--delay", type=float, default=1.0, help="Pause between questions, seconds")
    p.add_argument("--limit", type=int, default=None, help="Only use the first N dataset rows (smoke test)")
    p.add_argument("--max-retries", type=int, default=3, help="Retries per question on transient errors")
    p.add_argument("--backoff", type=float, default=5.0, help="Base retry backoff, seconds (doubles each retry)")
    args = p.parse_args()

    # API key is deliberately env-only: a CLI flag would leak into shell history.
    args.api_key = os.getenv("RAG_API_KEY")
    missing = []
    if not args.api_key:
        missing.append("RAG_API_KEY (environment or .env)")
    if args.course_id is None:
        missing.append("--course-id / RAG_COURSE_ID")
    if args.user_id is None:
        missing.append("--user-id / RAG_USER_ID")
    if missing:
        p.error("missing required values: " + "; ".join(missing))
    args.course_id = int(args.course_id)
    args.user_id = int(args.user_id)
    args.base_url = args.base_url.rstrip("/")
    return args


def redact(text, secret):
    text = str(text)
    return text.replace(secret, "***") if secret else text


def make_row_id(question, reference):
    raw = f"{question}\x1f{reference}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def load_questions(path, limit):
    if not path.exists():
        sys.exit(f"Input dataset not found: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ("user_input", "reference"):
        if col not in df.columns:
            sys.exit(f"Input dataset is missing required column '{col}'. Columns: {list(df.columns)}")

    total = len(df)
    if "synthesizer_name" in df.columns:
        mask = df["synthesizer_name"].str.contains("single_hop", case=False, na=False)
        df = df[~mask]
        print(f"Single-hop safety filter: removed {int(mask.sum())} of {total} rows")
    else:
        print("WARNING: no 'synthesizer_name' column; single-hop filter could not be applied")

    df = df[df["user_input"].str.strip() != ""].copy()
    for col in ("persona_name", "query_style", "query_length", "synthesizer_name"):
        if col not in df.columns:
            df[col] = ""
    df["row_id"] = [make_row_id(q, r) for q, r in zip(df["user_input"], df["reference"])]

    dupes = int(df["row_id"].duplicated().sum())
    if dupes:
        print(f"WARNING: {dupes} duplicate question/reference rows dropped")
        df = df.drop_duplicates("row_id")

    df = df.reset_index(drop=True)
    if limit is not None:
        df = df.head(limit)
    return df


def discover_endpoint(session, base_url, headers, timeout):
    """Find the POST route whose request body is ChatRequest, via the live /openapi.json."""
    url = f"{base_url}/openapi.json"
    try:
        r = session.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        spec = r.json()
    except Exception as exc:
        sys.exit(f"Could not read {url} to auto-detect the chat route ({type(exc).__name__}: {exc}).\n"
                 f"Pass --endpoint explicitly (e.g. --endpoint /chat).")
    found = []
    for path, item in spec.get("paths", {}).items():
        post = item.get("post")
        if not post:
            continue
        schema = (post.get("requestBody", {}).get("content", {})
                  .get("application/json", {}).get("schema", {}))
        if schema.get("$ref", "").endswith("/ChatRequest"):
            found.append(path)
    if len(found) == 1:
        return found[0]
    if not found:
        sys.exit("No POST route using the ChatRequest schema was found in /openapi.json. "
                 "Pass --endpoint explicitly.")
    sys.exit(f"Multiple POST routes use ChatRequest: {found}. Pass --endpoint explicitly.")


def retry_after_seconds(resp):
    val = resp.headers.get("Retry-After")
    if val and val.strip().isdigit():
        return float(val.strip())
    return None


def ask(session, url, headers, payload, args):
    """POST one question with retries. Returns a result dict; never raises on API errors."""
    attempts = 0
    started = time.perf_counter()
    result = {"status": "error", "http_status": "", "error": "", "response": "",
              "contexts": [], "retrieved_chunks": ""}
    while True:
        attempts += 1
        retryable = False
        wait = None
        try:
            resp = session.post(url, json=payload, headers=headers, timeout=args.timeout)
        except requests.RequestException as exc:
            result["http_status"] = ""
            result["error"] = redact(f"{type(exc).__name__}: {exc}", args.api_key)
            retryable = True
        else:
            result["http_status"] = resp.status_code
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except ValueError:
                    data = None
                if not isinstance(data, dict) or "answer" not in data:
                    result["error"] = "200 response was not the expected JSON (missing 'answer')"
                elif "retrieved_contexts" not in data:
                    result["error"] = ("response has no 'retrieved_contexts' field; "
                                       "restart the backend so the new schema is deployed")
                else:
                    result.update(status="ok", error="",
                                  response=data.get("answer") or "",
                                  contexts=list(data.get("retrieved_contexts") or []),
                                  retrieved_chunks=data.get("retrieved_chunks", ""))
                    break
            else:
                try:
                    detail = resp.json().get("detail", resp.text[:500])
                except Exception:
                    detail = resp.text[:500]
                result["error"] = redact(f"HTTP {resp.status_code}: {detail}", args.api_key)
                retryable = resp.status_code in RETRYABLE_STATUS
                wait = retry_after_seconds(resp)
                if resp.status_code == 401 or (resp.status_code == 503 and "RAG_API_KEY" in str(detail)):
                    result["fatal"] = True
                    break
        if not retryable or attempts > args.max_retries:
            break
        sleep_for = wait if wait is not None else args.backoff * (2 ** (attempts - 1))
        print(f"    retry {attempts}/{args.max_retries} in {sleep_for:.0f}s ({result['error'][:120]})")
        time.sleep(sleep_for)
    result["attempts"] = attempts
    result["latency_s"] = round(time.perf_counter() - started, 2)
    return result


def load_existing(path):
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {row["row_id"]: row.to_dict() for _, row in df.iterrows()}


def save_atomic(path, records, order):
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [records[i] for i in order if i in records]
    leftovers = [r for i, r in records.items() if i not in set(order)]  # keep rows from earlier datasets
    df = pd.DataFrame(ordered + leftovers)
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[OUTPUT_COLUMNS]
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(path)


def main():
    load_env()
    args = parse_args()

    questions = load_questions(args.input, args.limit)
    print(f"Questions in scope: {len(questions)}  (input: {args.input})")

    headers = {
        "X-RAG-API-Key": args.api_key,
        "Content-Type": "application/json",
        "X-Tunnel-Skip-AntiPhishing-Page": "true",  # harmless unless a Dev Tunnel shows its interstitial
    }
    session = requests.Session()

    endpoint = args.endpoint or discover_endpoint(session, args.base_url, headers, 30)
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    url = args.base_url + endpoint
    print(f"Chat endpoint: POST {url}")
    print(f"course_id={args.course_id} user_id={args.user_id} timeout={args.timeout}s delay={args.delay}s")

    records = load_existing(args.output)
    order = list(questions["row_id"])
    done = {i for i, r in records.items() if r.get("status") == "ok"}
    pending = [i for i in order if i not in done]
    print(f"Already collected OK: {len(done & set(order))} | to process now: {len(pending)}\n")

    by_id = questions.set_index("row_id")
    for n, row_id in enumerate(pending, start=1):
        q = by_id.loc[row_id]
        payload = {
            "question": q["user_input"],
            "course_id": args.course_id,
            "user_id": args.user_id,
            "include_course_documents": True,
            "history": [],
            "tool_id": None,
        }
        print(f"[{n}/{len(pending)}] {row_id}  {q['user_input'][:80]!r}")
        res = ask(session, url, headers, payload, args)

        contexts = res["contexts"]
        records[row_id] = {
            "row_id": row_id,
            "user_input": q["user_input"],
            "response": res["response"],
            "retrieved_contexts": json.dumps(contexts, ensure_ascii=False),
            "reference": q["reference"],
            "persona_name": q["persona_name"],
            "query_style": q["query_style"],
            "query_length": q["query_length"],
            "synthesizer_name": q["synthesizer_name"],
            "status": res["status"],
            "http_status": res["http_status"],
            "error": res["error"],
            "retrieved_chunks": res["retrieved_chunks"],
            "empty_contexts": res["status"] == "ok" and len(contexts) == 0,
            "latency_s": res["latency_s"],
            "attempts": res["attempts"],
            "endpoint": endpoint,
            "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        save_atomic(args.output, records, order)  # incremental checkpoint

        if res["status"] == "ok":
            flag = "  (NO CONTEXTS RETRIEVED)" if not contexts else ""
            print(f"    ok  {res['latency_s']}s  chunks={res['retrieved_chunks']}{flag}")
        else:
            print(f"    FAILED  {res['error'][:200]}")
        if res.get("fatal"):
            print("\nAuthentication/config error; stopping so you can fix it. Progress is saved.")
            sys.exit(2)
        if n < len(pending) and args.delay > 0:
            time.sleep(args.delay)

    final = [records[i] for i in order if i in records]
    ok = sum(1 for r in final if r["status"] == "ok")
    empty = sum(1 for r in final if str(r.get("empty_contexts")) == "True")
    failed = len(final) - ok
    print(f"\nDone. ok={ok} failed={failed} ok_but_no_contexts={empty} -> {args.output}")
    if failed:
        print("Re-run the same command to retry failed rows.")
        sys.exit(1)


if __name__ == "__main__":
    main()