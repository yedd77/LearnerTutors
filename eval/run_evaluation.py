from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_INPUT = PROJECT_ROOT / "eval" / "output" / "rag_responses.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval" / "results"

METRIC_COLUMNS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate collected RAG responses using RAGAS 0.4.3."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "evaluation_results.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "evaluation_summary.json",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--delay",
        type=float,
        default=float(os.getenv("RAGAS_DELAY", "0")),
        help="Pause between rows (seconds), useful for local Ollama.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("RAGAS_TIMEOUT", "600")),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(os.getenv("RAGAS_MAX_RETRIES", "3")),
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-evaluate rows that already have a result but contain an error.",
    )
    return parser.parse_args()


def load_input(path: Path, limit: int | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    required = {"user_input", "response", "retrieved_contexts", "reference"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    if "status" in df.columns:
        failed = df["status"].str.lower().ne("ok")
        if failed.any():
            print(
                f"Skipping {int(failed.sum())} rows whose collection status is not 'ok'."
            )
            df = df.loc[~failed].copy()

    for column in ("user_input", "response", "reference"):
        df = df[df[column].str.strip().ne("")]

    if limit is not None:
        df = df.head(limit)

    return df.reset_index(drop=True)


def parse_contexts(value: str, row_number: int) -> list[str]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Row {row_number}: retrieved_contexts is not valid JSON: {exc}"
        ) from exc

    if not isinstance(parsed, list):
        raise ValueError(
            f"Row {row_number}: retrieved_contexts must be a JSON array."
        )

    contexts = [str(item).strip() for item in parsed if str(item).strip()]
    return contexts


def make_evaluators(timeout: float):
    """
    Ollama exposes an OpenAI-compatible API at /v1.

    Configure RAGAS_LLM_MODEL explicitly in .env. The embedding model may
    reuse the backend embedding-model setting through RAGAS_EMBEDDING_MODEL.
    """
    base_url = os.getenv("RAGAS_BASE_URL", "http://localhost:11434/v1").rstrip("/")
    api_key = os.getenv("RAGAS_API_KEY", "ollama")
    llm_model = os.getenv("RAGAS_LLM_MODEL", "").strip()
    if not llm_model:
        raise RuntimeError("Set RAGAS_LLM_MODEL to the evaluation generation model.")
    embedding_model = os.getenv("RAGAS_EMBEDDING_MODEL", "nomic-embed-text").strip()

    async_client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=0,
    )
    sync_client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=0,
    )

    evaluator_llm = llm_factory(llm_model, client=async_client, max_tokens=4096)
    evaluator_embeddings = embedding_factory(
        "openai",
        model=embedding_model,
        client=async_client,
        interface="modern",
    )

    metrics = {
        "faithfulness": Faithfulness(llm=evaluator_llm, max_retries=3),
        "answer_relevancy": AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        ),
        "context_precision": ContextPrecision(llm=evaluator_llm),
        "context_recall": ContextRecall(llm=evaluator_llm),
    }

    return metrics, async_client, sync_client, llm_model, embedding_model, base_url


def load_existing_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "row_id" not in df.columns:
        return {}

    return {
        str(row["row_id"]): row.to_dict()
        for _, row in df.iterrows()
        if row.get("row_id")
    }


def stable_row_id(row: pd.Series, index: int) -> str:
    row_id = str(row.get("row_id", "")).strip()
    return row_id or f"row-{index + 1:06d}"


def save_results(path: Path, records: dict[str, dict[str, Any]], order: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [records[key] for key in order if key in records]
    columns = [
        "row_id",
        "user_input",
        "response",
        "reference",
        "retrieved_contexts",
        *METRIC_COLUMNS,
        "evaluation_status",
        "evaluation_error",
        "evaluation_latency_s",
        "evaluated_at",
    ]
    output = pd.DataFrame(ordered)
    for column in columns:
        if column not in output.columns:
            output[column] = ""
    output = output[columns]
    temp_path = path.with_suffix(path.suffix + ".tmp")
    output.to_csv(temp_path, index=False, encoding="utf-8")
    temp_path.replace(path)


async def score_one(
    row: pd.Series,
    row_number: int,
    metrics: dict[str, Any],
    max_retries: int,
) -> dict[str, Any]:
    contexts = parse_contexts(row["retrieved_contexts"], row_number)

    sample = {
        "user_input": row["user_input"].strip(),
        "response": row["response"].strip(),
        "retrieved_contexts": contexts,
        "reference": row["reference"].strip(),
    }

    scores: dict[str, Any] = {}
    errors: list[str] = []
    metric_samples = {
        "faithfulness": {
            "user_input": sample["user_input"],
            "response": sample["response"],
            "retrieved_contexts": sample["retrieved_contexts"],
        },
        "answer_relevancy": {
            "user_input": sample["user_input"],
            "response": sample["response"],
        },
        "context_precision": {
            "user_input": sample["user_input"],
            "reference": sample["reference"],
            "retrieved_contexts": sample["retrieved_contexts"],
        },
        "context_recall": {
            "user_input": sample["user_input"],
            "reference": sample["reference"],
            "retrieved_contexts": sample["retrieved_contexts"],
        },
    }

    for metric_name, metric in metrics.items():
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                result = await metric.ascore(**metric_samples[metric_name])
                value = result.value
                if value is None:
                    scores[metric_name] = ""
                else:
                    numeric_value = float(value)
                    scores[metric_name] = (
                        numeric_value if math.isfinite(numeric_value) else ""
                    )
                last_error = None
                break
            except Exception as exc:  # retain row-level failures for diagnosis
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < max_retries:
                    wait = min(5 * (2**attempt), 30)
                    print(
                        f"      {metric_name}: retry {attempt + 1}/{max_retries} "
                        f"in {wait}s ({last_error[:160]})"
                    )
                    await asyncio.sleep(wait)

        if last_error:
            scores[metric_name] = ""
            errors.append(f"{metric_name}: {last_error}")

    return {
        "scores": scores,
        "error": " | ".join(errors),
        "contexts": contexts,
    }


def write_summary(
    path: Path,
    results: pd.DataFrame,
    llm_model: str,
    embedding_model: str,
    base_url: str,
) -> None:
    summary: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ragas_version": "0.4.3",
        "evaluator_llm": llm_model,
        "evaluator_embedding_model": embedding_model,
        "evaluator_base_url": base_url,
        "total_rows": int(len(results)),
        "successful_rows": int(
            results["evaluation_status"].eq("ok").sum()
        ) if len(results) else 0,
        "failed_rows": int(
            results["evaluation_status"].ne("ok").sum()
        ) if len(results) else 0,
        "mean_scores": {},
    }

    for metric in METRIC_COLUMNS:
        values = pd.to_numeric(results[metric], errors="coerce").dropna()
        summary["mean_scores"][metric] = (
            round(float(values.mean()), 4) if not values.empty else None
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


async def run() -> int:
    args = parse_args()
    source = load_input(args.input, args.limit)

    if source.empty:
        print("No valid collected rows found. Nothing to evaluate.")
        return 1

    (
        metrics,
        async_client,
        sync_client,
        llm_model,
        embedding_model,
        base_url,
    ) = make_evaluators(args.timeout)

    existing = load_existing_results(args.output)
    row_ids = [
        stable_row_id(row, index)
        for index, (_, row) in enumerate(source.iterrows())
    ]

    # Only skip rows that completed all four metrics without an evaluation error.
    pending_indices = []
    for index, row_id in enumerate(row_ids):
        previous = existing.get(row_id)
        if previous:
            complete = (
                previous.get("evaluation_status") == "ok"
                and not previous.get("evaluation_error")
                and all(str(previous.get(metric, "")).strip() for metric in METRIC_COLUMNS)
            )
            if complete or (
                not args.retry_failed
                and previous.get("evaluation_status") == "ok"
                and not previous.get("evaluation_error")
            ):
                continue
        pending_indices.append(index)

    print(f"Input rows: {len(source)}")
    print(f"Already evaluated: {len(source) - len(pending_indices)}")
    print(f"To evaluate: {len(pending_indices)}")
    print(f"RAGAS evaluator: {llm_model}")
    print(f"Embedding model: {embedding_model}")
    print(f"Evaluator URL: {base_url}")
    print("Metrics: " + ", ".join(METRIC_COLUMNS) + "\n")

    records = existing
    started_all = time.perf_counter()

    try:
        for count, index in enumerate(pending_indices, start=1):
            row = source.iloc[index]
            row_id = row_ids[index]
            print(
                f"[{count}/{len(pending_indices)}] {row_id} "
                f"{row['user_input'][:90]!r}"
            )
            started = time.perf_counter()

            base_record = {
                "row_id": row_id,
                "user_input": row["user_input"],
                "response": row["response"],
                "reference": row["reference"],
                "retrieved_contexts": row["retrieved_contexts"],
                **{metric: "" for metric in METRIC_COLUMNS},
                "evaluation_status": "error",
                "evaluation_error": "",
                "evaluation_latency_s": "",
                "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }

            try:
                result = await score_one(
                    row=row,
                    row_number=index + 2,  # CSV header is line 1
                    metrics=metrics,
                    max_retries=args.max_retries,
                )
                base_record.update(result["scores"])
                base_record["evaluation_error"] = result["error"]
                base_record["evaluation_status"] = (
                    "ok" if not result["error"] else "partial"
                )
                if not result["contexts"]:
                    print("    WARNING: no retrieved contexts; retrieval metrics may be invalid.")
            except Exception as exc:
                base_record["evaluation_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )

            base_record["evaluation_latency_s"] = round(
                time.perf_counter() - started, 2
            )
            records[row_id] = base_record
            save_results(args.output, records, row_ids)

            scores_text = " | ".join(
                f"{name}={base_record[name]}"
                for name in METRIC_COLUMNS
            )
            print(
                f"    {base_record['evaluation_status']} "
                f"{base_record['evaluation_latency_s']}s | {scores_text}"
            )
            if base_record["evaluation_error"]:
                print(f"    ERROR: {base_record['evaluation_error'][:500]}")

            if args.delay > 0 and count < len(pending_indices):
                await asyncio.sleep(args.delay)

    finally:
        await async_client.close()
        sync_client.close()

    final_records = [records[key] for key in row_ids if key in records]
    final_df = pd.DataFrame(final_records)
    write_summary(
        args.summary,
        final_df,
        llm_model=llm_model,
        embedding_model=embedding_model,
        base_url=base_url,
    )

    successful = int(final_df["evaluation_status"].eq("ok").sum())
    failed = len(final_df) - successful
    print(
        f"\nFinished in {time.perf_counter() - started_all:.1f}s. "
        f"successful={successful} failed_or_partial={failed}"
    )
    print(f"Results: {args.output}")
    print(f"Summary: {args.summary}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(run()))
    except KeyboardInterrupt:
        print("\nInterrupted. Completed rows have been checkpointed.")
        raise SystemExit(130)
