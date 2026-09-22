#!/usr/bin/env python3
"""Evaluate the DIAL multimodal endpoint on the MMStar benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


OPTION_NAMES = tuple("ABCD")


def set_csv_field_limit() -> None:
    field_limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(field_limit)
            return
        except OverflowError:
            field_limit //= 10


def load_dataset(path: Path) -> List[Dict[str, str]]:
    set_csv_field_limit()
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Dataset has no header: {path}")
        required = {"index", "question", "answer", "image"}
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"Dataset is missing columns: {', '.join(missing)}")
        rows = [dict(row) for row in reader]

    seen = set()
    for row in rows:
        try:
            index = int(row["index"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid dataset index: {row.get('index')!r}") from exc
        if index in seen:
            raise ValueError(f"Duplicate dataset index: {index}")
        seen.add(index)
        row["index"] = str(index)
        answer = (row.get("answer") or "").strip().upper()
        if answer not in OPTION_NAMES:
            raise ValueError(f"Invalid answer for index {index}: {answer!r}")
        if not (row.get("image") or "").strip():
            raise ValueError(f"Row {index} has no image")
    return rows


def choices_for(row: Dict[str, str]) -> Dict[str, str]:
    text = (row.get("question") or "").strip()
    marker = re.search(r"\b(?:Options|Choices)\s*:\s*", text, flags=re.IGNORECASE)
    option_text = text[marker.end() :] if marker else text
    matches = list(
        re.finditer(
            r"(?:^|[\s,;])(?:\(([A-D])\)|\[([A-D])\]|([A-D])\s*[:.)])\s*",
            option_text,
            flags=re.IGNORECASE,
        )
    )
    choices: Dict[str, str] = {}
    for position, match in enumerate(matches):
        name = next(group for group in match.groups() if group is not None).upper()
        end = matches[position + 1].start() if position + 1 < len(matches) else len(option_text)
        value = option_text[match.end() : end].strip(" \t\r\n,;")
        if name in OPTION_NAMES and value:
            choices[name] = value
    answer = (row.get("answer") or "").strip().upper()
    if len(choices) < 2 or answer not in choices:
        index = row.get("index", "?")
        raise ValueError(f"Could not parse options for row {index}: {text[:240]!r}")
    return choices


def build_prompt(row: Dict[str, str]) -> str:
    option_names = ", ".join(choices_for(row))
    return (
        (row.get("question") or "").strip()
        + "\nAnswer with only the single uppercase letter of the correct option "
        + f"({option_names}). Do not explain."
    )


def infer_option(answer: str, choices: Dict[str, str]) -> Optional[str]:
    answer = str(answer).strip()
    if not answer:
        return None
    valid = set(choices)
    upper = answer.upper()
    if upper in valid:
        return upper

    normalized = upper
    for character in ".()[],:;!*#{}，。：；！（）【】":
        normalized = normalized.replace(character, " ")
    tokens = normalized.split()
    mentioned = [name for name in valid if name in tokens]
    if len(mentioned) == 1 and tokens.index(mentioned[0]) >= len(tokens) - 5:
        return mentioned[0]

    for pattern in (
        r"(?:CORRECT\s+)?ANSWER\s+IS\s+\**([A-D])\**",
        r"(?:OPTION|CHOICE)\s*(?:IS|:)?\s*([A-D])",
        r"(?:答案|选项|选择)\s*(?:是|为|：|:)?\s*([A-D])",
    ):
        match = re.search(pattern, upper, flags=re.IGNORECASE)
        if match and match.group(1) in valid:
            return match.group(1)

    total_choice_length = sum(len(value) for value in choices.values())
    if len(answer) <= 2 * max(total_choice_length, 1):
        text_matches = [
            name for name, value in choices.items() if value and value.lower() in answer.lower()
        ]
        if len(text_matches) == 1:
            return text_matches[0]
    return None


def endpoint_from(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/api/v1"):
        return base + "/chat/completions"
    return base + "/api/v1/chat/completions"


def response_text(response: Dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("API response has no choices[0].message.content") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    raise ValueError(f"Unsupported API content type: {type(content).__name__}")


def call_api(
    endpoint: str,
    row: Dict[str, str],
    timeout: float,
    retries: int,
) -> Tuple[Dict[str, Any], float]:
    image = (row.get("image") or "").strip()
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    },
                    {"type": "text", "text": build_prompt(row)},
                ],
            }
        ],
        "stream": False,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    http_request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        started = time.monotonic()
        try:
            with urllib.request.urlopen(http_request, timeout=timeout) as response:
                parsed = json.loads(response.read().decode("utf-8"))
            return parsed, time.monotonic() - started
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
        if attempt < retries:
            wait_seconds = min(2**attempt, 8)
            print(f"  request failed, retrying in {wait_seconds}s: {last_error}")
            time.sleep(wait_seconds)
    assert last_error is not None
    raise last_error


def append_record(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()


def completed_records(path: Path) -> Dict[str, Dict[str, Any]]:
    completed: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if record.get("status") == "ok":
                completed[str(record["index"])] = record
    return completed


def mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def average_metric(records: Sequence[Dict[str, Any]], name: str) -> Optional[float]:
    values = [
        float(record[name])
        for record in records
        if isinstance(record.get(name), (int, float))
    ]
    return mean(values)


def grouped_accuracy(
    records: Sequence[Dict[str, Any]], key: str
) -> Dict[str, Optional[float]]:
    groups: Dict[str, List[float]] = defaultdict(list)
    for record in records:
        label = str(record.get(key) or "unknown").strip()
        groups[label].append(float(bool(record.get("correct"))))
    return {label: mean(hits) for label, hits in sorted(groups.items())}


def make_summary(
    selected: Sequence[Dict[str, str]], records: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    selected_indices = {row["index"] for row in selected}
    selected_records = [records[index] for index in selected_indices if index in records]
    correct = sum(bool(record.get("correct")) for record in selected_records)
    pending = len(selected_indices) - len(selected_records)
    return {
        "dataset": "MMStar",
        "selected_questions": len(selected),
        "completed_questions": len(selected_records),
        "pending_questions": pending,
        "unparsed_questions": sum(
            record.get("prediction") is None for record in selected_records
        ),
        "correct_questions": correct,
        "accuracy": mean(
            [float(bool(record.get("correct"))) for record in selected_records]
        ),
        "accuracy_percent": (
            correct * 100.0 / len(selected_records) if selected_records else None
        ),
        "average_ttft_s": average_metric(selected_records, "ttft_s"),
        "average_total_s": average_metric(selected_records, "total_s"),
        "average_tps": average_metric(selected_records, "tokens_per_second"),
        "average_decode_tps": average_metric(
            selected_records, "decode_tokens_per_second"
        ),
        "category_accuracy": grouped_accuracy(selected_records, "category"),
        "l2_category_accuracy": grouped_accuracy(selected_records, "l2_category"),
        "source_benchmark_accuracy": grouped_accuracy(selected_records, "bench"),
    }


def write_summary(path: Path, summary: Dict[str, Any]) -> Path:
    summary_path = path.with_suffix(".summary.json")
    temporary = summary_path.with_name(summary_path.name + ".part")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, summary_path)
    return summary_path


def print_summary(summary: Dict[str, Any], summary_path: Path) -> None:
    print("\nEvaluation summary")
    print(
        f"  completed: {summary['completed_questions']}/"
        f"{summary['selected_questions']}, pending: {summary['pending_questions']}, "
        f"unparsed: {summary['unparsed_questions']}"
    )
    accuracy = summary["accuracy_percent"]
    print("  accuracy: " + (f"{accuracy:.2f}%" if accuracy is not None else "n/a"))
    print(f"  summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-base",
        default="http://127.0.0.1:8082",
        help="API host, /api/v1 base, or full /chat/completions URL",
    )
    parser.add_argument(
        "--data", type=Path, default=Path("eval/mmstar/MMStar.tsv")
    )
    parser.add_argument(
        "--limit", type=int, help="evaluate only the first N questions"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("eval/mmstar/results.jsonl")
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    if args.retries < 0:
        parser.error("--retries cannot be negative")
    return args


def main() -> int:
    args = parse_args()
    try:
        rows = load_dataset(args.data)
        selected = rows[: args.limit] if args.limit is not None else rows
        for row in selected:
            choices_for(row)
        records = completed_records(args.output)
    except (OSError, ValueError) as exc:
        print(f"Failed to prepare evaluation: {exc}", file=sys.stderr)
        return 2

    endpoint = endpoint_from(args.api_base)
    pending = [row for row in selected if row["index"] not in records]
    print(f"Dataset: {len(rows)} questions; selected: {len(selected)}")
    print(f"Endpoint: {endpoint}")
    print(f"Output: {args.output} ({len(records)} completed records found)")

    for position, row in enumerate(pending, 1):
        index = row["index"]
        try:
            choices = choices_for(row)
            response, elapsed_s = call_api(
                endpoint, row, args.timeout, args.retries
            )
            raw_output = response_text(response)
            prediction = infer_option(raw_output, choices)
            answer = (row.get("answer") or "").strip().upper()
            record: Dict[str, Any] = {
                "status": "ok",
                "index": int(index),
                "answer": answer,
                "prediction": prediction,
                "correct": prediction == answer,
                "raw_output": raw_output,
                "category": row.get("category", ""),
                "l2_category": row.get("l2_category", ""),
                "bench": row.get("bench", ""),
                "elapsed_s": elapsed_s,
                "ttft_s": response.get("ttft_s"),
                "total_s": response.get("total_s"),
                "tokens_per_second": response.get("tokens_per_second"),
                "decode_tokens_per_second": response.get(
                    "decode_tokens_per_second"
                ),
                "generated_tokens": response.get("generated_tokens"),
            }
            append_record(args.output, record)
            records[index] = record
            shown_prediction = prediction or "?"
            print(
                f"[{position}/{len(pending)}] index={index} "
                f"pred={shown_prediction} answer={answer} "
                f"{'OK' if record['correct'] else 'WRONG'} {elapsed_s:.2f}s"
            )
        except (OSError, ValueError, RuntimeError) as exc:
            append_record(
                args.output,
                {"status": "error", "index": int(index), "error": str(exc)},
            )
            print(f"[{position}/{len(pending)}] index={index} ERROR: {exc}")

    summary = make_summary(selected, records)
    summary_path = write_summary(args.output, summary)
    print_summary(summary, summary_path)
    return 0 if summary["pending_questions"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
