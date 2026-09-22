#!/usr/bin/env python3
"""Evaluate an OpenAI-style multimodal endpoint on MMBench without VLMEvalKit."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_DATA_URL = (
    "https://opencompass.openxlab.space/utils/benchmarks/MMBench/"
    "MMBench_DEV_CN_V11.tsv"
)
OPTION_NAMES = tuple("ABCD")
INDEX_MODULUS = 1_000_000


def set_csv_field_limit() -> None:
    """Allow csv to read the large base64 image field on all platforms."""
    field_limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(field_limit)
            return
        except OverflowError:
            field_limit //= 10


def download_dataset(url: str, destination: Path, insecure: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "mmbench-light/1.0"})
    context = ssl._create_unverified_context() if insecure else None
    print(f"Downloading {url}")
    try:
        with urllib.request.urlopen(request, timeout=60, context=context) as response:
            total = int(response.headers.get("Content-Length", "0"))
            received = 0
            with temporary.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    received += len(chunk)
                    if total:
                        print(
                            f"\rDownloaded {received / 1024 / 1024:.1f}/"
                            f"{total / 1024 / 1024:.1f} MiB",
                            end="",
                            flush=True,
                        )
        print()
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


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

    for row in rows:
        try:
            index = int(row["index"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid dataset index: {row.get('index')!r}") from exc
        row["index"] = str(index)
        row["base_index"] = str(index % INDEX_MODULUS)
    return rows


def image_table(rows: Iterable[Dict[str, str]]) -> Dict[str, str]:
    images: Dict[str, str] = {}
    for row in rows:
        value = (row.get("image") or "").strip()
        if value and not value.isdigit():
            images[row["index"]] = value
    return images


def resolve_image(row: Dict[str, str], images: Dict[str, str]) -> str:
    value = (row.get("image") or "").strip()
    if not value:
        raise ValueError(f"Row {row['index']} has no image")
    if value.isdigit():
        try:
            return images[value]
        except KeyError as exc:
            raise ValueError(
                f"Row {row['index']} refers to missing image row {value}"
            ) from exc
    return value


def ordered_base_indices(rows: Iterable[Dict[str, str]]) -> List[str]:
    result: List[str] = []
    seen = set()
    for row in rows:
        base_index = row["base_index"]
        if base_index not in seen:
            seen.add(base_index)
            result.append(base_index)
    return result


def select_rows(
    rows: Sequence[Dict[str, str]], mode: str, limit: Optional[int]
) -> List[Dict[str, str]]:
    base_indices = ordered_base_indices(rows)
    if limit is not None:
        base_indices = base_indices[:limit]
    selected_bases = set(base_indices)
    selected = [row for row in rows if row["base_index"] in selected_bases]
    if mode == "base":
        selected = [row for row in selected if row["index"] == row["base_index"]]
    return selected


def choices_for(row: Dict[str, str]) -> Dict[str, str]:
    return {
        name: (row.get(name) or "").strip()
        for name in OPTION_NAMES
        if (row.get(name) or "").strip()
    }


def build_prompt(row: Dict[str, str]) -> str:
    lines: List[str] = []
    hint = (row.get("hint") or "").strip()
    if hint:
        lines.append(f"提示：{hint}")
    lines.append(f"问题：{(row.get('question') or '').strip()}")
    lines.append("选项：")
    for name, value in choices_for(row).items():
        lines.append(f"{name}. {value}")
    lines.append("请只回答正确选项的一个大写字母（例如 A），不要解释。")
    return "\n".join(lines)


def infer_option(answer: str, choices: Dict[str, str]) -> Optional[str]:
    """Parse an option letter using VLMEvalKit-compatible direct matching."""
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

    patterns = (
        r"(?:CORRECT\s+)?ANSWER\s+IS\s+\**([A-D])\**",
        r"(?:答案|选项|选择)\s*(?:是|为|：|:)?\s*([A-D])",
    )
    for pattern in patterns:
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
    image: str,
    timeout: float,
    retries: int,
) -> Tuple[Dict[str, Any], float]:
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
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
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


def average_metric(records: Iterable[Dict[str, Any]], name: str) -> Optional[float]:
    values = [
        float(record[name])
        for record in records
        if isinstance(record.get(name), (int, float))
    ]
    return mean(values)


def make_summary(
    selected: Sequence[Dict[str, str]],
    records: Dict[str, Dict[str, Any]],
    mode: str,
) -> Dict[str, Any]:
    selected_indices = {row["index"] for row in selected}
    selected_records = [records[index] for index in selected_indices if index in records]
    completed_indices = {str(record["index"]) for record in selected_records}
    pending = len(selected_indices.difference(completed_indices))
    correct_rows = sum(bool(record.get("correct")) for record in selected_records)
    unparsed_rows = sum(record.get("prediction") is None for record in selected_records)

    expected_groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in selected:
        expected_groups[row["base_index"]].append(row)
    group_hits: Dict[str, int] = {}
    for base_index, group_rows in expected_groups.items():
        group_records = [records.get(row["index"]) for row in group_rows]
        if all(record is not None for record in group_records):
            group_hits[base_index] = int(
                all(bool(record.get("correct")) for record in group_records if record)
            )

    category_hits: Dict[str, List[int]] = defaultdict(list)
    base_rows = {row["base_index"]: row for row in selected if row["index"] == row["base_index"]}
    for base_index, hit in group_hits.items():
        category = (base_rows.get(base_index, {}).get("category") or "unknown").strip()
        category_hits[category].append(hit)

    complete = pending == 0
    official_accuracy = mean(list(group_hits.values())) if complete else None
    summary: Dict[str, Any] = {
        "mode": mode,
        "selected_rows": len(selected),
        "selected_base_questions": len(expected_groups),
        "completed_rows": len(selected_records),
        "pending_rows": pending,
        "unparsed_rows": unparsed_rows,
        "correct_rows": correct_rows,
        "row_accuracy": mean(
            [float(bool(record.get("correct"))) for record in selected_records]
        ),
        "completed_groups": len(group_hits),
        "official_accuracy": official_accuracy,
        "official_accuracy_percent": (
            official_accuracy * 100 if official_accuracy is not None else None
        ),
        "average_ttft_s": average_metric(selected_records, "ttft_s"),
        "average_total_s": average_metric(selected_records, "total_s"),
        "average_tps": average_metric(selected_records, "tokens_per_second"),
        "average_decode_tps": average_metric(
            selected_records, "decode_tokens_per_second"
        ),
        "category_accuracy": {
            category: mean(hits) for category, hits in sorted(category_hits.items())
        },
    }
    return summary


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
        f"  completed: {summary['completed_rows']}/{summary['selected_rows']} rows, "
        f"pending: {summary['pending_rows']}, unparsed: {summary['unparsed_rows']}"
    )
    row_accuracy = summary["row_accuracy"]
    print(
        "  row accuracy: "
        + (f"{row_accuracy * 100:.2f}%" if row_accuracy is not None else "n/a")
    )
    official_accuracy = summary["official_accuracy"]
    if official_accuracy is None:
        print("  official accuracy: n/a (finish pending rows first)")
    else:
        label = "CircularEval" if summary["mode"] == "circular" else "base-only"
        print(f"  {label} accuracy: {official_accuracy * 100:.2f}%")
    print(f"  summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-base",
        default="http://127.0.0.1:8082",
        help="API host, /api/v1 base, or full /chat/completions URL",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("eval/mmbench/MMBench_DEV_CN_V11.tsv"),
    )
    parser.add_argument("--data-url", default=DEFAULT_DATA_URL)
    parser.add_argument(
        "--download",
        action="store_true",
        help="download the dataset when --data does not exist",
    )
    parser.add_argument(
        "--insecure-download",
        action="store_true",
        help="disable TLS certificate verification for the dataset download",
    )
    parser.add_argument("--mode", choices=("base", "circular"), default="base")
    parser.add_argument(
        "--limit",
        type=int,
        help="evaluate only the first N base questions (all rotations in circular mode)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("eval/mmbench/results_base.jsonl")
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
    if not args.data.exists():
        if not args.download:
            print(
                f"Dataset not found: {args.data}\n"
                "Run again with --download (and --insecure-download if its certificate fails).",
                file=sys.stderr,
            )
            return 2
        try:
            download_dataset(args.data_url, args.data, args.insecure_download)
        except Exception as exc:
            print(f"Dataset download failed: {exc}", file=sys.stderr)
            return 2

    try:
        rows = load_dataset(args.data)
        selected = select_rows(rows, args.mode, args.limit)
        images = image_table(rows)
        records = completed_records(args.output)
    except (OSError, ValueError) as exc:
        print(f"Failed to prepare evaluation: {exc}", file=sys.stderr)
        return 2

    if not selected:
        print("No rows selected", file=sys.stderr)
        return 2

    endpoint = endpoint_from(args.api_base)
    pending = [row for row in selected if row["index"] not in records]
    print(
        f"Dataset: {len(rows)} rows; selected: {len(selected)} rows / "
        f"{len(ordered_base_indices(selected))} base questions"
    )
    print(f"Endpoint: {endpoint}")
    print(f"Output: {args.output} ({len(records)} completed records found)")

    for position, row in enumerate(pending, 1):
        index = row["index"]
        try:
            image = resolve_image(row, images)
            response, elapsed_s = call_api(
                endpoint, row, image, args.timeout, args.retries
            )
            raw_output = response_text(response)
            prediction = infer_option(raw_output, choices_for(row))
            answer = (row.get("answer") or "").strip().upper()
            record: Dict[str, Any] = {
                "status": "ok",
                "index": int(index),
                "base_index": int(row["base_index"]),
                "answer": answer,
                "prediction": prediction,
                "correct": prediction == answer,
                "raw_output": raw_output,
                "category": row.get("category", ""),
                "l2_category": row.get("l2-category", ""),
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
            record = {
                "status": "error",
                "index": int(index),
                "base_index": int(row["base_index"]),
                "error": str(exc),
            }
            append_record(args.output, record)
            print(f"[{position}/{len(pending)}] index={index} ERROR: {exc}")

    summary = make_summary(selected, records, args.mode)
    summary_path = write_summary(args.output, summary)
    print_summary(summary, summary_path)
    return 0 if summary["pending_rows"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
