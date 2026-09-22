#!/usr/bin/env python3
"""Evaluate DIAL's native Video module on MVBench.

This script uses only the Python standard library and ffmpeg. The DIAL server
performs Qwen3-VL's default uniform video sampling (2 FPS, 4..768 frames).
For MVBench rows with start/end annotations, it losslessly transcodes the
annotated interval in memory before sending it to DIAL.
"""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence
from urllib import error, request


OPTION_NAMES = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
VIDEO_EXTENSIONS = {
    ".avi",
    ".flv",
    ".gif",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}

# Task names and annotation filenames follow the official MVBench notebook.
TASK_FILES = {
    "Action Sequence": "action_sequence.json",
    "Action Prediction": "action_prediction.json",
    "Action Antonym": "action_antonym.json",
    "Fine-grained Action": "fine_grained_action.json",
    "Unexpected Action": "unexpected_action.json",
    "Object Existence": "object_existence.json",
    "Object Interaction": "object_interaction.json",
    "Object Shuffle": "object_shuffle.json",
    "Moving Direction": "moving_direction.json",
    "Action Localization": "action_localization.json",
    "Scene Transition": "scene_transition.json",
    "Action Count": "action_count.json",
    "Moving Count": "moving_count.json",
    "Moving Attribute": "moving_attribute.json",
    "State Change": "state_change.json",
    "Fine-grained Pose": "fine_grained_pose.json",
    "Character Order": "character_order.json",
    "Egocentric Navigation": "egocentric_navigation.json",
    "Episodic Reasoning": "episodic_reasoning.json",
    "Counterfactual Inference": "counterfactual_inference.json",
}

SYSTEM_PROMPT = (
    "Carefully watch the video and pay attention to the cause and sequence of events, "
    "the detail and movement of objects, and the action and pose of persons. Based on "
    "your observations, select the best option that accurately addresses the question."
)


@dataclass(frozen=True)
class Case:
    case_id: str
    task: str
    annotation_file: Path
    row_index: int
    video_name: str
    video_path: Path
    question: str
    candidates: tuple[str, ...]
    answer_text: str
    answer_option: str
    start: float | None
    end: float | None


def endpoint_from(api_base: str) -> str:
    base = api_base.strip().rstrip("/")
    if not base.startswith(("http://", "https://")):
        base = "http://" + base
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/api/v1"):
        return base + "/chat/completions"
    return base + "/api/v1/chat/completions"


def guess_media_type(path: Path) -> str:
    return {
        ".avi": "video/x-msvideo",
        ".gif": "image/gif",
        ".m4v": "video/mp4",
        ".mkv": "video/x-matroska",
        ".mov": "video/quicktime",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
    }.get(path.suffix.lower(), "application/octet-stream")


def normalize_task(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def selected_tasks(raw_tasks: Sequence[str] | None) -> list[tuple[str, str]]:
    if not raw_tasks:
        return list(TASK_FILES.items())
    aliases = {normalize_task(name): (name, filename) for name, filename in TASK_FILES.items()}
    aliases.update(
        {normalize_task(Path(filename).stem): (name, filename) for name, filename in TASK_FILES.items()}
    )
    result: list[tuple[str, str]] = []
    for raw in raw_tasks:
        key = normalize_task(raw)
        if key not in aliases:
            choices = ", ".join(TASK_FILES)
            raise ValueError(f"unknown MVBench task {raw!r}; choices: {choices}")
        item = aliases[key]
        if item not in result:
            result.append(item)
    return result


def annotation_root(dataset_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    candidate = dataset_root / "json"
    return candidate if candidate.is_dir() else dataset_root


def video_root(dataset_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    candidate = dataset_root / "video"
    return candidate if candidate.is_dir() else dataset_root


def build_video_index(root: Path) -> tuple[dict[str, list[Path]], dict[str, Path]]:
    by_basename: dict[str, list[Path]] = defaultdict(list)
    by_relative: dict[str, Path] = {}
    for directory, _, filenames in os.walk(root):
        parent = Path(directory)
        for filename in filenames:
            path = parent / filename
            if path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            by_basename[filename].append(path)
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:
                relative = path.as_posix()
            by_relative[relative] = path
    return dict(by_basename), by_relative


def resolve_video(
    raw_name: str,
    root: Path,
    by_basename: dict[str, list[Path]],
    by_relative: dict[str, Path],
) -> Path:
    normalized = raw_name.replace("\\", "/").lstrip("./")
    direct = root / normalized
    if direct.is_file():
        return direct
    if normalized in by_relative:
        return by_relative[normalized]
    matches = by_basename.get(Path(normalized).name, [])
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"video not found under {root}: {raw_name}")
    rendered = ", ".join(str(path) for path in matches[:5])
    raise ValueError(f"ambiguous video basename {raw_name!r}: {rendered}")


def answer_option(answer: str, candidates: Sequence[str]) -> str:
    answer = str(answer).strip()
    for index, candidate in enumerate(candidates):
        if answer == str(candidate).strip():
            return OPTION_NAMES[index]
    match = re.fullmatch(r"\(?\s*([A-Z])\s*\)?(?:\s+.*)?", answer.upper())
    if match and OPTION_NAMES.index(match.group(1)) < len(candidates):
        return match.group(1)
    raise ValueError(f"answer {answer!r} does not match any candidate")


def optional_float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    result = float(value)
    return result if result >= 0 else None


def load_cases(
    annotations: Path,
    videos: Path,
    tasks: Sequence[tuple[str, str]],
    limit: int | None,
    limit_per_task: int | None,
) -> list[Case]:
    by_basename, by_relative = build_video_index(videos)
    if not by_basename:
        raise ValueError(f"no video files found under {videos}")

    cases: list[Case] = []
    missing_annotations: list[Path] = []
    for task, filename in tasks:
        path = annotations / filename
        if not path.is_file():
            missing_annotations.append(path)
            continue
        with path.open(encoding="utf-8") as source:
            rows = json.load(source)
        if not isinstance(rows, list):
            raise ValueError(f"annotation must contain a JSON array: {path}")
        task_count = 0
        for row_index, row in enumerate(rows):
            if limit is not None and len(cases) >= limit:
                break
            if limit_per_task is not None and task_count >= limit_per_task:
                break
            if not isinstance(row, dict):
                raise ValueError(f"{path}[{row_index}] is not an object")
            raw_video = str(row.get("video") or "").strip()
            question = str(row.get("question") or "").strip()
            candidates_raw = row.get("candidates") or row.get("choices")
            answer = str(row.get("answer") or "").strip()
            if not raw_video or not question or not isinstance(candidates_raw, list) or not answer:
                raise ValueError(f"incomplete annotation at {path}[{row_index}]")
            candidates = tuple(str(value).strip() for value in candidates_raw)
            if not 2 <= len(candidates) <= len(OPTION_NAMES):
                raise ValueError(f"invalid candidates at {path}[{row_index}]")
            resolved = resolve_video(raw_video, videos, by_basename, by_relative)
            option = answer_option(answer, candidates)
            start = optional_float(row, "start")
            end = optional_float(row, "end")
            if start is not None and end is not None and end <= start:
                raise ValueError(f"invalid start/end at {path}[{row_index}]")
            case_id = f"{Path(filename).stem}:{row_index}:{raw_video}"
            cases.append(
                Case(
                    case_id=case_id,
                    task=task,
                    annotation_file=path,
                    row_index=row_index,
                    video_name=raw_video,
                    video_path=resolved,
                    question=question,
                    candidates=candidates,
                    answer_text=answer,
                    answer_option=option,
                    start=start,
                    end=end,
                )
            )
            task_count += 1
        if limit is not None and len(cases) >= limit:
            break

    if missing_annotations:
        rendered = "\n".join(f"  {path}" for path in missing_annotations)
        raise FileNotFoundError(f"missing MVBench annotation files:\n{rendered}")
    return cases


def build_prompt(case: Case) -> str:
    lines = [f"Question: {case.question}", "Options:"]
    lines.extend(
        f"({OPTION_NAMES[index]}) {candidate}"
        for index, candidate in enumerate(case.candidates)
    )
    lines.append("Only give the best option as one uppercase letter. Do not explain.")
    return "\n".join(lines)


def clip_video_all_frames(case: Case, ffmpeg_bin: str) -> tuple[bytes, str, bool]:
    if case.start is None or case.end is None:
        return case.video_path.read_bytes(), guess_media_type(case.video_path), False
    command = [
        ffmpeg_bin,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{case.start:.6f}",
        "-to",
        f"{case.end:.6f}",
        "-i",
        str(case.video_path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vsync",
        "0",
        "-c:v",
        "ffv1",
        "-f",
        "matroska",
        "pipe:1",
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"ffmpeg not found: {ffmpeg_bin}") from exc
    if result.returncode != 0:
        details = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg interval decode failed: {details}")
    if not result.stdout:
        raise RuntimeError("ffmpeg returned an empty bounded video")
    return result.stdout, "video/x-matroska", True


def video_payload(
    case: Case,
    ffmpeg_bin: str,
    ignore_bounds: bool,
    max_video_bytes: int,
) -> tuple[bytes, str, bool]:
    if ignore_bounds:
        data = case.video_path.read_bytes()
        media_type = guess_media_type(case.video_path)
        bounded = False
    else:
        data, media_type, bounded = clip_video_all_frames(case, ffmpeg_bin)
    if len(data) > max_video_bytes:
        raise ValueError(
            f"video payload is {len(data)} bytes, exceeding --max-video-bytes "
            f"{max_video_bytes}; it was not truncated"
        )
    return data, media_type, bounded


def response_text(response: dict[str, Any]) -> str:
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
    raise ValueError(f"unsupported API content type: {type(content).__name__}")


def infer_option(answer: str, candidate_count: int) -> str | None:
    valid = set(OPTION_NAMES[:candidate_count])
    upper = answer.strip().upper()
    if upper in valid:
        return upper
    for pattern in (
        r"(?:BEST\s+)?OPTION\s*(?:IS|:)?\s*\(?([A-Z])\)?",
        r"(?:CORRECT\s+)?ANSWER\s*(?:IS|:)?\s*\(?([A-Z])\)?",
        r"(?:答案|选项|选择)\s*(?:是|为|：|:)?\s*\(?([A-Z])\)?",
        r"^\s*\(?([A-Z])\)?(?:[.、:\s]|$)",
    ):
        match = re.search(pattern, upper)
        if match and match.group(1) in valid:
            return match.group(1)
    return None


def call_api(
    endpoint: str,
    case: Case,
    video: bytes,
    media_type: str,
    timeout: float,
    retries: int,
) -> tuple[dict[str, Any], float]:
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_base64",
                        "media_type": media_type,
                        "data": base64.b64encode(video).decode("ascii"),
                    },
                    {"type": "text", "text": build_prompt(case)},
                ],
            },
        ],
        "stream": False,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        http_request = request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        try:
            with request.urlopen(http_request, timeout=timeout) as response:
                parsed = json.loads(response.read().decode("utf-8"))
            return parsed, time.monotonic() - started
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")[:1000]
            last_error = RuntimeError(f"HTTP {exc.code}: {details}")
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
        if attempt < retries:
            wait_seconds = min(2**attempt, 8)
            print(f"  request failed, retrying in {wait_seconds}s: {last_error}")
            time.sleep(wait_seconds)
    assert last_error is not None
    raise last_error


def append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if isinstance(record, dict):
                records.append(record)
    return records


def completed_ids(records: Iterable[dict[str, Any]]) -> set[str]:
    return {
        str(record["id"])
        for record in records
        if record.get("status") == "ok" and record.get("id") is not None
    }


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def make_summary(cases: Sequence[Case], records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    selected_ids = {case.case_id for case in cases}
    latest: dict[str, dict[str, Any]] = {}
    errors: dict[str, dict[str, Any]] = {}
    for record in records:
        case_id = str(record.get("id") or "")
        if case_id not in selected_ids:
            continue
        if record.get("status") == "ok":
            latest[case_id] = record
            errors.pop(case_id, None)
        elif case_id not in latest:
            errors[case_id] = record
    completed = list(latest.values())
    correct = sum(bool(record.get("correct")) for record in completed)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in completed:
        by_task[str(record.get("task") or "unknown")].append(record)
    per_task = {
        task: {
            "completed": len(items),
            "correct": sum(bool(item.get("correct")) for item in items),
            "accuracy": mean([float(bool(item.get("correct"))) for item in items]),
        }
        for task, items in sorted(by_task.items())
    }
    metric_names = (
        "request_elapsed_s",
        "ttft_s",
        "total_s",
        "tokens_per_second",
        "decode_tokens_per_second",
    )
    averages = {
        name: mean(
            [float(record[name]) for record in completed if isinstance(record.get(name), (int, float))]
        )
        for name in metric_names
    }
    return {
        "selected": len(cases),
        "completed": len(completed),
        "correct": correct,
        "accuracy": correct / len(completed) if completed else None,
        "pending": len(cases) - len(completed),
        "latest_errors": len(errors),
        "averages": averages,
        "per_task": per_task,
    }


def write_summary(output: Path, summary: dict[str, Any]) -> Path:
    path = output.with_suffix(".summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as destination:
        json.dump(summary, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DIAL on MVBench with native full-frame video input."
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Extracted MVBench root")
    parser.add_argument("--annotations-dir", type=Path, help="Defaults to DATASET_ROOT/json")
    parser.add_argument("--videos-dir", type=Path, help="Defaults to DATASET_ROOT/video")
    parser.add_argument("--api", default="http://127.0.0.1:8082", help="DIAL API base URL")
    parser.add_argument("--output", type=Path, default=Path("eval/mvbench/results.jsonl"))
    parser.add_argument("--task", action="append", help="Task name or JSON stem; repeat as needed")
    parser.add_argument("--limit", type=int, help="Maximum total cases")
    parser.add_argument("--limit-per-task", type=int, help="Maximum cases per selected task")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--max-video-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--ffmpeg-bin", default=os.environ.get("DIAL_FFMPEG_BIN", "ffmpeg"))
    parser.add_argument(
        "--ignore-bounds",
        action="store_true",
        help="Send each full source video instead of its annotated start/end interval",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate files without API calls")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip completed JSONL rows")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    annotations = annotation_root(dataset_root, args.annotations_dir)
    videos = video_root(dataset_root, args.videos_dir)
    if not annotations.is_dir():
        raise FileNotFoundError(f"annotations directory not found: {annotations}")
    if not videos.is_dir():
        raise FileNotFoundError(f"videos directory not found: {videos}")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.limit_per_task is not None and args.limit_per_task < 1:
        raise ValueError("--limit-per-task must be positive")
    if args.max_video_bytes < 1:
        raise ValueError("--max-video-bytes must be positive")

    tasks = selected_tasks(args.task)
    cases = load_cases(annotations, videos, tasks, args.limit, args.limit_per_task)
    print(
        f"[MVBench] loaded {len(cases)} cases from {len(tasks)} task(s); "
        f"annotations={annotations} videos={videos}"
    )
    if args.dry_run:
        bounded = sum(case.start is not None and case.end is not None for case in cases)
        total_bytes = sum(case.video_path.stat().st_size for case in cases)
        print(
            f"[MVBench] dry-run ok: bounded={bounded} source_bytes={total_bytes} "
            f"unique_videos={len({case.video_path for case in cases})}"
        )
        return 0

    records = [] if args.no_resume else load_records(args.output)
    completed = set() if args.no_resume else completed_ids(records)
    endpoint = endpoint_from(args.api)
    pending = [case for case in cases if case.case_id not in completed]
    print(f"[MVBench] completed={len(completed)} pending={len(pending)} endpoint={endpoint}")

    for position, case in enumerate(pending, 1):
        print(
            f"[{position}/{len(pending)}] {case.task} {case.video_name} "
            f"gold={case.answer_option}"
        )
        base_record: dict[str, Any] = {
            "id": case.case_id,
            "task": case.task,
            "video": case.video_name,
            "video_path": str(case.video_path),
            "question": case.question,
            "candidates": list(case.candidates),
            "gold": case.answer_option,
            "answer_text": case.answer_text,
            "start": case.start,
            "end": case.end,
        }
        try:
            payload, media_type, bounded = video_payload(
                case,
                args.ffmpeg_bin,
                args.ignore_bounds,
                args.max_video_bytes,
            )
            response, elapsed = call_api(
                endpoint,
                case,
                payload,
                media_type,
                args.timeout,
                args.retries,
            )
            raw_answer = response_text(response)
            prediction = infer_option(raw_answer, len(case.candidates))
            record = {
                **base_record,
                "status": "ok",
                "bounded_interval": bounded,
                "payload_bytes": len(payload),
                "prediction": prediction,
                "raw_answer": raw_answer,
                "correct": prediction == case.answer_option,
                "request_elapsed_s": elapsed,
            }
            for metric in (
                "ttft_s",
                "total_s",
                "tokens_per_second",
                "decode_tokens_per_second",
                "generated_tokens",
                "distributed_overhead_s",
                "remote_compute_s",
                "remote_requests",
            ):
                if response.get(metric) is not None:
                    record[metric] = response[metric]
            print(
                f"  prediction={prediction or '?'} correct={record['correct']} "
                f"elapsed={elapsed:.2f}s answer={raw_answer[:120]!r}"
            )
        except Exception as exc:
            record = {**base_record, "status": "error", "error": str(exc)}
            print(f"  ERROR: {exc}", file=sys.stderr)
        append_record(args.output, record)
        records.append(record)
        summary = make_summary(cases, records)
        write_summary(args.output, summary)

    summary = make_summary(cases, records)
    summary_path = write_summary(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[MVBench] results={args.output} summary={summary_path}")
    return 0 if summary["completed"] == summary["selected"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
