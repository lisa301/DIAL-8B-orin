#!/usr/bin/env python3
"""Send one video to DIAL's native Video module.

The file is uploaded as-is. By default the server follows the Qwen3-VL video
processor (uniform 2 FPS sampling, 4..768 frames). The server keeps all frame
processing in memory and does not write image files. Pass --video-no-sample to
the server when an explicit full-frame run is required.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys
from typing import Any
from urllib import error, request


DEFAULT_API_BASE = "http://127.0.0.1:8082"
DEFAULT_MAX_BYTES = 256 * 1024 * 1024


def guess_media_type(path: Path) -> str | None:
    return {
        ".mp4": "video/mp4",
        ".m4v": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo",
    }.get(path.suffix.lower())


def endpoint_from_base(api_client: str) -> str:
    base = api_client.strip()
    if not base.startswith(("http://", "https://")):
        base = "http://" + base
    base = base.rstrip("/")
    suffix = "/api/v1/chat/completions"
    return base if base.endswith(suffix) else base + suffix


def build_payload(video: Path, question: str, max_bytes: int) -> dict[str, Any]:
    size = video.stat().st_size
    if size > max_bytes:
        raise ValueError(
            f"video is {size} bytes, exceeding --max-bytes {max_bytes}; "
            "the file will not be truncated"
        )
    encoded = base64.b64encode(video.read_bytes()).decode("ascii")
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_base64",
                        "media_type": guess_media_type(video),
                        "data": encoded,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ],
        "stream": False,
    }


def send_request(api_client: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    http_request = request.Request(
        endpoint_from_base(api_client),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API returned HTTP {exc.code}: {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"cannot connect to DIAL API: {exc.reason}") from exc


def response_text(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        rendered = json.dumps(response, ensure_ascii=False)
        raise RuntimeError(f"unexpected API response: {rendered}") from exc
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a video with DIAL's native Video module (official uniform sampling by default)."
    )
    parser.add_argument("--video", type=Path, required=True, help="Local video file")
    parser.add_argument("--ask", required=True, help="Question about the video")
    parser.add_argument("--api-client", default=DEFAULT_API_BASE, help="DIAL API base URL")
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help="Reject files larger than this; never truncate (default: 256 MiB)",
    )
    parser.add_argument("--timeout", type=float, default=3600.0, help="API timeout in seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"video not found: {video}")
    if args.max_bytes <= 0:
        raise ValueError("--max-bytes must be positive")

    payload = build_payload(video, args.ask, args.max_bytes)
    print(
        f"[video] uploading file: {video.name} ({video.stat().st_size} bytes); "
        "server uses Qwen3-VL official uniform sampling by default",
        file=sys.stderr,
    )
    response = send_request(args.api_client, payload, args.timeout)
    print(response_text(response))
    metrics = " ".join(
        f"{name}={response[name]}"
        for name in ("ttft_s", "total_s", "tokens_per_second", "generated_tokens")
        if response.get(name) is not None
    )
    if metrics:
        print(f"[metrics] {metrics}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
