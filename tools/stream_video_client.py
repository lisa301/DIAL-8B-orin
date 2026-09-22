#!/usr/bin/env python3
"""Legacy live-stream monitor using periodic frame extraction.

For local video analysis without frame sampling, use video_inference_client.py
or spm-cli --video. This script remains for unbounded RTSP/HTTP live streams.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover - exercised only on minimal deployments
    Image = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]


def find_ffmpeg(user_path: Optional[str]) -> str:
    if user_path:
        p = Path(user_path)
        if not p.exists():
            raise FileNotFoundError(p)
        return str(p)
    for cand in ("ffmpeg", "/opt/sophon/sophon-ffmpeg-latest/bin/ffmpeg"):
        hit = shutil.which(cand) if cand == "ffmpeg" else (cand if Path(cand).exists() else None)
        if hit:
            return str(hit)
    raise FileNotFoundError("ffmpeg not found, set --ffmpeg-bin")


def find_ffprobe(ffmpeg_bin: str) -> Optional[str]:
    # Prefer ffprobe located next to ffmpeg, then fallback to PATH.
    p = Path(ffmpeg_bin)
    sibling = p.with_name("ffprobe")
    if sibling.exists():
        return str(sibling)
    hit = shutil.which("ffprobe")
    return str(hit) if hit else None


def probe_video_source(ffprobe_bin: str, source: str, rtsp_transport: str) -> None:
    cmd = [
        ffprobe_bin,
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if source.lower().startswith("rtsp://"):
        cmd.extend(["-rtsp_transport", rtsp_transport])
    cmd.extend(
        [
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,avg_frame_rate,bit_rate",
            "-of",
            "default=noprint_wrappers=1",
            source,
        ]
    )
    try:
        out = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=8)
    except Exception as e:
        print(f"[WARN] ffprobe failed: {e}")
        return
    if out.returncode != 0:
        err = (out.stderr or "").strip()
        print(f"[WARN] ffprobe exit={out.returncode} {err}")
        return
    info = (out.stdout or "").strip()
    if info:
        print("source_probe:")
        for line in info.splitlines():
            print(f"  {line}")


def is_jpeg_complete(path: Path) -> bool:
    # A complete JPEG starts with FFD8 and ends with FFD9.
    try:
        size = path.stat().st_size
        if size < 4:
            return False
        with path.open("rb") as f:
            head = f.read(2)
            f.seek(-2, 2)
            tail = f.read(2)
        return head == b"\xff\xd8" and tail == b"\xff\xd9"
    except OSError:
        return False


def is_png_complete(path: Path) -> bool:
    # A complete PNG has fixed 8-byte signature and ends with IEND chunk marker.
    try:
        size = path.stat().st_size
        if size < 32:
            return False
        with path.open("rb") as f:
            head = f.read(8)
            f.seek(-12, 2)
            tail = f.read(12)
        return head == b"\x89PNG\r\n\x1a\n" and tail == b"\x00\x00\x00\x00IEND\xaeB`\x82"
    except OSError:
        return False


def is_image_complete(path: Path, frame_format: str) -> bool:
    fmt = frame_format.lower()
    if fmt in ("jpg", "jpeg"):
        return is_jpeg_complete(path)
    if fmt == "png":
        return is_png_complete(path)
    return False


def probe_image_file(ffprobe_bin: Optional[str], path: Path) -> None:
    if not ffprobe_bin:
        return
    cmd = [
        ffprobe_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt",
        "-of",
        "default=noprint_wrappers=1",
        str(path),
    ]
    out = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if out.returncode != 0:
        return
    info = (out.stdout or "").strip()
    if info:
        one_line = ", ".join(line.strip() for line in info.splitlines() if line.strip())
        print(f"[frame_probe] {path.name}: {one_line}")


def sanitize_model_output(text: str) -> str:
    # Drop noisy tool/runtime lines and keep only user-facing model content.
    ansi_re = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")
    out_lines: List[str] = []
    for raw in text.splitlines():
        line = ansi_re.sub("", raw).rstrip()
        normalized = line.lstrip()
        lowered = normalized.lower()
        if not line:
            out_lines.append("")
            continue
        if normalized.startswith("[choose_model]:"):
            continue
        if normalized.startswith("[metrics]"):
            continue
        if lowered.startswith("performance:"):
            continue
        if normalized.startswith("性能指标"):
            continue
        if "ttft_s=" in lowered and ("tps=" in lowered or "total_s=" in lowered):
            continue
        if normalized.startswith("ffmpeg:"):
            continue
        if normalized.startswith("work_dir:"):
            continue
        if normalized.startswith("source_probe:"):
            continue
        if normalized.startswith("[frame_file]"):
            continue
        if normalized.startswith("[frame_probe]"):
            continue
        if normalized.startswith("[swscaler @"):
            continue
        if normalized.startswith("press Ctrl+C to stop"):
            continue
        out_lines.append(line)

    # Collapse excessive blank lines.
    compact: List[str] = []
    blank = False
    for line in out_lines:
        if line == "":
            if blank:
                continue
            blank = True
            compact.append(line)
        else:
            blank = False
            compact.append(line)
    return "\n".join(compact).strip()


def extract_last_metrics(text: str) -> str:
    # Pick last metrics line from spm-cli output and return payload only.
    ansi_re = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")
    last = ""
    for raw in text.splitlines():
        line = ansi_re.sub("", raw).strip()
        lowered = line.lower()
        payload = ""
        if line.startswith("[metrics]"):
            payload = line[len("[metrics]") :].strip()
        elif lowered.startswith("performance:"):
            payload = line.split(":", 1)[1].strip()
        elif line.startswith("性能指标"):
            payload = line.split(":", 1)[1].strip() if ":" in line else line
        elif "ttft_s=" in lowered and ("tps=" in lowered or "total_s=" in lowered):
            # Fallback: tolerate custom wrappers/prefixes as long as payload looks metric-like.
            payload = line
        if payload:
            last = payload
    return last


def wait_frame_ready(
    path: Path,
    frame_format: str,
    timeout_sec: float,
    poll_sec: float,
    min_bytes: int,
    stable_checks: int = 2,
) -> bool:
    deadline = time.time() + max(0.1, timeout_sec)
    last_size = -1
    stable = 0
    while time.time() < deadline:
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        complete = size >= min_bytes and is_image_complete(path, frame_format)
        if complete and size == last_size:
            stable += 1
            if stable >= stable_checks:
                return True
        else:
            stable = 0
        last_size = size
        time.sleep(max(0.02, poll_sec))
    return False


FrameSignature = Tuple[int, int, bytes]


def frame_signature(path: Path, sample_size: int) -> FrameSignature:
    """Build a cheap visual signature for nearby video frames.

    The signature deliberately uses a small grayscale image. It is only used to
    decide whether a new model request is necessary; the original frame is
    still sent whenever the similarity gate is not met.
    """
    if sample_size < 2:
        raise ValueError("frame signature size must be >= 2")
    if Image is None or ImageOps is None:
        raise RuntimeError("Pillow is not installed; frame reuse is unavailable")
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("L")
        width, height = image.size
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        sampled = image.resize((sample_size, sample_size), resampling)
        return width, height, sampled.tobytes()


def frame_similarity(previous: FrameSignature, current: FrameSignature) -> float:
    """Return a [0, 1] similarity score based on normalized grayscale MAD."""
    previous_width, previous_height, previous_pixels = previous
    current_width, current_height, current_pixels = current
    if (previous_width, previous_height) != (current_width, current_height):
        return 0.0
    if len(previous_pixels) != len(current_pixels) or not previous_pixels:
        return 0.0
    difference = sum(abs(a - b) for a, b in zip(previous_pixels, current_pixels))
    mean_absolute_difference = difference / (255.0 * len(previous_pixels))
    return max(0.0, min(1.0, 1.0 - mean_absolute_difference))


def run_one(
    spm_cli: Path,
    api_client: str,
    prompt: str,
    image_path: Path,
    stream: bool,
    metrics: bool,
    verbose: bool,
) -> Tuple[int, str, str]:
    cmd = [
        str(spm_cli),
        "--api-client",
        api_client,
        "--image",
        str(image_path),
        "--ask",
        prompt,
    ]
    # spm-cli uses boolean flags; do not pass string values like "true"/"false".
    if not stream:
        cmd.append("--no-stream")
    if metrics:
        cmd.append("--metrics")
    print(f"\n[{time.strftime('%F %T')}] frame: {image_path.name}")
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, errors="replace")
    merged = (proc.stdout or "")
    if proc.stderr:
        if merged and not merged.endswith("\n"):
            merged += "\n"
        merged += proc.stderr
    metrics_line = extract_last_metrics(merged) if metrics else ""
    cleaned = sanitize_model_output(merged)
    if cleaned:
        print(cleaned)
    elif verbose and merged.strip():
        print(merged.strip())
    if metrics and metrics_line:
        print(f"Performance: {metrics_line}")
    return proc.returncode, cleaned, metrics_line


def build_ffmpeg_cmd(
    ffmpeg_bin: str,
    source: str,
    interval_sec: float,
    out_pattern: str,
    rtsp_transport: str,
    frame_format: str,
    jpeg_q: int,
) -> List[str]:
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if source.lower().startswith("rtsp://"):
        cmd.extend(["-rtsp_transport", rtsp_transport])
    cmd.extend(
        [
            "-i",
            source,
            "-vf",
            f"fps=1/{interval_sec}",
            "-vsync",
            "vfr",
        ]
    )
    fmt = frame_format.lower()
    if fmt in ("jpg", "jpeg"):
        cmd.extend(["-q:v", str(jpeg_q)])
    elif fmt == "png":
        # Keep png encoding fast; source quality is still the hard upper bound.
        cmd.extend(["-compression_level", "1"])
    else:
        raise ValueError(f"unsupported frame format: {frame_format}")
    cmd.append(out_pattern)
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Video source: rtsp/http/local-file")
    ap.add_argument("--interval-sec", type=float, default=10.0, help="Frame sampling interval in seconds")
    ap.add_argument("--spm-cli", default="./target/release/spm-cli", help="Path to spm-cli")
    ap.add_argument("--api-client", required=True, help="spm api base URL, e.g. http://127.0.0.1:8082")
    ap.add_argument("--prompt", default="请描述当前画面。", help="Prompt for each frame")
    ap.add_argument("--stream", action="store_true", default=True, help="Enable streaming response")
    ap.add_argument("--no-stream", dest="stream", action="store_false", help="Disable streaming response")
    ap.add_argument(
        "--metrics",
        dest="metrics",
        action="store_true",
        default=True,
        help="Print per-frame metrics right after each frame result (default: on)",
    )
    ap.add_argument(
        "--no-metrics",
        dest="metrics",
        action="store_false",
        help="Disable per-frame metrics output",
    )
    ap.add_argument("--max-frames", type=int, default=0, help="Stop after N frames, 0 means unlimited")
    default_work_dir = str((Path(__file__).resolve().parents[1] / "pictures"))
    ap.add_argument(
        "--work-dir",
        default=default_work_dir,
        help="Directory for extracted frames",
    )
    ap.add_argument(
        "--keep-frames",
        dest="keep_frames",
        action="store_true",
        default=True,
        help="Keep extracted frame images (default: keep)",
    )
    ap.add_argument(
        "--no-keep-frames",
        dest="keep_frames",
        action="store_false",
        help="Delete extracted frame images after inference",
    )
    ap.add_argument("--poll-sec", type=float, default=0.5, help="Polling interval for new frame files")
    ap.add_argument("--ffmpeg-bin", default=None, help="Path to ffmpeg binary")
    ap.add_argument(
        "--frame-format",
        default="jpg",
        choices=["jpg", "png"],
        help="Extracted frame image format",
    )
    ap.add_argument(
        "--jpeg-q",
        type=int,
        default=2,
        help="JPEG quality scale used by ffmpeg when --frame-format=jpg (smaller is better, 1..31)",
    )
    ap.add_argument(
        "--skip-source-probe",
        action="store_true",
        help="Skip ffprobe of source stream metadata",
    )
    ap.add_argument(
        "--frame-ready-timeout-sec",
        type=float,
        default=3.0,
        help="Max wait time for a frame file to become complete/stable",
    )
    ap.add_argument(
        "--frame-ready-poll-sec",
        type=float,
        default=0.1,
        help="Polling interval while waiting for frame file to be stable",
    )
    ap.add_argument(
        "--min-frame-bytes",
        type=int,
        default=4096,
        help="Minimum frame file size in bytes before sending to model",
    )
    ap.add_argument(
        "--frame-similarity-threshold",
        type=float,
        default=0.985,
        help="Reuse the previous answer when frame similarity is at least this value (0 disables)",
    )
    ap.add_argument(
        "--frame-signature-size",
        type=int,
        default=32,
        help="Low-resolution grayscale side length used for frame comparison",
    )
    ap.add_argument(
        "--frame-reuse-max-skips",
        type=int,
        default=30,
        help="Force a fresh inference after this many reused frames; 0 means unlimited",
    )
    ap.add_argument(
        "--no-frame-reuse",
        dest="frame_reuse",
        action="store_false",
        default=True,
        help="Disable visual similarity based frame reuse",
    )
    ap.add_argument(
        "--rtsp-transport",
        default="tcp",
        choices=["tcp", "udp"],
        help="RTSP transport for rtsp:// source",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logs (ffmpeg command, frame size/probe, diagnostics)",
    )
    args = ap.parse_args()

    if args.interval_sec <= 0:
        raise ValueError("--interval-sec must be > 0")
    if args.poll_sec <= 0:
        raise ValueError("--poll-sec must be > 0")
    if args.jpeg_q < 1 or args.jpeg_q > 31:
        raise ValueError("--jpeg-q must be in [1,31]")
    if not 0.0 <= args.frame_similarity_threshold <= 1.0:
        raise ValueError("--frame-similarity-threshold must be in [0, 1]")
    if args.frame_signature_size < 2:
        raise ValueError("--frame-signature-size must be >= 2")
    if args.frame_reuse_max_skips < 0:
        raise ValueError("--frame-reuse-max-skips must be >= 0")
    if args.frame_reuse and Image is None:
        print("[WARN] Pillow is not installed; disabling frame similarity reuse")
        args.frame_reuse = False

    spm_cli = Path(args.spm_cli)
    if not spm_cli.exists():
        raise FileNotFoundError(spm_cli)

    ffmpeg_bin = find_ffmpeg(args.ffmpeg_bin)
    ffprobe_bin = find_ffprobe(ffmpeg_bin)

    run_dir = Path(args.work_dir) / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    frame_suffix = "jpg" if args.frame_format.lower() == "jpg" else "png"
    out_pattern = str(run_dir / f"frame_%010d.{frame_suffix}")

    ffmpeg_cmd = build_ffmpeg_cmd(
        ffmpeg_bin=ffmpeg_bin,
        source=args.source,
        interval_sec=args.interval_sec,
        out_pattern=out_pattern,
        rtsp_transport=args.rtsp_transport,
        frame_format=args.frame_format,
        jpeg_q=args.jpeg_q,
    )
    if args.verbose:
        print("ffmpeg:", " ".join(ffmpeg_cmd))
        print(f"work_dir: {run_dir}")
        if not args.skip_source_probe:
            if ffprobe_bin:
                probe_video_source(ffprobe_bin, args.source, args.rtsp_transport)
            else:
                print("[WARN] ffprobe not found; skip source metadata probe")
        print("press Ctrl+C to stop")

    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd)
    seen: Set[str] = set()
    processed = 0
    last_inferred_signature: Optional[FrameSignature] = None
    last_inferred_frame = ""
    last_response = ""
    reused_since_inference = 0

    def stop_ffmpeg() -> None:
        if ffmpeg_proc.poll() is None:
            ffmpeg_proc.terminate()
            try:
                ffmpeg_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ffmpeg_proc.kill()
                ffmpeg_proc.wait(timeout=5)

    def process_frame(frame: Path) -> Optional[int]:
        """Process one completed frame and return its process code.

        A None result means the frame was not complete yet and should remain
        unmarked so the polling loop can retry it.
        """
        nonlocal processed
        nonlocal last_inferred_signature, last_inferred_frame
        nonlocal last_response, reused_since_inference

        ok = wait_frame_ready(
            frame,
            frame_format=args.frame_format,
            timeout_sec=args.frame_ready_timeout_sec,
            poll_sec=args.frame_ready_poll_sec,
            min_bytes=max(1, args.min_frame_bytes),
        )
        if not ok:
            print(f"[WARN] frame not ready yet, retry later: {frame.name}")
            return None

        seen.add(str(frame))
        if args.verbose:
            try:
                fsz = frame.stat().st_size
                print(f"[frame_file] {frame.name} size={fsz} bytes")
            except OSError:
                pass
            probe_image_file(ffprobe_bin, frame)

        current_signature: Optional[FrameSignature]
        try:
            current_signature = frame_signature(frame, args.frame_signature_size)
        except Exception as exc:
            # A signature failure must not drop an actual inference request.
            print(f"[WARN] frame similarity check failed for {frame.name}: {exc}")
            current_signature = None

        similarity: Optional[float] = None
        can_reuse = False
        if (
            args.frame_reuse
            and args.frame_similarity_threshold > 0.0
            and current_signature is not None
            and last_inferred_signature is not None
            and last_response
        ):
            similarity = frame_similarity(last_inferred_signature, current_signature)
            skip_limit_reached = (
                args.frame_reuse_max_skips > 0
                and reused_since_inference >= args.frame_reuse_max_skips
            )
            can_reuse = (
                similarity >= args.frame_similarity_threshold and not skip_limit_reached
            )

        if can_reuse:
            processed += 1
            reused_since_inference += 1
            print(f"\n[{time.strftime('%F %T')}] frame: {frame.name}")
            print(
                "[frame_reuse] reused answer from "
                f"{last_inferred_frame} similarity={similarity:.5f} "
                f"threshold={args.frame_similarity_threshold:.5f} "
                f"skips={reused_since_inference}"
            )
            print(last_response)
            if args.metrics:
                print(
                    "Performance: frame_reused=1 "
                    f"similarity={similarity:.5f} source_frame={last_inferred_frame}"
                )
            if not args.keep_frames:
                try:
                    frame.unlink()
                except OSError:
                    pass
            return 0

        code, response, _metrics_line = run_one(
            spm_cli=spm_cli,
            api_client=args.api_client,
            prompt=args.prompt,
            image_path=frame,
            stream=args.stream,
            metrics=args.metrics,
            verbose=args.verbose,
        )
        processed += 1
        if code != 0:
            print(f"[WARN] spm-cli exit code: {code}")
        elif current_signature is not None:
            # Compare future frames with the last frame that actually reached
            # the model, rather than chaining small differences indefinitely.
            last_inferred_signature = current_signature
            last_inferred_frame = frame.name
            last_response = response
            reused_since_inference = 0
        if not args.keep_frames:
            try:
                frame.unlink()
            except OSError:
                pass
        return code

    try:
        while True:
            frames = sorted(run_dir.glob(f"frame_*.{frame_suffix}"))
            new_frames = [p for p in frames if str(p) not in seen]

            for frame in new_frames:
                code = process_frame(frame)
                if code is None:
                    continue
                if args.max_frames > 0 and processed >= args.max_frames:
                    print(f"processed {processed} frame(s), exit")
                    stop_ffmpeg()
                    return 0

            ret = ffmpeg_proc.poll()
            if ret is not None:
                # Input ended or ffmpeg exited; drain remaining frames then exit.
                remaining = sorted(
                    [p for p in run_dir.glob(f"frame_*.{frame_suffix}") if str(p) not in seen]
                )
                for frame in remaining:
                    code = process_frame(frame)
                    if code is None:
                        continue
                if ret != 0:
                    print(f"[WARN] ffmpeg exited with code {ret}")
                print(f"ffmpeg ended, processed {processed} frame(s)")
                return 0

            time.sleep(args.poll_sec)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        stop_ffmpeg()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
