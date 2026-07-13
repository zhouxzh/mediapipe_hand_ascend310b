#!/usr/bin/env python3
"""Batch-generate legacy MediaPipe graph annotations for PianoVAM videos."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/PianoVAM_v1")
    parser.add_argument("--split", default="test", help="Metadata split to annotate, or 'all'.")
    parser.add_argument("--record-time", action="append", default=[], help="Recording id. Can be repeated.")
    parser.add_argument("--output-root", default="", help="Default: <data-root>/mediapipe_legacy_annotations")
    parser.add_argument("--model-complexity", type=int, choices=[0, 1], default=1)
    parser.add_argument("--max-num-hands", type=int, default=2)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--running-mode", choices=["both", "image", "tracking"], default="both")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--save-vis", type=int, default=0)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_metadata(data_root: Path) -> list[dict[str, Any]]:
    metadata = json.loads((data_root / "metadata.json").read_text(encoding="utf-8-sig"))
    records: list[dict[str, Any]] = []
    for key, item in metadata.items():
        record = dict(item)
        record["metadata_id"] = str(key)
        records.append(record)
    records.sort(key=lambda item: str(item.get("record_time", "")))
    return records


def select_records(args: argparse.Namespace, data_root: Path) -> list[dict[str, Any]]:
    records = load_metadata(data_root)
    if args.record_time:
        wanted = set(args.record_time)
        records = [item for item in records if str(item.get("record_time")) in wanted]
    elif args.split != "all":
        records = [item for item in records if str(item.get("split")) == args.split]
    if args.max_videos:
        records = records[: args.max_videos]
    return records


def selected_modes(running_mode: str) -> list[str]:
    if running_mode == "both":
        return ["image", "tracking"]
    return [running_mode]


def default_output_root(data_root: Path, running_mode: str) -> Path:
    if running_mode == "both":
        return data_root / "mediapipe_legacy_annotations"
    return data_root / f"mediapipe_legacy_cpu_{running_mode}_annotations"


def split_stream_annotations(output_dir: Path, modes: list[str]) -> None:
    annotations_path = output_dir / "mediapipe_annotations.json"
    if not annotations_path.exists():
        return
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    frames = annotations.get("frames", [])
    summary = annotations.get("summary", {})
    schema = annotations.get("schema", {})
    for mode in modes:
        mode_summary = dict(summary)
        mode_summary["reference_stream"] = mode
        mode_summary["annotations_json"] = str(output_dir / f"{mode}_mediapipe_annotations.json")
        mode_annotations = {
            "summary": mode_summary,
            "schema": {
                "reference_stream": mode,
                mode: schema.get(mode, ""),
                "palm_detections": schema.get("palm_detections", ""),
                "hands": schema.get("hands", ""),
            },
            "frames": [
                {
                    "frame_index": frame["frame_index"],
                    "timestamp_sec": frame.get("timestamp_sec"),
                    "width": frame.get("width"),
                    "height": frame.get("height"),
                    mode: frame.get(mode, {}),
                }
                for frame in frames
            ],
        }
        (output_dir / f"{mode}_mediapipe_annotations.json").write_text(
            json.dumps(mode_annotations, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def annotation_command(args: argparse.Namespace, video_path: Path, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "annotate_mediapipe_video.py"),
        "--video",
        str(video_path),
        "--output-dir",
        str(output_dir),
        "--model-complexity",
        str(args.model_complexity),
        "--max-num-hands",
        str(args.max_num_hands),
        "--min-detection-confidence",
        str(args.min_detection_confidence),
        "--min-tracking-confidence",
        str(args.min_tracking_confidence),
        "--running-mode",
        args.running_mode,
        "--frame-stride",
        str(args.frame_stride),
        "--start-frame",
        str(args.start_frame),
        "--save-vis",
        str(args.save_vis),
    ]
    if args.max_frames:
        cmd.extend(["--max-frames", str(args.max_frames)])
    if args.save_video:
        cmd.append("--save-video")
    if args.force:
        cmd.append("--force")
    return cmd


def main() -> int:
    args = parse_args()
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")

    data_root = resolve_path(args.data_root)
    output_root = resolve_path(args.output_root) if args.output_root else default_output_root(data_root, args.running_mode)
    output_root.mkdir(parents=True, exist_ok=True)
    modes = selected_modes(args.running_mode)
    records = select_records(args, data_root)
    if not records:
        raise ValueError(f"No PianoVAM records selected from {data_root}")

    results: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        record_time = str(record["record_time"])
        video_path = data_root / "Video" / f"{record_time}.mp4"
        output_dir = output_root / record_time
        annotations_path = output_dir / "mediapipe_annotations.json"
        print(f"[pianovam-mediapipe] {index}/{len(records)} {record_time}", flush=True)
        if not video_path.exists():
            results.append({"record_time": record_time, "status": "missing_video", "video": str(video_path)})
            print(f"  missing video: {video_path}", flush=True)
            continue
        if annotations_path.exists() and not args.force:
            split_stream_annotations(output_dir, modes)
            results.append(
                {
                    "record_time": record_time,
                    "status": "skipped_existing",
                    "annotations": str(annotations_path),
                    "output_dir": str(output_dir),
                }
            )
            print(f"  skipped existing: {annotations_path}", flush=True)
            continue

        cmd = annotation_command(args, video_path, output_dir)
        if args.dry_run:
            results.append({"record_time": record_time, "status": "dry_run", "command": cmd})
            print("  " + " ".join(cmd), flush=True)
            continue

        start = time.perf_counter()
        completed = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
        elapsed = time.perf_counter() - start
        if completed.returncode == 0:
            split_stream_annotations(output_dir, modes)
        results.append(
            {
                "record_time": record_time,
                "status": "ok" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
                "elapsed_sec": elapsed,
                "video": str(video_path),
                "output_dir": str(output_dir),
                "annotations": str(annotations_path),
            }
        )

    summary = {
        "task": "annotate_pianovam_mediapipe_legacy",
        "data_root": str(data_root),
        "output_root": str(output_root),
        "split": args.split,
        "running_mode": args.running_mode,
        "streams": modes,
        "records": len(records),
        "ok": sum(1 for item in results if item["status"] in {"ok", "skipped_existing"}),
        "failed": sum(1 for item in results if item["status"] == "failed"),
        "missing_video": sum(1 for item in results if item["status"] == "missing_video"),
        "results": results,
    }
    summary_path = output_root / "annotation_batch_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed"] == 0 and summary["missing_video"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
