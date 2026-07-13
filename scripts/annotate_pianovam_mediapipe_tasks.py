#!/usr/bin/env python3
"""Annotate PianoVAM videos with MediaPipe Tasks HandLandmarker.

This is intended for GPU image/video references. MediaPipe Tasks exposes hand
landmarks, world landmarks, and handedness. It does not expose raw palm detector
boxes/keypoints, so this script records only the hand landmarks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/PianoVAM_v1")
    parser.add_argument("--split", default="test", help="Metadata split to annotate, or 'all'.")
    parser.add_argument("--record-time", action="append", default=[], help="Recording id. Can be repeated.")
    parser.add_argument("--output-root", default="", help="Default: <data-root>/mediapipe_tasks_gpu_tracking_annotations")
    parser.add_argument("--model", default="models/mediapipe/hand_landmarker.task")
    parser.add_argument("--delegate", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--running-mode", choices=["video", "image"], default="video")
    parser.add_argument("--num-hands", type=int, default=2)
    parser.add_argument("--min-hand-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-hand-presence-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
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


def scalar_summary(values: list[float], prefix: str) -> dict[str, float]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {
            f"{prefix}_mean_ms": math.nan,
            f"{prefix}_median_ms": math.nan,
            f"{prefix}_p95_ms": math.nan,
        }
    arr = np.asarray(clean, dtype=np.float64)
    return {
        f"{prefix}_mean_ms": float(np.mean(arr)),
        f"{prefix}_median_ms": float(np.median(arr)),
        f"{prefix}_p95_ms": float(np.percentile(arr, 95)),
    }


def hand_bbox_from_landmarks(hand21_px: np.ndarray, width: int, height: int) -> list[float]:
    xy = np.asarray(hand21_px, dtype=np.float32)[:, :2]
    return [
        float(np.clip(np.nanmin(xy[:, 0]), 0.0, float(width))),
        float(np.clip(np.nanmin(xy[:, 1]), 0.0, float(height))),
        float(np.clip(np.nanmax(xy[:, 0]), 0.0, float(width))),
        float(np.clip(np.nanmax(xy[:, 1]), 0.0, float(height))),
    ]


def normalized_landmarks_to_pixel_array(landmarks: Any, width: int, height: int) -> np.ndarray:
    points = []
    for point in landmarks:
        points.append([float(point.x) * float(width), float(point.y) * float(height), float(point.z) * float(width)])
    return np.asarray(points, dtype=np.float32)


def normalized_landmarks_to_list(landmarks: Any) -> list[list[float]]:
    return [[float(point.x), float(point.y), float(point.z)] for point in landmarks]


def world_landmarks_to_list(landmarks: Any | None) -> list[list[float]]:
    if landmarks is None:
        return []
    return [[float(point.x), float(point.y), float(point.z)] for point in landmarks]


def handedness_to_dict(classifications: Any | None) -> dict[str, Any]:
    if not classifications:
        return {"label": "", "score": math.nan, "index": -1}
    item = classifications[0]
    label = (
        getattr(item, "category_name", "")
        or getattr(item, "display_name", "")
        or getattr(item, "label", "")
        or ""
    )
    return {
        "label": str(label),
        "score": float(getattr(item, "score", math.nan)),
        "index": int(getattr(item, "index", -1)),
    }


def stream_name_from_running_mode(running_mode: str) -> str:
    return "tracking" if running_mode == "video" else "image"


def default_output_root(data_root: Path, args: argparse.Namespace) -> Path:
    if args.delegate == "gpu" and args.running_mode == "video":
        return data_root / "mediapipe_tasks_gpu_tracking_annotations"
    return data_root / f"mediapipe_tasks_{args.delegate}_{args.running_mode}_annotations"


def tasks_result_to_stream(result: Any, width: int, height: int, stream_name: str) -> dict[str, Any]:
    hand_landmarks = list(getattr(result, "hand_landmarks", []) or [])
    world_landmarks = list(getattr(result, "hand_world_landmarks", []) or [])
    handedness = list(getattr(result, "handedness", []) or [])
    hands: list[dict[str, Any]] = []
    for hand_index, landmarks in enumerate(hand_landmarks):
        hand21_px = normalized_landmarks_to_pixel_array(landmarks, width, height)
        handness = handedness_to_dict(handedness[hand_index] if hand_index < len(handedness) else None)
        hands.append(
            {
                "hand_index": int(hand_index),
                "source": f"mediapipe_tasks_{stream_name}",
                "handedness": handness["label"],
                "handedness_score": handness["score"],
                "handedness_index": handness["index"],
                "palm_detection_index": -1,
                "palm_detection": None,
                "palm_source": "not_exposed_by_mediapipe_tasks",
                "palm_bbox_xyxy_px": None,
                "palm_bbox_xyxy_norm": None,
                "palm7_keypoints_px": None,
                "palm7_keypoints_norm": None,
                "hand_bbox_xyxy_px": hand_bbox_from_landmarks(hand21_px, width, height),
                "hand21_keypoints_px": hand21_px.astype(float).tolist(),
                "hand21_keypoints_norm": normalized_landmarks_to_list(landmarks),
                "world21_keypoints_m": world_landmarks_to_list(
                    world_landmarks[hand_index] if hand_index < len(world_landmarks) else None
                ),
            }
        )
    return {
        "palm_detections": [],
        "hand_rects_from_palm_detections": [],
        "hands": hands,
        "hand_rects_from_landmarks": [],
    }


HAND_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


def draw_tracking(frame_bgr: np.ndarray, hands: list[dict[str, Any]]) -> np.ndarray:
    canvas = frame_bgr.copy()
    for hand in hands:
        points = np.asarray(hand["hand21_keypoints_px"], dtype=np.float32)
        color = (40, 220, 120) if int(hand["hand_index"]) == 0 else (255, 170, 40)
        for start, end in HAND_EDGES:
            a = tuple(np.round(points[start, :2]).astype(int))
            b = tuple(np.round(points[end, :2]).astype(int))
            cv2.line(canvas, a, b, color, 2)
        for point in points:
            xy = tuple(np.round(point[:2]).astype(int))
            cv2.circle(canvas, xy, 3, (245, 245, 245), -1)
            cv2.circle(canvas, xy, 3, color, 1)
        box = np.asarray(hand["hand_bbox_xyxy_px"], dtype=np.float32)
        x1, y1 = [int(round(float(v))) for v in box[:2]]
        label = f"trk{int(hand['hand_index']) + 1}"
        if hand.get("handedness"):
            label += f" {hand['handedness']}"
        cv2.putText(canvas, label, (max(x1, 8), max(y1 - 8, 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return canvas


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def create_landmarker(args: argparse.Namespace, model_path: Path) -> Any:
    import mediapipe as mp

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    delegate = BaseOptions.Delegate.GPU if args.delegate == "gpu" else BaseOptions.Delegate.CPU
    mode = VisionRunningMode.VIDEO if args.running_mode == "video" else VisionRunningMode.IMAGE
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path), delegate=delegate),
        running_mode=mode,
        num_hands=args.num_hands,
        min_hand_detection_confidence=args.min_hand_detection_confidence,
        min_hand_presence_confidence=args.min_hand_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    return HandLandmarker.create_from_options(options)


def annotate_record(args: argparse.Namespace, record: dict[str, Any], data_root: Path, output_root: Path) -> dict[str, Any]:
    import mediapipe as mp

    record_time = str(record["record_time"])
    video_path = data_root / "Video" / f"{record_time}.mp4"
    output_dir = output_root / record_time
    annotations_path = output_dir / "mediapipe_annotations.json"
    if not video_path.exists():
        return {"record_time": record_time, "status": "missing_video", "video": str(video_path)}
    if annotations_path.exists() and not args.force:
        return {"record_time": record_time, "status": "skipped_existing", "annotations": str(annotations_path)}
    if args.dry_run:
        return {"record_time": record_time, "status": "dry_run", "video": str(video_path), "output_dir": str(output_dir)}

    if output_dir.exists() and args.force:
        for child in output_dir.iterdir():
            if child.is_dir():
                import shutil

                shutil.rmtree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "vis"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"record_time": record_time, "status": "failed_open_video", "video": str(video_path)}
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer: cv2.VideoWriter | None = None
    annotated_video_path = ""
    if args.save_video:
        annotated_video_path = str(output_dir / "annotated_mediapipe_tasks_tracking.mp4")
        writer = cv2.VideoWriter(annotated_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps or 25.0, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create annotated video: {annotated_video_path}")

    frames: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    hand_rows: list[dict[str, Any]] = []
    stream_name = stream_name_from_running_mode(args.running_mode)
    stream_ms: list[float] = []
    saved_vis = 0
    processed_frames = 0
    start_wall = time.perf_counter()
    model_path = resolve_path(args.model)

    with create_landmarker(args, model_path) as landmarker:
        try:
            frame_index = -1
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame_index += 1
                if frame_index < args.start_frame:
                    continue
                if (frame_index - args.start_frame) % args.frame_stride != 0:
                    continue
                if args.max_frames and processed_frames >= args.max_frames:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                t0 = time.perf_counter()
                if args.running_mode == "video":
                    timestamp_ms = int(round((frame_index / fps) * 1000.0)) if fps > 0 else processed_frames * 33
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                else:
                    result = landmarker.detect(mp_image)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                stream_ms.append(elapsed_ms)
                stream = tasks_result_to_stream(result, width, height, stream_name)
                hands = stream["hands"]

                frames.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_sec": frame_index / fps if fps > 0 else math.nan,
                        "width": width,
                        "height": height,
                        stream_name: stream,
                    }
                )
                frame_rows.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_sec": frame_index / fps if fps > 0 else math.nan,
                        f"{stream_name}_hands": len(hands),
                        f"{stream_name}_ms": elapsed_ms,
                    }
                )
                for hand in hands:
                    box = hand["hand_bbox_xyxy_px"]
                    hand_rows.append(
                        {
                            "frame_index": frame_index,
                            "mode": stream_name,
                            "hand_index": hand["hand_index"],
                            "handedness": hand["handedness"],
                            "handedness_score": hand["handedness_score"],
                            "hand_x1": box[0],
                            "hand_y1": box[1],
                            "hand_x2": box[2],
                            "hand_y2": box[3],
                        }
                    )

                if args.save_vis and saved_vis < args.save_vis and hands:
                    cv2.imwrite(str(vis_dir / f"frame_{frame_index:06d}.jpg"), draw_tracking(frame_bgr, hands))
                    saved_vis += 1
                if writer is not None:
                    writer.write(draw_tracking(frame_bgr, hands))

                processed_frames += 1
                if processed_frames % 300 == 0:
                    print(
                        f"[mediapipe-tasks-{args.delegate}-{args.running_mode}] "
                        f"{record_time}: processed {processed_frames} frames",
                        flush=True,
                    )
        finally:
            cap.release()
            if writer is not None:
                writer.release()

    elapsed_wall = time.perf_counter() - start_wall
    summary: dict[str, Any] = {
        "task": "annotate_pianovam_mediapipe_tasks",
        "video": str(video_path),
        "output_dir": str(output_dir),
        "model": str(model_path),
        "model_url": DEFAULT_MODEL_URL,
        "delegate": args.delegate,
        "running_mode": args.running_mode.upper(),
        "stream": stream_name,
        "mediapipe_version": str(mp.__version__),
        "source_frame_count": source_frame_count,
        "processed_frames": processed_frames,
        "frame_stride": args.frame_stride,
        "start_frame": args.start_frame,
        "max_frames": args.max_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "num_hands": args.num_hands,
        "min_hand_detection_confidence": args.min_hand_detection_confidence,
        "min_hand_presence_confidence": args.min_hand_presence_confidence,
        "min_tracking_confidence": args.min_tracking_confidence,
        "annotations_json": str(annotations_path),
        "frames_csv": str(output_dir / "frames.csv"),
        "hands_csv": str(output_dir / "hands.csv"),
        "visualizations": saved_vis,
        "annotated_video": annotated_video_path,
        "elapsed_wall_sec": elapsed_wall,
        "effective_fps": processed_frames / elapsed_wall if elapsed_wall > 0 else math.nan,
        **scalar_summary(stream_ms, stream_name),
    }
    annotations = {
        "summary": summary,
        "schema": {
            stream_name: (
                "MediaPipe Tasks HandLandmarker VIDEO mode with temporal tracking."
                if args.running_mode == "video"
                else "MediaPipe Tasks HandLandmarker IMAGE mode; each processed frame is independent."
            ),
            "hands": "21 image-space hand landmarks, world landmarks, and handedness.",
            "palm_detections": "Always empty because MediaPipe Tasks does not expose raw palm detections.",
        },
        "frames": frames,
    }
    annotations_path.write_text(json.dumps(annotations, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "frames.csv", frame_rows)
    write_csv(output_dir / "hands.csv", hand_rows)
    return {
        "record_time": record_time,
        "status": "ok",
        "elapsed_sec": elapsed_wall,
        "processed_frames": processed_frames,
        "effective_fps": summary["effective_fps"],
        "output_dir": str(output_dir),
        "annotations": str(annotations_path),
    }


def main() -> int:
    args = parse_args()
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    data_root = resolve_path(args.data_root)
    output_root = resolve_path(args.output_root) if args.output_root else default_output_root(data_root, args)
    model_path = resolve_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"MediaPipe task model not found: {model_path}\n"
            f"Download it with:\n  wget -O {model_path} {DEFAULT_MODEL_URL}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    records = select_records(args, data_root)
    if not records:
        raise ValueError(f"No PianoVAM records selected from {data_root}")

    results: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        record_time = str(record["record_time"])
        print(f"[pianovam-mediapipe-tasks] {index}/{len(records)} {record_time}", flush=True)
        try:
            result = annotate_record(args, record, data_root, output_root)
        except Exception as exc:
            result = {"record_time": record_time, "status": "failed", "error": repr(exc)}
            print(f"  failed: {exc!r}", flush=True)
        results.append(result)
        print(f"  {result['status']}", flush=True)

    batch_summary = {
        "task": "annotate_pianovam_mediapipe_tasks",
        "data_root": str(data_root),
        "output_root": str(output_root),
        "model": str(model_path),
        "delegate": args.delegate,
        "split": args.split,
        "records": len(records),
        "ok": sum(1 for item in results if item["status"] in {"ok", "skipped_existing"}),
        "failed": sum(1 for item in results if item["status"] == "failed"),
        "missing_video": sum(1 for item in results if item["status"] == "missing_video"),
        "results": results,
    }
    (output_root / "annotation_batch_summary.json").write_text(
        json.dumps(batch_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(batch_summary, ensure_ascii=False, indent=2))
    return 0 if batch_summary["failed"] == 0 and batch_summary["missing_video"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
