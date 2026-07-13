#!/usr/bin/env python3
"""Compare PianoVAM Handskeleton labels with legacy MediaPipe annotations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hand_pipeline.eval import box_iou  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/PianoVAM_v1")
    parser.add_argument("--annotation-root", default="", help="Default: <data-root>/mediapipe_legacy_annotations")
    parser.add_argument("--split", default="test", help="Metadata split to compare, or 'all'.")
    parser.add_argument("--record-time", action="append", default=[], help="Recording id. Can be repeated.")
    parser.add_argument("--streams", default="image,tracking", help="Comma-separated reference streams.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--match-iou", type=float, default=0.10)
    parser.add_argument("--max-videos", type=int, default=0)
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


def default_output_dir(args: argparse.Namespace) -> Path:
    suffix = "_".join(parse_streams(args.streams))
    return PROJECT_ROOT / "runs" / "pianovam_handskeleton_mediapipe_compare" / f"{args.split}_{suffix}"


def parse_streams(streams: str) -> list[str]:
    values = [item.strip() for item in streams.split(",") if item.strip()]
    invalid = [item for item in values if item not in {"image", "tracking"}]
    if invalid:
        raise ValueError(f"Unknown stream(s): {invalid}")
    return values


def hand_bbox(points: np.ndarray, width: int, height: int) -> np.ndarray:
    xy = np.asarray(points, dtype=np.float32)[:, :2]
    return np.asarray(
        [
            float(np.clip(np.nanmin(xy[:, 0]), 0.0, float(width))),
            float(np.clip(np.nanmin(xy[:, 1]), 0.0, float(height))),
            float(np.clip(np.nanmax(xy[:, 0]), 0.0, float(width))),
            float(np.clip(np.nanmax(xy[:, 1]), 0.0, float(height))),
        ],
        dtype=np.float32,
    )


def handskeleton_refs(frame_item: dict[str, Any], width: int, height: int) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for label in ("Left", "Right"):
        points = frame_item.get(label)
        if points is None:
            continue
        arr = np.asarray(points, dtype=np.float32)
        if arr.shape != (21, 3):
            continue
        px = arr.copy()
        px[:, 0] *= float(width)
        px[:, 1] *= float(height)
        px[:, 2] *= float(width)
        refs.append(
            {
                "label": label,
                "hand21": px,
                "hand_bbox": hand_bbox(px, width, height),
            }
        )
    return refs


def mediapipe_preds(frame_stream: dict[str, Any], width: int, height: int) -> list[dict[str, Any]]:
    preds: list[dict[str, Any]] = []
    for hand in frame_stream.get("hands", []):
        points = np.asarray(hand.get("hand21_keypoints_px", []), dtype=np.float32)
        if points.shape != (21, 3):
            continue
        bbox = hand.get("hand_bbox_xyxy_px")
        if bbox is None:
            bbox_arr = hand_bbox(points, width, height)
        else:
            bbox_arr = np.asarray(bbox, dtype=np.float32)
        preds.append(
            {
                "hand_index": int(hand.get("hand_index", len(preds))),
                "handedness": str(hand.get("handedness", "")),
                "handedness_score": float(hand.get("handedness_score", math.nan)),
                "palm_detection_index": int(hand.get("palm_detection_index", -1)),
                "palm_source": str(hand.get("palm_source", "")),
                "hand21": points,
                "hand_bbox": bbox_arr,
            }
        )
    return preds


def classify_stage_failure(
    *,
    stream: str,
    reference_hands: int,
    palm_detections: int,
    mediapipe_hands: int,
    matched_hands: int,
) -> str:
    if reference_hands == 0:
        return "no_reference_hands" if mediapipe_hands == 0 else "extra_mediapipe_hands"
    if matched_hands == reference_hands and mediapipe_hands == reference_hands:
        return "all_hands_matched"
    if stream != "image":
        if mediapipe_hands < reference_hands:
            return "tracking_landmark_or_roi_miss"
        if matched_hands < reference_hands:
            return "tracking_localization_or_matching_error"
        return "tracking_extra_mediapipe_hands"
    if palm_detections == 0:
        return "palm_no_detection"
    if palm_detections < reference_hands:
        return "palm_under_detected"
    if mediapipe_hands < reference_hands:
        return "landmark_under_detected_after_palm"
    if matched_hands < reference_hands:
        return "landmark_localization_or_matching_error"
    if mediapipe_hands > reference_hands:
        return "extra_mediapipe_hands"
    return "other_mismatch"


def match_hands(
    refs: list[dict[str, Any]],
    preds: list[dict[str, Any]],
    match_iou: float,
) -> tuple[list[tuple[int, int, float]], int, int]:
    if not refs or not preds:
        return [], len(refs), len(preds)
    matches: list[tuple[int, int, float]] = []
    used_pred: set[int] = set()
    pred_boxes = np.stack([pred["hand_bbox"] for pred in preds], axis=0)
    for ref_index, ref in enumerate(refs):
        ious = box_iou(ref["hand_bbox"], pred_boxes)
        order = np.argsort(ious)[::-1]
        for pos in order:
            pred_index = int(pos)
            if pred_index in used_pred:
                continue
            iou = float(ious[pred_index])
            if iou >= match_iou:
                used_pred.add(pred_index)
                matches.append((ref_index, pred_index, iou))
            break
    return matches, len(refs) - len(matches), len(preds) - len(used_pred)


def point_error(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    if a.shape != b.shape or a.size == 0:
        return {"mean": math.nan, "median": math.nan, "p95": math.nan, "max": math.nan}
    err = np.linalg.norm(a[:, :2].astype(np.float32) - b[:, :2].astype(np.float32), axis=1)
    return {
        "mean": float(np.mean(err)),
        "median": float(np.median(err)),
        "p95": float(np.percentile(err, 95)),
        "max": float(np.max(err)),
    }


def scalar_summary(values: list[float], prefix: str) -> dict[str, float]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_median": math.nan,
            f"{prefix}_p95": math.nan,
            f"{prefix}_max": math.nan,
        }
    arr = np.asarray(clean, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
        f"{prefix}_max": float(np.max(arr)),
    }


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


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.{digits}f}"
    return str(value)


def summarize_stream(
    stream: str,
    frame_rows: list[dict[str, Any]],
    match_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    frames = [row for row in frame_rows if row["stream"] == stream]
    matches = [row for row in match_rows if row["stream"] == stream]
    reference_hands = sum(int(row["handskeleton_hands"]) for row in frames)
    mediapipe_hands = sum(int(row["mediapipe_hands"]) for row in frames)
    palm_detections = sum(int(row["mediapipe_palm_detections"]) for row in frames)
    matched_hands = len(matches)
    count_mismatch_frames = sum(1 for row in frames if int(row["handskeleton_hands"]) != int(row["mediapipe_hands"]))
    stage_counts: dict[str, int] = {}
    for row in frames:
        key = str(row.get("stage_diagnosis", "unknown"))
        stage_counts[key] = stage_counts.get(key, 0) + 1
    return {
        "stream": stream,
        "processed_frames": len(frames),
        "handskeleton_hands": reference_hands,
        "mediapipe_hands": mediapipe_hands,
        "matched_hands": matched_hands,
        "unmatched_handskeleton_hands": reference_hands - matched_hands,
        "unmatched_mediapipe_hands": mediapipe_hands - matched_hands,
        "precision": matched_hands / max(mediapipe_hands, 1),
        "recall": matched_hands / max(reference_hands, 1),
        "miss_rate": (reference_hands - matched_hands) / max(reference_hands, 1),
        "count_mismatch_frames": count_mismatch_frames,
        "count_mismatch_rate": count_mismatch_frames / max(len(frames), 1),
        "palm_detection_frames": sum(1 for row in frames if int(row["mediapipe_palm_detections"]) > 0),
        "palm_detection_total": palm_detections,
        "palm_to_landmark_ratio": mediapipe_hands / max(palm_detections, 1),
        "stage_diagnosis_frames": stage_counts,
        **scalar_summary([row["hand21_mean_px"] for row in matches], "hand21_mean_px"),
        **scalar_summary([row["hand21_p95_px"] for row in matches], "hand21_p95_px"),
        **scalar_summary([row["hand21_max_px"] for row in matches], "hand21_max_px"),
        **scalar_summary([row["match_iou"] for row in matches], "hand_bbox_iou"),
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# PianoVAM Handskeleton vs Legacy MediaPipe",
        "",
        "## Inputs",
        "",
        f"- data root: `{summary['data_root']}`",
        f"- annotation root: `{summary['annotation_root']}`",
        f"- split: `{summary['split']}`",
        f"- videos: `{summary['videos']}`",
        f"- match IoU: `{summary['match_iou']}`",
        "",
        "## Stream Results",
        "",
        "| Stream | Frames | Matched | Precision | Recall | Miss rate | Count mismatch | Hand21 mean px | Hand21 P95 px | BBox IoU | Palm frames |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["streams"]:
        lines.append(
            "| "
            f"{item['stream']} | "
            f"{item['processed_frames']} | "
            f"{item['matched_hands']} | "
            f"{fmt(item['precision'])} | "
            f"{fmt(item['recall'])} | "
            f"{fmt(item['miss_rate'])} | "
            f"{fmt(item['count_mismatch_rate'])} | "
            f"{fmt(item['hand21_mean_px_mean'])} | "
            f"{fmt(item['hand21_p95_px_mean'])} | "
            f"{fmt(item['hand_bbox_iou_mean'])} | "
            f"{item['palm_detection_frames']} |"
        )
    lines.extend(
        [
            "",
            "## Stage Diagnosis",
            "",
            "| Stream | Diagnosis | Frames |",
            "| --- | --- | ---: |",
        ]
    )
    for item in summary["streams"]:
        for key, value in sorted(item.get("stage_diagnosis_frames", {}).items()):
            lines.append(f"| {item['stream']} | {key} | {value} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `image` is legacy MediaPipe with `use_prev_landmarks=False`, so palm detection should be run per processed frame.",
            "- `tracking` is legacy MediaPipe with `use_prev_landmarks=True`, so palm detections are absent on frames where tracking reuses previous landmarks.",
            "- The stage diagnosis uses frame-level counts. For image mode, a palm count below the Handskeleton hand count is treated as a palm-stage miss; enough palms but too few hand landmarks is treated as a landmark-stage miss after palm detection.",
            "- Hands are matched by hand-landmark bounding-box IoU, then keypoint error is computed in original video pixels.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def video_size(video_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()


def compare_record(
    *,
    record: dict[str, Any],
    data_root: Path,
    annotation_root: Path,
    streams: list[str],
    match_iou: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    record_time = str(record["record_time"])
    skeleton_path = data_root / "Handskeleton" / f"{record_time}.json"
    video_path = data_root / "Video" / f"{record_time}.mp4"
    annotations_path = annotation_root / record_time / "mediapipe_annotations.json"
    if not skeleton_path.exists():
        raise FileNotFoundError(str(skeleton_path))
    if not annotations_path.exists():
        raise FileNotFoundError(str(annotations_path))
    width, height = video_size(video_path)
    skeleton = json.loads(skeleton_path.read_text(encoding="utf-8-sig"))
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    frame_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    for frame in annotations.get("frames", []):
        frame_index = int(frame["frame_index"])
        skeleton_item = skeleton.get(str(frame_index))
        if skeleton_item is None:
            continue
        refs = handskeleton_refs(skeleton_item, width, height)
        for stream in streams:
            frame_stream = frame.get(stream, {})
            preds = mediapipe_preds(frame_stream, width, height)
            matches, unmatched_ref, unmatched_pred = match_hands(refs, preds, match_iou)
            palm_count = len(frame_stream.get("palm_detections", []))
            stage_diagnosis = classify_stage_failure(
                stream=stream,
                reference_hands=len(refs),
                palm_detections=palm_count,
                mediapipe_hands=len(preds),
                matched_hands=len(matches),
            )
            frame_rows.append(
                {
                    "record_time": record_time,
                    "split": record.get("split", ""),
                    "frame_index": frame_index,
                    "stream": stream,
                    "handskeleton_hands": len(refs),
                    "mediapipe_hands": len(preds),
                    "matched_hands": len(matches),
                    "unmatched_handskeleton_hands": unmatched_ref,
                    "unmatched_mediapipe_hands": unmatched_pred,
                    "mediapipe_palm_detections": palm_count,
                    "mediapipe_palm_rois": len(frame_stream.get("hand_rects_from_palm_detections", [])),
                    "mediapipe_tracking_rois": len(frame_stream.get("hand_rects_from_landmarks", [])),
                    "stage_diagnosis": stage_diagnosis,
                }
            )
            for ref_index, pred_index, iou in matches:
                ref = refs[ref_index]
                pred = preds[pred_index]
                err = point_error(pred["hand21"], ref["hand21"])
                match_rows.append(
                    {
                        "record_time": record_time,
                        "split": record.get("split", ""),
                        "frame_index": frame_index,
                        "stream": stream,
                        "handskeleton_label": ref["label"],
                        "mediapipe_index": pred["hand_index"],
                        "mediapipe_handedness": pred["handedness"],
                        "mediapipe_handedness_score": pred["handedness_score"],
                        "palm_detection_index": pred["palm_detection_index"],
                        "palm_source": pred["palm_source"],
                        "match_iou": iou,
                        "hand21_mean_px": err["mean"],
                        "hand21_median_px": err["median"],
                        "hand21_p95_px": err["p95"],
                        "hand21_max_px": err["max"],
                    }
                )
    return frame_rows, match_rows


def main() -> int:
    args = parse_args()
    data_root = resolve_path(args.data_root)
    annotation_root = resolve_path(args.annotation_root) if args.annotation_root else data_root / "mediapipe_legacy_annotations"
    output_dir = resolve_path(args.output_dir) if args.output_dir else default_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    streams = parse_streams(args.streams)
    records = select_records(args, data_root)
    if not records:
        raise ValueError(f"No PianoVAM records selected from {data_root}")

    all_frame_rows: list[dict[str, Any]] = []
    all_match_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, record in enumerate(records, start=1):
        record_time = str(record["record_time"])
        print(f"[handskeleton-compare] {index}/{len(records)} {record_time}", flush=True)
        try:
            frame_rows, match_rows = compare_record(
                record=record,
                data_root=data_root,
                annotation_root=annotation_root,
                streams=streams,
                match_iou=args.match_iou,
            )
        except Exception as exc:
            failures.append({"record_time": record_time, "error": str(exc)})
            print(f"  failed: {exc}", flush=True)
            continue
        all_frame_rows.extend(frame_rows)
        all_match_rows.extend(match_rows)

    stream_summaries = [summarize_stream(stream, all_frame_rows, all_match_rows) for stream in streams]
    summary = {
        "task": "compare_pianovam_handskeleton_mediapipe",
        "data_root": str(data_root),
        "annotation_root": str(annotation_root),
        "output_dir": str(output_dir),
        "split": args.split,
        "record_times": [str(record["record_time"]) for record in records],
        "videos": len(records),
        "streams": stream_summaries,
        "match_iou": args.match_iou,
        "failures": failures,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "frames.csv", all_frame_rows)
    write_csv(output_dir / "matches.csv", all_match_rows)
    write_report(output_dir / "report.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
