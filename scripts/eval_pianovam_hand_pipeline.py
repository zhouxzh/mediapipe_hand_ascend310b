#!/usr/bin/env python3
"""Evaluate the hand pipeline on PianoVAM videos and skeleton labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hand_pipeline.eval import box_iou  # noqa: E402
from hand_pipeline.two_stage import OmHandPipeline  # noqa: E402
from hand_pipeline.two_stage import OnnxHandPipeline  # noqa: E402
from hand_pipeline.visualization import HAND_EDGES  # noqa: E402


MODEL_SETS = {
    "full": {
        "onnx_detector": "models/onnx/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices.onnx",
        "onnx_landmark": "models/onnx/mediapipe_legacy_0_10_14_hand_landmark_full.onnx",
        "om_detector": "models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om",
        "om_landmark": "models/om/mediapipe_legacy_0_10_14_hand_landmark_full.om",
    },
    "lite": {
        "onnx_detector": "models/onnx/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices.onnx",
        "onnx_landmark": "models/onnx/mediapipe_legacy_0_10_14_hand_landmark_lite.onnx",
        "om_detector": "models/om/mediapipe_legacy_0_10_14_palm_detection_lite_downsample_resize_maxpool_slices_origin_dtype.om",
        "om_landmark": "models/om/mediapipe_legacy_0_10_14_hand_landmark_lite.om",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/PianoVAM_v1")
    parser.add_argument("--split", default="test", help="Metadata split to evaluate, or 'all'.")
    parser.add_argument("--record-time", action="append", default=[], help="Recording id. Can be repeated.")
    parser.add_argument("--backend", choices=["onnx", "om"], default="om")
    parser.add_argument("--model-set", choices=sorted(MODEL_SETS), default="full")
    parser.add_argument("--detector", default="", help="Override detector model path.")
    parser.add_argument("--landmark", default="", help="Override landmark model path.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--pipeline-mode", choices=["image", "tracking"], default="tracking")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.3)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--min-hand-score", type=float, default=0.5)
    parser.add_argument("--match-iou", type=float, default=0.10)
    parser.add_argument("--frame-stride", type=int, default=60)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames-per-video", type=int, default=0)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--save-vis", type=int, default=0)
    parser.add_argument("--reload-detector-each-frame", action="store_true")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def default_output_dir(args: argparse.Namespace) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"{args.split}_{args.backend}_{args.model_set}_{args.pipeline_mode}_{stamp}"
    return PROJECT_ROOT / "runs" / "pianovam_hand_pipeline" / name


def load_metadata(data_root: Path) -> list[dict[str, Any]]:
    metadata_path = data_root / "metadata.json"
    data = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    records: list[dict[str, Any]] = []
    for key, item in data.items():
        record = dict(item)
        record["metadata_id"] = str(key)
        records.append(record)
    return records


def select_records(args: argparse.Namespace, data_root: Path) -> list[dict[str, Any]]:
    records = load_metadata(data_root)
    if args.record_time:
        wanted = set(args.record_time)
        records = [item for item in records if str(item.get("record_time")) in wanted]
    elif args.split != "all":
        records = [item for item in records if str(item.get("split")) == args.split]
    records.sort(key=lambda item: str(item.get("record_time", "")))
    if args.max_videos:
        records = records[: args.max_videos]
    return records


def skeleton_hands(frame_item: dict[str, Any], width: int, height: int) -> list[dict[str, Any]]:
    hands: list[dict[str, Any]] = []
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
        hands.append(
            {
                "label": label,
                "hand21": px,
                "hand_bbox": hand_bbox_from_points(px, width, height),
            }
        )
    return hands


def hand_bbox_from_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
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


def prediction_hands(predictions: list[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    hands: list[dict[str, Any]] = []
    for index, pred in enumerate(predictions):
        hand21 = np.asarray(pred.get("hand21", []), dtype=np.float32)
        if hand21.shape != (21, 3):
            continue
        hands.append(
            {
                "index": index,
                "score": float(pred.get("score", math.nan)),
                "hand_score": float(pred.get("hand_score", math.nan)),
                "source_roi": str(pred.get("source_roi", "")),
                "hand21": hand21,
                "hand_bbox": hand_bbox_from_points(hand21, width, height),
            }
        )
    return hands


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


def point_error(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    if pred.shape != ref.shape or pred.size == 0:
        return {"mean": math.nan, "median": math.nan, "p95": math.nan, "max": math.nan}
    err = np.linalg.norm(pred[:, :2].astype(np.float32) - ref[:, :2].astype(np.float32), axis=1)
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


def draw_overlay(
    frame: np.ndarray,
    refs: list[dict[str, Any]],
    preds: list[dict[str, Any]],
    frame_index: int,
    record_time: str,
) -> np.ndarray:
    canvas = frame.copy()
    for ref in refs:
        points = ref["hand21"]
        for start, end in HAND_EDGES:
            cv2.line(canvas, tuple(np.round(points[start, :2]).astype(int)), tuple(np.round(points[end, :2]).astype(int)), (255, 120, 40), 2)
        for point in points:
            cv2.circle(canvas, tuple(np.round(point[:2]).astype(int)), 3, (255, 120, 40), -1)
    for pred in preds:
        points = pred["hand21"]
        for start, end in HAND_EDGES:
            cv2.line(canvas, tuple(np.round(points[start, :2]).astype(int)), tuple(np.round(points[end, :2]).astype(int)), (60, 230, 120), 2)
        for point in points:
            cv2.circle(canvas, tuple(np.round(point[:2]).astype(int)), 3, (60, 230, 120), -1)
    cv2.putText(
        canvas,
        f"{record_time} frame {frame_index} PianoVAM orange, pipeline green",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return canvas


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# PianoVAM Hand Pipeline Evaluation",
        "",
        "## Result",
        "",
        f"- backend: `{summary['backend']}`",
        f"- model set: `{summary['model_set']}`",
        f"- split: `{summary['split']}`",
        f"- videos: `{summary['videos']}`",
        f"- processed frames: `{summary['processed_frames']}`",
        f"- matched hands: `{summary['matched_hands']}`",
        f"- detection precision: `{fmt(summary['detection_precision'])}`",
        f"- detection recall: `{fmt(summary['detection_recall'])}`",
        f"- miss rate: `{fmt(summary['miss_rate'])}`",
        f"- count mismatch rate: `{fmt(summary['count_mismatch_rate'], 6)}`",
        f"- hand21 mean px: `{fmt(summary['hand21_mean_px_mean'])}`",
        f"- hand21 p95 px: `{fmt(summary['hand21_mean_px_p95'])}`",
        f"- hand bbox IoU mean: `{fmt(summary['hand_bbox_iou_mean'])}`",
        f"- estimated pipeline FPS: `{fmt(summary['estimated_pipeline_fps'])}`",
        "",
        "## Timing",
        "",
        "| Stage | Mean | Median | P95 | Max |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| total ms | {fmt(summary['total_ms_mean'])} | {fmt(summary['total_ms_median'])} | {fmt(summary['total_ms_p95'])} | {fmt(summary['total_ms_max'])} |",
        f"| detector ms | {fmt(summary['detector_ms_mean'])} | {fmt(summary['detector_ms_median'])} | {fmt(summary['detector_ms_p95'])} | {fmt(summary['detector_ms_max'])} |",
        f"| landmark ms | {fmt(summary['landmark_ms_mean'])} | {fmt(summary['landmark_ms_median'])} | {fmt(summary['landmark_ms_p95'])} | {fmt(summary['landmark_ms_max'])} |",
        "",
        "## Notes",
        "",
        "- PianoVAM Handskeleton labels are MediaPipe-generated landmarks, so these metrics measure consistency against that reference, not manual ground truth.",
        "- `total_ms` is Python pipeline time for the selected backend and includes preprocessing, decode, ROI, model inference, and postprocessing.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_pipeline(args: argparse.Namespace, detector: Path, landmark: Path) -> Any:
    common = {
        "score_threshold": args.score_threshold,
        "nms_iou": args.nms_iou,
        "max_det": args.max_det,
        "max_hands": args.max_hands,
        "min_hand_score": args.min_hand_score,
        "mode": args.pipeline_mode,
    }
    if args.backend == "onnx":
        return OnnxHandPipeline(detector, landmark, **common)
    return OmHandPipeline(
        detector,
        landmark,
        device_id=args.device_id,
        reload_detector_each_frame=args.reload_detector_each_frame,
        **common,
    )


def evaluate_record(
    *,
    record: dict[str, Any],
    data_root: Path,
    pipeline: Any,
    args: argparse.Namespace,
    output_dir: Path,
    vis_dir: Path,
    saved_vis: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    record_time = str(record["record_time"])
    video_path = data_root / "Video" / f"{record_time}.mp4"
    skeleton_path = data_root / "Handskeleton" / f"{record_time}.json"
    if not video_path.exists():
        raise FileNotFoundError(str(video_path))
    if not skeleton_path.exists():
        raise FileNotFoundError(str(skeleton_path))

    skeleton = json.loads(skeleton_path.read_text(encoding="utf-8-sig"))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    frame_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    pipeline.reset()
    try:
        source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        processed = 0
        frame_index = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if frame_index < args.start_frame:
                continue
            if (frame_index - args.start_frame) % args.frame_stride != 0:
                continue
            if args.max_frames_per_video and processed >= args.max_frames_per_video:
                break
            ref_item = skeleton.get(str(frame_index))
            if ref_item is None:
                continue

            result = pipeline.infer(frame)
            refs = skeleton_hands(ref_item, width, height)
            preds = prediction_hands(result.predictions, width, height)
            matches, unmatched_ref, unmatched_pred = match_hands(refs, preds, args.match_iou)
            row = {
                "record_time": record_time,
                "split": record.get("split", ""),
                "frame_index": frame_index,
                "time_sec": frame_index / fps if fps > 0 else math.nan,
                "reference_hands": len(refs),
                "predicted_hands": len(preds),
                "matched_hands": len(matches),
                "unmatched_reference_hands": unmatched_ref,
                "unmatched_predicted_hands": unmatched_pred,
                "total_ms": float(result.timings.get("total_ms", math.nan)),
                "preprocess_ms": float(result.timings.get("preprocess_ms", math.nan)),
                "detector_ms": float(result.timings.get("detector_ms", math.nan)),
                "decode_ms": float(result.timings.get("decode_ms", math.nan)),
                "roi_ms": float(result.timings.get("roi_ms", math.nan)),
                "landmark_ms": float(result.timings.get("landmark_ms", math.nan)),
                "post_ms": float(result.timings.get("post_ms", math.nan)),
                "palm_detector_skipped": bool(result.timings.get("palm_detector_skipped", False)),
                "source_frame_count": source_frame_count,
                "fps": fps,
                "width": width,
                "height": height,
            }
            frame_rows.append(row)

            for ref_index, pred_index, iou in matches:
                ref = refs[ref_index]
                pred = preds[pred_index]
                err = point_error(pred["hand21"], ref["hand21"])
                match_rows.append(
                    {
                        "record_time": record_time,
                        "frame_index": frame_index,
                        "reference_label": ref["label"],
                        "predicted_index": pred["index"],
                        "match_iou": iou,
                        "hand21_mean_px": err["mean"],
                        "hand21_median_px": err["median"],
                        "hand21_p95_px": err["p95"],
                        "hand21_max_px": err["max"],
                        "predicted_score": pred["score"],
                        "predicted_hand_score": pred["hand_score"],
                        "source_roi": pred["source_roi"],
                    }
                )

            if args.save_vis and saved_vis < args.save_vis and (refs or preds):
                vis_path = vis_dir / f"{record_time}_frame_{frame_index:06d}.jpg"
                cv2.imwrite(str(vis_path), draw_overlay(frame, refs, preds, frame_index, record_time))
                saved_vis += 1
            processed += 1
    finally:
        cap.release()
    return frame_rows, match_rows, saved_vis


def main() -> int:
    args = parse_args()
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")

    data_root = resolve_path(args.data_root)
    output_dir = resolve_path(args.output_dir) if args.output_dir else default_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "vis"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    records = select_records(args, data_root)
    if not records:
        raise ValueError(f"No PianoVAM records selected from {data_root}")

    defaults = MODEL_SETS[args.model_set]
    detector = resolve_path(args.detector or defaults[f"{args.backend}_detector"])
    landmark = resolve_path(args.landmark or defaults[f"{args.backend}_landmark"])
    if not detector.exists():
        raise FileNotFoundError(str(detector))
    if not landmark.exists():
        raise FileNotFoundError(str(landmark))

    pipeline = create_pipeline(args, detector, landmark)
    frame_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    saved_vis = 0
    try:
        for index, record in enumerate(records, start=1):
            print(f"[pianovam] {index}/{len(records)} {record['record_time']}", flush=True)
            record_frame_rows, record_match_rows, saved_vis = evaluate_record(
                record=record,
                data_root=data_root,
                pipeline=pipeline,
                args=args,
                output_dir=output_dir,
                vis_dir=vis_dir,
                saved_vis=saved_vis,
            )
            frame_rows.extend(record_frame_rows)
            match_rows.extend(record_match_rows)
    finally:
        pipeline.close()

    processed_frames = len(frame_rows)
    reference_hands = sum(int(row["reference_hands"]) for row in frame_rows)
    predicted_hands = sum(int(row["predicted_hands"]) for row in frame_rows)
    matched_hands = len(match_rows)
    count_mismatch_frames = sum(1 for row in frame_rows if int(row["reference_hands"]) != int(row["predicted_hands"]))
    timing_summary = {
        **scalar_summary([row["total_ms"] for row in frame_rows], "total_ms"),
        **scalar_summary([row["preprocess_ms"] for row in frame_rows], "preprocess_ms"),
        **scalar_summary([row["detector_ms"] for row in frame_rows], "detector_ms"),
        **scalar_summary([row["decode_ms"] for row in frame_rows], "decode_ms"),
        **scalar_summary([row["roi_ms"] for row in frame_rows], "roi_ms"),
        **scalar_summary([row["landmark_ms"] for row in frame_rows], "landmark_ms"),
        **scalar_summary([row["post_ms"] for row in frame_rows], "post_ms"),
    }
    mean_total_ms = timing_summary["total_ms_mean"]
    summary: dict[str, Any] = {
        "task": "eval_pianovam_hand_pipeline",
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "split": args.split,
        "record_times": [str(record["record_time"]) for record in records],
        "videos": len(records),
        "backend": args.backend,
        "model_set": args.model_set,
        "detector": str(detector),
        "landmark": str(landmark),
        "pipeline_mode": args.pipeline_mode,
        "frame_stride": args.frame_stride,
        "start_frame": args.start_frame,
        "max_frames_per_video": args.max_frames_per_video,
        "processed_frames": processed_frames,
        "reference_hands": reference_hands,
        "predicted_hands": predicted_hands,
        "matched_hands": matched_hands,
        "unmatched_reference_hands": reference_hands - matched_hands,
        "unmatched_predicted_hands": predicted_hands - matched_hands,
        "detection_precision": matched_hands / max(predicted_hands, 1),
        "detection_recall": matched_hands / max(reference_hands, 1),
        "miss_rate": (reference_hands - matched_hands) / max(reference_hands, 1),
        "count_mismatch_frames": count_mismatch_frames,
        "count_mismatch_rate": count_mismatch_frames / max(processed_frames, 1),
        "match_iou": args.match_iou,
        "visualizations": saved_vis,
        "estimated_pipeline_fps": 1000.0 / mean_total_ms if math.isfinite(mean_total_ms) and mean_total_ms > 0 else math.nan,
        **scalar_summary([row["hand21_mean_px"] for row in match_rows], "hand21_mean_px"),
        **scalar_summary([row["hand21_p95_px"] for row in match_rows], "hand21_p95_px"),
        **scalar_summary([row["hand21_max_px"] for row in match_rows], "hand21_max_px"),
        **scalar_summary([row["match_iou"] for row in match_rows], "hand_bbox_iou"),
        **timing_summary,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "frames.csv", frame_rows)
    write_csv(output_dir / "matches.csv", match_rows)
    write_report(output_dir / "report.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
