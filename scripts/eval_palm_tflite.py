#!/usr/bin/env python3
"""Evaluate extracted MediaPipe palm detector TFLite model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from hand_pipeline.decode import decode_raw_palm
from hand_pipeline.decode import generate_palm_anchors
from hand_pipeline.decode import weighted_nms
from hand_pipeline.eval import PalmPrediction
from hand_pipeline.eval import box_iou
from hand_pipeline.eval import detection_metrics
from hand_pipeline.eval import list_images
from hand_pipeline.eval import load_targets
from hand_pipeline.inference import TfliteModel
from hand_pipeline.preprocess import image_to_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/palm_datasets")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--model",
        default="models/tflite/mediapipe_legacy_0_10_14_palm_detection_full.tflite",
    )
    parser.add_argument("--output-dir", default="runs/eval_palm_detector_tflite")
    parser.add_argument("--official-mediapipe", default="runs/mediapipe_baseline_vs_om/mediapipe_predictions.json")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--operating-conf", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.3)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--save-vis", type=int, default=0)
    return parser.parse_args()


def prediction_to_json(pred: PalmPrediction) -> dict[str, Any]:
    return {
        "image_id": pred.image_id,
        "image": pred.image_name,
        "score": pred.score,
        "box": pred.box.astype(float).tolist(),
        "palm7": None if pred.palm7 is None else pred.palm7.astype(float).tolist(),
    }


def load_official_mediapipe_reference(
    path: Path,
    targets_by_image: dict[str, Any],
    image_names: set[str] | None = None,
) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    items = json.loads(path.read_text(encoding="utf-8"))
    preds: list[tuple[str, float, np.ndarray]] = []
    for item in items:
        box = item.get("box")
        image = item.get("image", "")
        if not box:
            continue
        image_name = Path(str(image).replace("\\", "/")).name
        if image_names is not None and image_name not in image_names:
            continue
        preds.append((image_name, float(item.get("score", 1.0)), np.array(box, dtype=np.float32)))

    used: dict[str, set[int]] = {name: set() for name in targets_by_image}
    matched = 0
    fp = 0
    for image_name, _score, box in sorted(preds, key=lambda x: x[1], reverse=True):
        targets = targets_by_image.get(image_name, [])
        if not targets:
            fp += 1
            continue
        boxes = np.stack([target.box for target in targets], axis=0)
        ious = box_iou(box, boxes)
        order = np.argsort(ious)[::-1]
        found = None
        for pos in order:
            pos_i = int(pos)
            if float(ious[pos_i]) >= 0.10 and pos_i not in used[image_name]:
                found = pos_i
                break
        if found is None:
            fp += 1
        else:
            used[image_name].add(found)
            matched += 1

    total_gt = sum(len(v) for v in targets_by_image.values())
    return {
        "status": "ok",
        "path": str(path),
        "note": "Official MediaPipe output is full-hand box, not palm bbox; evaluated as visibility reference at IoU>=0.10.",
        "predictions": len(preds),
        "matched_targets@0.10": matched,
        "fp@0.10": fp,
        "fn@0.10": total_gt - matched,
        "precision@0.10": matched / max(len(preds), 1),
        "recall@0.10": matched / max(total_gt, 1),
    }


def draw_predictions(image: np.ndarray, preds: list[PalmPrediction]) -> np.ndarray:
    canvas = image.copy()
    for pred in preds:
        x1, y1, x2, y2 = [int(round(x)) for x in pred.box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
        if pred.palm7 is not None:
            for x, y in pred.palm7:
                cv2.circle(canvas, (int(round(x)), int(round(y))), 2, (0, 120, 255), -1)
        cv2.putText(
            canvas,
            f"{pred.score:.2f}",
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )
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


def write_report(path: Path, summary: dict[str, Any]) -> None:
    det = summary["palm_detector_tflite"]
    official = summary["official_mediapipe_reference"]
    model_name = Path(str(det.get("model", "palm_detector"))).name
    lines = [
        "# Palm Detector TFLite Evaluation",
        "",
        "## Dataset",
        "",
        f"- split: `{summary['split']}`",
        f"- images: `{summary['images']}`",
        f"- palm GT: `{det['gt_targets']}`",
        "",
        "## Detector Metrics",
        "",
        "| Model | Predictions | TP | FP | FN | Precision | Recall | AP@0.50 | AP@0.75 | mAP@0.50:0.95 | total_mean_ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| `{model_name}` | {det['predictions']} | {det['tp']} | {det['fp']} | {det['fn']} | "
            f"{det['precision']:.4f} | {det['recall']:.4f} | {det['ap@0.50']:.4f} | "
            f"{det['ap@0.75']:.4f} | {det['map@0.50:0.95']:.4f} | {det['total_mean_ms']:.4f} |"
        ),
        "",
        "The operating point fixes the score threshold at `operating_conf` and sweeps IoU.",
        "",
        "| IoU | TP | FP | FN | Precision | Recall | Miss rate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for iou_key, item in det.get("operating_iou_sweep", {}).items():
        lines.append(
            f"| {iou_key} | {item['tp']} | {item['fp']} | {item['fn']} | "
            f"{item['precision']:.4f} | {item['recall']:.4f} | {item['miss_rate']:.4f} |"
        )
    lines += [
        "",
        "## Official MediaPipe Reference",
        "",
        "The official HandLandmarker reference exposes full-hand boxes, not palm boxes. The loose IoU>=0.10 metric is only a visibility reference.",
        "",
        "| Source | Predictions | Matched | FP | FN | Precision | Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if official.get("status") == "ok":
        lines.append(
            f"| official MediaPipe full image | {official['predictions']} | {official['matched_targets@0.10']} | "
            f"{official['fp@0.10']} | {official['fn@0.10']} | {official['precision@0.10']:.4f} | {official['recall@0.10']:.4f} |"
        )
    else:
        lines.append(f"| official MediaPipe full image | missing | | | | | |")
    lines += [
        "",
        "## Notes",
        "",
        "- If mAP or recall regresses, inspect anchor generation, output order, letterbox removal, and NMS first.",
        "- Do not compare official full-hand boxes as palm bbox AP; the box definitions are different.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "vis"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data)
    image_paths = list_images(data_root / args.split / "images")
    if args.max_images:
        image_paths = image_paths[: args.max_images]
    targets_by_image = load_targets(data_root, args.split)
    if args.max_images:
        keep_names = {p.name for p in image_paths}
        targets_by_image = {k: v for k, v in targets_by_image.items() if k in keep_names}

    model = TfliteModel(args.model, num_threads=args.num_threads)
    anchors = generate_palm_anchors()
    predictions: list[PalmPrediction] = []
    timing_rows: list[dict[str, Any]] = []
    total_times: list[float] = []
    preprocess_times: list[float] = []
    infer_times: list[float] = []
    decode_times: list[float] = []
    saved_vis = 0

    for image_id, path in enumerate(image_paths):
        total_start = time.perf_counter()
        pre_start = time.perf_counter()
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to read image: {path}")
        tensor, letterbox = image_to_tensor(image, input_size=192)
        preprocess_ms = (time.perf_counter() - pre_start) * 1000.0

        infer_start = time.perf_counter()
        raw_boxes, raw_scores = model(tensor)
        infer_ms = (time.perf_counter() - infer_start) * 1000.0

        decode_start = time.perf_counter()
        decoded = decode_raw_palm(raw_boxes, raw_scores, anchors, letterbox, score_threshold=args.score_threshold)
        decoded = weighted_nms(decoded, iou_threshold=args.nms_iou, max_detections=args.max_det)
        decode_ms = (time.perf_counter() - decode_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0

        image_preds: list[PalmPrediction] = []
        for det in decoded:
            pred = PalmPrediction(
                image_id=image_id,
                image_name=path.name,
                score=det.score,
                box=det.box.astype(np.float32),
                palm7=det.keypoints.astype(np.float32),
            )
            image_preds.append(pred)
            predictions.append(pred)
        if args.save_vis and saved_vis < args.save_vis and image_preds:
            cv2.imwrite(str(vis_dir / f"{path.stem}.jpg"), draw_predictions(image, image_preds))
            saved_vis += 1

        total_times.append(total_ms)
        preprocess_times.append(preprocess_ms)
        infer_times.append(infer_ms)
        decode_times.append(decode_ms)
        timing_rows.append(
            {
                "image": path.name,
                "detections": len(image_preds),
                "preprocess_ms": preprocess_ms,
                "infer_ms": infer_ms,
                "decode_ms": decode_ms,
                "total_ms": total_ms,
            }
        )

    metrics = detection_metrics(predictions, targets_by_image, args.operating_conf, 0.50)
    metrics.update(
        {
            "model": str(args.model),
            "score_threshold": args.score_threshold,
            "operating_conf": args.operating_conf,
            "nms_iou": args.nms_iou,
            "max_det": args.max_det,
            "preprocess_mean_ms": float(np.mean(preprocess_times)) if preprocess_times else 0.0,
            "infer_mean_ms": float(np.mean(infer_times)) if infer_times else 0.0,
            "decode_mean_ms": float(np.mean(decode_times)) if decode_times else 0.0,
            "total_mean_ms": float(np.mean(total_times)) if total_times else 0.0,
            "fps": 1000.0 / max(float(np.mean(total_times)), 1e-9) if total_times else 0.0,
        }
    )
    official = load_official_mediapipe_reference(
        Path(args.official_mediapipe),
        targets_by_image,
        image_names={p.name for p in image_paths},
    )
    summary = {
        "task": "eval_mediapipe_palm_detector_tflite",
        "data": str(data_root),
        "split": args.split,
        "images": len(image_paths),
        "palm_detector_tflite": metrics,
        "official_mediapipe_reference": official,
    }

    (output_dir / "predictions.json").write_text(
        json.dumps([prediction_to_json(pred) for pred in predictions], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "timings.csv", timing_rows)
    write_report(output_dir / "report.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

