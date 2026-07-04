"""Evaluation helpers for palm detection and hand landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05, dtype=np.float32)


@dataclass(frozen=True)
class PalmTarget:
    image_id: int
    image_name: str
    target_index: int
    box: np.ndarray
    palm7: np.ndarray


@dataclass(frozen=True)
class PalmPrediction:
    image_id: int
    image_name: str
    score: float
    box: np.ndarray
    palm7: np.ndarray | None


def list_images(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def xywhn_to_xyxy(values: list[float], width: int, height: int) -> np.ndarray:
    cx, cy, bw, bh = values
    cx *= width
    cy *= height
    bw *= width
    bh *= height
    return np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dtype=np.float32)


def load_targets(data_root: Path, split: str) -> dict[str, list[PalmTarget]]:
    image_dir = data_root / split / "images"
    label_dir = data_root / split / "labels"
    targets_by_image: dict[str, list[PalmTarget]] = {}
    for image_id, path in enumerate(list_images(image_dir)):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to read image: {path}")
        height, width = image.shape[:2]
        label_path = label_dir / f"{path.stem}.txt"
        targets: list[PalmTarget] = []
        if label_path.exists():
            for target_index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                values = [float(x) for x in line.split()]
                if len(values) < 19:
                    raise ValueError(f"{label_path} has {len(values)} fields, expected at least 19")
                palm7 = np.array(values[5:19], dtype=np.float32).reshape(7, 2)
                palm7[:, 0] *= width
                palm7[:, 1] *= height
                targets.append(
                    PalmTarget(
                        image_id=image_id,
                        image_name=path.name,
                        target_index=target_index,
                        box=xywhn_to_xyxy(values[1:5], width, height),
                        palm7=palm7,
                    )
                )
        targets_by_image[path.name] = targets
    return targets_by_image


def flatten_targets(targets_by_image: dict[str, list[PalmTarget]]) -> list[PalmTarget]:
    return [target for targets in targets_by_image.values() for target in targets]


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area1 + area2 - inter, 1e-9)


def target_key(target: PalmTarget) -> str:
    return f"{target.image_name}#{target.target_index}"


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapezoid(np.interp(x, mrec, mpre), x))


def evaluate_ap(
    preds: list[PalmPrediction],
    targets_by_image: dict[str, list[PalmTarget]],
    thresholds: np.ndarray,
    sim_fn: Callable[[PalmPrediction, PalmTarget], float],
) -> dict[str, float]:
    ordered = sorted(preds, key=lambda p: p.score, reverse=True)
    n_gt = len(flatten_targets(targets_by_image))
    aps: list[float] = []
    for threshold in thresholds:
        used: dict[str, set[int]] = {name: set() for name in targets_by_image}
        tp = np.zeros(len(ordered), dtype=np.float32)
        fp = np.zeros(len(ordered), dtype=np.float32)
        for idx, pred in enumerate(ordered):
            targets = targets_by_image.get(pred.image_name, [])
            if not targets:
                fp[idx] = 1
                continue
            sims = np.array([sim_fn(pred, target) for target in targets], dtype=np.float32)
            order = np.argsort(sims)[::-1]
            matched = None
            for pos in order:
                pos_i = int(pos)
                if float(sims[pos_i]) >= float(threshold) and pos_i not in used[pred.image_name]:
                    matched = pos_i
                    break
            if matched is None:
                fp[idx] = 1
            else:
                tp[idx] = 1
                used[pred.image_name].add(matched)
        tp_c = np.cumsum(tp)
        fp_c = np.cumsum(fp)
        recalls = tp_c / max(n_gt, 1)
        precisions = tp_c / np.maximum(tp_c + fp_c, 1e-9)
        aps.append(compute_ap(recalls, precisions))
    return {
        "ap@0.50": float(aps[0]) if aps else 0.0,
        "ap@0.75": float(aps[5]) if len(aps) > 5 else float("nan"),
        "map@0.50:0.95": float(np.mean(aps)) if aps else 0.0,
    }


def match_predictions(
    preds: list[PalmPrediction],
    targets_by_image: dict[str, list[PalmTarget]],
    conf: float,
    iou_threshold: float,
) -> tuple[dict[str, dict[str, Any]], int, int]:
    ordered = sorted([p for p in preds if p.score >= conf], key=lambda p: p.score, reverse=True)
    used: dict[str, set[int]] = {name: set() for name in targets_by_image}
    matches: dict[str, dict[str, Any]] = {}
    fp = 0
    for pred in ordered:
        targets = targets_by_image.get(pred.image_name, [])
        if not targets:
            fp += 1
            continue
        boxes = np.stack([target.box for target in targets], axis=0)
        ious = box_iou(pred.box, boxes)
        order = np.argsort(ious)[::-1]
        matched = None
        for pos in order:
            pos_i = int(pos)
            if float(ious[pos_i]) >= iou_threshold and pos_i not in used[pred.image_name]:
                matched = pos_i
                break
        if matched is None:
            fp += 1
            continue
        target = targets[matched]
        used[pred.image_name].add(matched)
        matches[target_key(target)] = {
            "score": pred.score,
            "iou": float(ious[matched]),
            "image_name": pred.image_name,
            "target_index": target.target_index,
        }
    return matches, len(ordered), fp


def detection_metrics(
    preds: list[PalmPrediction],
    targets_by_image: dict[str, list[PalmTarget]],
    conf: float,
    iou_threshold: float,
) -> dict[str, Any]:
    targets = flatten_targets(targets_by_image)
    matches, considered, fp = match_predictions(preds, targets_by_image, conf, iou_threshold)
    tp = len(matches)
    fn = len(targets) - tp
    ap = evaluate_ap(preds, targets_by_image, IOU_THRESHOLDS, lambda p, t: float(box_iou(p.box, t.box[None, :])[0]))
    operating_iou_sweep: dict[str, dict[str, Any]] = {}
    for threshold in (0.10, 0.25, 0.50, 0.75):
        sweep_matches, sweep_considered, sweep_fp = match_predictions(preds, targets_by_image, conf, threshold)
        sweep_tp = len(sweep_matches)
        sweep_fn = len(targets) - sweep_tp
        operating_iou_sweep[f"{threshold:.2f}"] = {
            "tp": sweep_tp,
            "fp": sweep_fp,
            "fn": sweep_fn,
            "precision": sweep_tp / max(sweep_considered, 1),
            "recall": sweep_tp / max(len(targets), 1),
            "miss_rate": sweep_fn / max(len(targets), 1),
        }
    return {
        "gt_targets": len(targets),
        "predictions": len(preds),
        "op_predictions": considered,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(considered, 1),
        "recall": tp / max(len(targets), 1),
        "miss_rate": fn / max(len(targets), 1),
        "ap@0.50": ap["ap@0.50"],
        "ap@0.75": ap["ap@0.75"],
        "map@0.50:0.95": ap["map@0.50:0.95"],
        "operating_iou_sweep": operating_iou_sweep,
    }
