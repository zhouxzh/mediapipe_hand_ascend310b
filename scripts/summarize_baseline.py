#!/usr/bin/env python3
"""Summarize baseline outputs and fail if core checks regress."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def fmt(value: Any, digits: int = 6) -> str:
    number = as_float(value)
    if math.isnan(number):
        return "NA"
    return f"{number:.{digits}f}"


def pick(data: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="runs/baseline")
    parser.add_argument("--max-legacy-rect-mean-px", type=float, default=0.05)
    parser.add_argument("--min-legacy-rect-pck05", type=float, default=0.999)
    parser.add_argument("--max-two-stage-legacy-mean-px", type=float, default=1.0)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    root = Path(args.output_root)
    palm = load_json(root / "palm_detector" / "summary.json")
    current = load_json(root / "two_stage_vs_current_tasks" / "summary.json")
    legacy_graph = load_json(root / "legacy_graph" / "summary.json")
    legacy = load_json(root / "two_stage_vs_legacy_graph" / "summary.json")
    legacy_rect = load_json(root / "legacy_rect_landmark" / "summary.json")
    handlm = load_json(root / "handlm_manual_gt" / "summary.json")
    matrix = load_json(root / "tflite_matrix" / "summary.json")
    run_config = load_json(root / "run_config.json")

    checks: list[dict[str, Any]] = []
    add_check(checks, "palm detector summary exists", palm is not None, str(root / "palm_detector" / "summary.json"))
    add_check(
        checks,
        "current Tasks two-stage summary exists",
        current is not None,
        str(root / "two_stage_vs_current_tasks" / "summary.json"),
    )
    add_check(checks, "legacy graph summary exists", legacy_graph is not None, str(root / "legacy_graph" / "summary.json"))
    add_check(
        checks,
        "two-stage vs legacy summary exists",
        legacy is not None,
        str(root / "two_stage_vs_legacy_graph" / "summary.json"),
    )
    add_check(
        checks,
        "legacy rect landmark summary exists",
        legacy_rect is not None,
        str(root / "legacy_rect_landmark" / "summary.json"),
    )
    add_check(
        checks,
        "manual hand landmark GT summary exists",
        handlm is not None,
        str(root / "handlm_manual_gt" / "summary.json"),
    )
    expected_split = (run_config or {}).get("palm_split")
    if expected_split:
        split_sources = {
            "palm detector": pick(palm, "split"),
            "current Tasks two-stage": pick(current, "split"),
            "legacy graph": pick(legacy_graph, "split"),
            "two-stage vs legacy": pick(legacy, "split"),
        }
        if matrix is not None and isinstance(matrix, list):
            matrix_splits = {item.get("split") for item in matrix if isinstance(item, dict)}
            split_sources["TFLite matrix"] = ",".join(sorted(str(item) for item in matrix_splits))
        for name, actual_split in split_sources.items():
            add_check(
                checks,
                f"{name} split matches run config",
                actual_split == expected_split,
                f"{actual_split} == {expected_split}",
            )

    legacy_rect_mean = as_float(pick(legacy_rect, "mean_px"))
    legacy_rect_pck05 = as_float(pick(legacy_rect, "pck@0.05"))
    legacy_mean = as_float(pick(legacy, "vs_mediapipe_mean_px"))

    add_check(
        checks,
        "legacy rect mean px threshold",
        not math.isnan(legacy_rect_mean) and legacy_rect_mean <= args.max_legacy_rect_mean_px,
        f"{fmt(legacy_rect_mean)} <= {args.max_legacy_rect_mean_px}",
    )
    add_check(
        checks,
        "legacy rect PCK@0.05 threshold",
        not math.isnan(legacy_rect_pck05) and legacy_rect_pck05 >= args.min_legacy_rect_pck05,
        f"{fmt(legacy_rect_pck05)} >= {args.min_legacy_rect_pck05}",
    )
    add_check(
        checks,
        "two-stage palm route vs legacy mean px threshold",
        not math.isnan(legacy_mean) and legacy_mean <= args.max_two_stage_legacy_mean_px,
        f"{fmt(legacy_mean)} <= {args.max_two_stage_legacy_mean_px}",
    )

    summary: dict[str, Any] = {
        "output_root": str(root),
        "run_config": run_config or {},
        "checks": checks,
        "palm_detector": {
            "data": pick(palm, "data"),
            "split": pick(palm, "split"),
            "images": pick(palm, "images"),
            "gt_targets": pick(palm, "palm_detector_tflite", "gt_targets"),
            "predictions": pick(palm, "palm_detector_tflite", "predictions"),
            "precision": pick(palm, "palm_detector_tflite", "precision"),
            "recall": pick(palm, "palm_detector_tflite", "recall"),
            "ap50": pick(palm, "palm_detector_tflite", "ap@0.50"),
            "map": pick(palm, "palm_detector_tflite", "map@0.50:0.95"),
            "total_mean_ms": pick(palm, "palm_detector_tflite", "total_mean_ms"),
        },
        "two_stage_vs_current_tasks": {
            "data": pick(current, "data"),
            "split": pick(current, "split"),
            "images": pick(current, "images"),
            "matched_hands": pick(current, "matched_hands"),
            "mean_px": pick(current, "vs_mediapipe_mean_px"),
            "median_px": pick(current, "vs_mediapipe_median_px"),
            "p95_px": pick(current, "vs_mediapipe_p95_px"),
            "nme": pick(current, "vs_mediapipe_nme"),
            "pck05": pick(current, "vs_mediapipe_pck@0.05"),
            "pck10": pick(current, "vs_mediapipe_pck@0.10"),
            "total_mean_ms": pick(current, "total_mean_ms"),
        },
        "two_stage_vs_legacy_graph": {
            "data": pick(legacy, "data"),
            "split": pick(legacy, "split"),
            "images": pick(legacy, "images"),
            "matched_hands": pick(legacy, "matched_hands"),
            "mean_px": pick(legacy, "vs_mediapipe_mean_px"),
            "median_px": pick(legacy, "vs_mediapipe_median_px"),
            "p95_px": pick(legacy, "vs_mediapipe_p95_px"),
            "nme": pick(legacy, "vs_mediapipe_nme"),
            "pck05": pick(legacy, "vs_mediapipe_pck@0.05"),
            "pck10": pick(legacy, "vs_mediapipe_pck@0.10"),
            "total_mean_ms": pick(legacy, "total_mean_ms"),
        },
        "legacy_rect_landmark": {
            "hands": pick(legacy_rect, "hands"),
            "mean_px": pick(legacy_rect, "mean_px"),
            "median_px": pick(legacy_rect, "median_px"),
            "p95_px": pick(legacy_rect, "p95_px"),
            "nme": pick(legacy_rect, "nme"),
            "pck05": pick(legacy_rect, "pck@0.05"),
            "pck10": pick(legacy_rect, "pck@0.10"),
        },
        "handlm_manual_gt": {
            "data": pick(handlm, "data"),
            "split": None,
            "images": pick(handlm, "images"),
            "hands": pick(handlm, "hands"),
            "visible_points": pick(handlm, "visible_points"),
            "models": pick(handlm, "models"),
        },
        "tflite_matrix": {
            "available": matrix is not None,
            "summary": matrix,
        },
    }

    output_json = Path(args.output_json) if args.output_json else root / "verification_summary.json"
    output_md = Path(args.output_md) if args.output_md else root / "verification_summary.md"

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Baseline Verification Summary",
        "",
        "This file is generated by `scripts/summarize_baseline.py`.",
        "Do not copy these values back into static documentation; rerun verification instead.",
        "",
        "## Run Configuration",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| palm dataset | `{(run_config or {}).get('palm_dataset', pick(palm, 'data'))}` |",
        f"| palm split | `{(run_config or {}).get('palm_split', pick(palm, 'split'))}` |",
        f"| two-stage dataset | `{(run_config or {}).get('palm_split_images', '')}` |",
        f"| hand landmark manual GT | `{(run_config or {}).get('handlm_dataset', pick(handlm, 'data'))}` |",
        f"| current MediaPipe reference | `{(run_config or {}).get('current_reference', pick(current, 'official_mediapipe'))}` |",
        f"| reference alignment | `{(run_config or {}).get('reference_alignment_note', 'references must match the evaluated split')}` |",
        "",
        "## Checks",
        "",
        "| Check | Result | Detail |",
        "| --- | --- | --- |",
    ]
    for item in checks:
        result = "PASS" if item["passed"] else "FAIL"
        lines.append(f"| {item['name']} | {result} | {item['detail']} |")

    def section(title: str, rows: list[tuple[str, Any]]) -> None:
        lines.extend(["", f"## {title}", "", "| Metric | Value |", "| --- | ---: |"])
        for key, value in rows:
            lines.append(f"| {key} | {fmt(value)} |")

    section(
        "Palm Detector",
        [
            ("images", pick(palm, "images")),
            ("gt_targets", pick(palm, "palm_detector_tflite", "gt_targets")),
            ("predictions", pick(palm, "palm_detector_tflite", "predictions")),
            ("precision", pick(palm, "palm_detector_tflite", "precision")),
            ("recall", pick(palm, "palm_detector_tflite", "recall")),
            ("AP@0.50", pick(palm, "palm_detector_tflite", "ap@0.50")),
            ("mAP@0.50:0.95", pick(palm, "palm_detector_tflite", "map@0.50:0.95")),
            ("total_mean_ms", pick(palm, "palm_detector_tflite", "total_mean_ms")),
        ],
    )
    section(
        "Two Stage vs Current Tasks",
        [
            ("images", pick(current, "images")),
            ("matched_hands", pick(current, "matched_hands")),
            ("mean_px", pick(current, "vs_mediapipe_mean_px")),
            ("median_px", pick(current, "vs_mediapipe_median_px")),
            ("p95_px", pick(current, "vs_mediapipe_p95_px")),
            ("NME", pick(current, "vs_mediapipe_nme")),
            ("PCK@0.05", pick(current, "vs_mediapipe_pck@0.05")),
            ("PCK@0.10", pick(current, "vs_mediapipe_pck@0.10")),
            ("total_mean_ms", pick(current, "total_mean_ms")),
        ],
    )
    section(
        "Two Stage vs Legacy Graph",
        [
            ("images", pick(legacy, "images")),
            ("matched_hands", pick(legacy, "matched_hands")),
            ("mean_px", pick(legacy, "vs_mediapipe_mean_px")),
            ("median_px", pick(legacy, "vs_mediapipe_median_px")),
            ("p95_px", pick(legacy, "vs_mediapipe_p95_px")),
            ("NME", pick(legacy, "vs_mediapipe_nme")),
            ("PCK@0.05", pick(legacy, "vs_mediapipe_pck@0.05")),
            ("PCK@0.10", pick(legacy, "vs_mediapipe_pck@0.10")),
            ("total_mean_ms", pick(legacy, "total_mean_ms")),
        ],
    )
    section(
        "Legacy Rect Landmark",
        [
            ("hands", pick(legacy_rect, "hands")),
            ("mean_px", pick(legacy_rect, "mean_px")),
            ("median_px", pick(legacy_rect, "median_px")),
            ("p95_px", pick(legacy_rect, "p95_px")),
            ("NME", pick(legacy_rect, "nme")),
            ("PCK@0.05", pick(legacy_rect, "pck@0.05")),
            ("PCK@0.10", pick(legacy_rect, "pck@0.10")),
        ],
    )
    lines.extend(["", "## Hand Landmark Manual GT", "", "| Model | Mean px | Median px | P95 px | NME | PCK@0.05 | PCK@0.10 | total_mean_ms |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for item in pick(handlm, "models") or []:
        lines.append(
            f"| `{item.get('key')}` | {fmt(item.get('gt_mean_px'))} | {fmt(item.get('gt_median_px'))} | "
            f"{fmt(item.get('gt_p95_px'))} | {fmt(item.get('gt_nme'))} | {fmt(item.get('gt_pck@0.05'))} | "
            f"{fmt(item.get('gt_pck@0.10'))} | {fmt(item.get('total_ms_mean'))} |"
        )

    if matrix is not None:
        lines.extend(["", "## TFLite Matrix", "", "`runs/baseline/tflite_matrix/summary.json` is available."])
    else:
        lines.extend(["", "## TFLite Matrix", "", "Not run. Use `--run-matrix` in `scripts/run_baseline.py` when needed."])

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    failed = [item for item in checks if not item["passed"]]
    if failed:
        print(f"Baseline verification failed: {len(failed)} check(s) failed. See {output_md}", file=sys.stderr)
        return 1
    print(f"Baseline verification passed. Summary: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

