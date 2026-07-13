#!/usr/bin/env python3
"""Run useful ONNX-vs-OM video conversion checks on Ascend 310B."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OPTIMIZED_PALM_ONNX = "models/onnx/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices.onnx"
ORIGINAL_PALM_ONNX = "models/onnx/mediapipe_legacy_0_10_14_palm_detection_full.onnx"
LEGACY_FULL_LANDMARK_ONNX = "models/onnx/mediapipe_legacy_0_10_14_hand_landmark_full.onnx"
OPTIMIZED_PALM_OM = "models/om/mediapipe_legacy_0_10_14_palm_detection_full_downsample_resize_maxpool_slices_origin_dtype.om"
LEGACY_FULL_LANDMARK_OM = "models/om/mediapipe_legacy_0_10_14_hand_landmark_full.om"


@dataclass(frozen=True)
class CheckCase:
    name: str
    description: str
    onnx_detector: str
    onnx_landmark: str
    om_detector: str
    om_landmark: str


CASES = {
    "optimized": CheckCase(
        name="optimized",
        description="ATC input optimized palm ONNX vs final optimized palm OM.",
        onnx_detector=OPTIMIZED_PALM_ONNX,
        onnx_landmark=LEGACY_FULL_LANDMARK_ONNX,
        om_detector=OPTIMIZED_PALM_OM,
        om_landmark=LEGACY_FULL_LANDMARK_OM,
    ),
    "original": CheckCase(
        name="original",
        description="Original legacy full palm ONNX vs final optimized palm OM.",
        onnx_detector=ORIGINAL_PALM_ONNX,
        onnx_landmark=LEGACY_FULL_LANDMARK_ONNX,
        om_detector=OPTIMIZED_PALM_OM,
        om_landmark=LEGACY_FULL_LANDMARK_OM,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="data/eval_videos/test.mp4")
    parser.add_argument("--output-root", default="runs/video_onnx_om_compare")
    parser.add_argument("--cases", nargs="+", choices=sorted(CASES), default=["optimized", "original"])
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--save-vis", type=int, default=8)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.3)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--min-hand-score", type=float, default=0.5)
    parser.add_argument("--pipeline-mode", choices=["tracking", "image"], default="tracking")
    parser.add_argument("--match-iou", type=float, default=0.1)
    parser.add_argument("--max-mean-hand21-px", type=float, default=2.0)
    parser.add_argument("--max-p95-hand21-px", type=float, default=5.0)
    parser.add_argument("--max-count-mismatch-rate", type=float, default=0.01)
    parser.add_argument("--keep-detector-loaded", action="store_true")
    return parser.parse_args()


def case_output_dir(output_root: Path, case_name: str, max_frames: int) -> Path:
    suffix = "full" if max_frames == 0 else f"first_{max_frames}"
    return output_root / f"{case_name}_{suffix}"


def run_case(args: argparse.Namespace, case: CheckCase, output_dir: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/compare_video_onnx_om.py",
        "--video",
        args.video,
        "--onnx-detector",
        case.onnx_detector,
        "--onnx-landmark",
        case.onnx_landmark,
        "--om-detector",
        case.om_detector,
        "--om-landmark",
        case.om_landmark,
        "--output-dir",
        str(output_dir),
        "--device-id",
        str(args.device_id),
        "--score-threshold",
        str(args.score_threshold),
        "--nms-iou",
        str(args.nms_iou),
        "--max-det",
        str(args.max_det),
        "--max-hands",
        str(args.max_hands),
        "--min-hand-score",
        str(args.min_hand_score),
        "--pipeline-mode",
        args.pipeline_mode,
        "--match-iou",
        str(args.match_iou),
        "--frame-stride",
        str(args.frame_stride),
        "--start-frame",
        str(args.start_frame),
        "--save-vis",
        str(args.save_vis),
        "--max-mean-hand21-px",
        str(args.max_mean_hand21_px),
        "--max-p95-hand21-px",
        str(args.max_p95_hand21_px),
        "--max-count-mismatch-rate",
        str(args.max_count_mismatch_rate),
    ]
    if args.max_frames:
        cmd.extend(["--max-frames", str(args.max_frames)])
    if not args.keep_detector_loaded:
        cmd.append("--reload-detector-each-frame")

    print(f"[video-check] running {case.name}: {case.description}", flush=True)
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
    summary_path = output_dir / "summary.json"
    summary: dict[str, Any] = {
        "case": case.name,
        "description": case.description,
        "returncode": completed.returncode,
        "summary_path": str(summary_path),
    }
    if summary_path.exists():
        summary.update(json.loads(summary_path.read_text(encoding="utf-8")))
    else:
        summary["consistent"] = False
    return summary


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Video ONNX vs OM Check Summary",
        "",
        "| Case | Consistent | Frames | Matched | Count mismatch | Hand21 mean | Hand21 P95 | Report |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in summaries:
        output_dir = Path(str(item.get("output_dir", "")))
        report = output_dir / "report.md" if output_dir else Path(str(item.get("summary_path", ""))).with_name("report.md")
        lines.append(
            "| "
            f"`{item['case']}` | "
            f"{item.get('consistent')} | "
            f"{item.get('processed_frames', 0)} | "
            f"{item.get('matched_hands', 0)} | "
            f"{fmt(item.get('count_mismatch_rate', 0.0), 6)} | "
            f"{fmt(item.get('hand21_mean_px_mean', 0.0))} | "
            f"{fmt(item.get('hand21_mean_px_p95', 0.0))} | "
            f"`{report}` |"
        )
    lines.append("")
    lines.append("Cases:")
    lines.append("")
    for item in summaries:
        lines.append(f"- `{item['case']}`: {item.get('description', '')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root = output_root if output_root.is_absolute() else PROJECT_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for case_name in args.cases:
        case = CASES[case_name]
        output_dir = case_output_dir(output_root, case.name, args.max_frames)
        summaries.append(run_case(args, case, output_dir))

    summary_path = output_root / "video_onnx_om_checks_summary.json"
    report_path = output_root / "video_onnx_om_checks_summary.md"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(report_path, summaries)
    print(json.dumps({"summary": str(summary_path), "report": str(report_path), "cases": summaries}, ensure_ascii=False, indent=2))
    return 0 if all(item.get("consistent") for item in summaries) else 2


if __name__ == "__main__":
    raise SystemExit(main())
