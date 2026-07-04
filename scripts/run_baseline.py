#!/usr/bin/env python3
"""Run the project baseline verification with plain Python subprocesses."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARENT_ROOT = PROJECT_ROOT.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def parse_command(value: str) -> list[str]:
    if not value:
        return [sys.executable]
    return shlex.split(value, posix=(os.name != "nt"))


def python_command(command: str, conda_env: str) -> list[str]:
    if conda_env:
        return ["conda", "run", "-n", conda_env, "python"]
    return parse_command(command)


def resolve_existing(candidates: list[Path], label: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    tried = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Cannot find {label}. Tried:\n  {tried}")


def is_under(path: Path, parent: Path) -> bool:
    path_text = os.path.normcase(str(path.resolve()))
    parent_text = os.path.normcase(str(parent.resolve()))
    return path_text == parent_text or path_text.startswith(parent_text + os.sep)


def run_step(name: str, python: list[str], script: str, args: list[str]) -> None:
    command = python + [str(SCRIPTS_DIR / script), *args]
    print()
    print(f"[{name}] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default="", help="Python command for current TFLite scripts.")
    parser.add_argument("--legacy-python", default="", help="Python command for legacy MediaPipe graph scripts.")
    parser.add_argument("--conda-env", default="", help="Optional conda env for current TFLite scripts.")
    parser.add_argument("--legacy-conda-env", default="", help="Optional conda env for legacy MediaPipe graph scripts.")
    parser.add_argument("--data", default="", help="Palm dataset directory.")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"], help="Palm dataset split.")
    parser.add_argument("--handlm-data", default="", help="Manual hand landmark dataset directory.")
    parser.add_argument("--current-reference", default="", help="Current MediaPipe Tasks reference JSON.")
    parser.add_argument("--output-root", default="", help="Output directory. Defaults to runs/baseline.")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--save-vis", type=int, default=0)
    parser.add_argument("--skip-legacy-graph", action="store_true")
    parser.add_argument("--run-matrix", action="store_true")
    parser.add_argument("--no-clean", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = build_args()

    data = (
        (Path(args.data) if Path(args.data).is_absolute() else PROJECT_ROOT / args.data).resolve()
        if args.data
        else resolve_existing(
            [
                PROJECT_ROOT / "data" / "palm_datasets",
                PARENT_ROOT / "data" / "palm_datasets",
            ],
            "palm dataset",
        )
    )
    handlm_data = (
        (Path(args.handlm_data) if Path(args.handlm_data).is_absolute() else PROJECT_ROOT / args.handlm_data).resolve()
        if args.handlm_data
        else resolve_existing(
            [
                PROJECT_ROOT / "data" / "handlm_datasets",
                PARENT_ROOT / "data" / "handlm_datasets",
            ],
            "manual hand landmark dataset",
        )
    )
    current_reference = (
        (Path(args.current_reference) if Path(args.current_reference).is_absolute() else PROJECT_ROOT / args.current_reference).resolve()
        if args.current_reference
        else resolve_existing(
            [
                PROJECT_ROOT / "references" / "current_tasks" / "mediapipe_predictions.json",
                PROJECT_ROOT / "runs" / "mediapipe_baseline_vs_om" / "mediapipe_predictions.json",
                PARENT_ROOT / "runs" / "mediapipe_baseline_vs_om" / "mediapipe_predictions.json",
            ],
            "current MediaPipe reference predictions",
        )
    )

    split_image_dir = data / args.split / "images"
    split_label_dir = data / args.split / "labels"
    if not split_image_dir.exists():
        raise FileNotFoundError(f"Cannot find palm dataset split image directory: {split_image_dir}")
    if not split_label_dir.exists():
        raise FileNotFoundError(f"Cannot find palm dataset split label directory: {split_label_dir}")

    output_root = (
        (Path(args.output_root) if Path(args.output_root).is_absolute() else PROJECT_ROOT / args.output_root).resolve()
        if args.output_root
        else PROJECT_ROOT / "runs" / "baseline"
    )
    runs_root = PROJECT_ROOT / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    if output_root.exists() and not args.no_clean:
        if output_root.resolve() == runs_root.resolve():
            raise RuntimeError(f"Refusing to clean the whole runs directory: {output_root}")
        if not is_under(output_root, runs_root):
            raise RuntimeError(f"Refusing to clean output outside runs/: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_config = {
        "project_root": str(PROJECT_ROOT),
        "output_root": str(output_root),
        "palm_dataset": str(data),
        "palm_split": args.split,
        "palm_split_images": str(split_image_dir),
        "palm_split_labels": str(split_label_dir),
        "handlm_dataset": str(handlm_data),
        "handlm_split": None,
        "current_reference": str(current_reference),
        "reference_alignment_note": "Current and legacy MediaPipe references must be generated from the same palm dataset split.",
        "run_matrix": bool(args.run_matrix),
        "max_images": args.max_images,
        "save_vis": args.save_vis,
    }
    (output_root / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    current_python = python_command(args.python, args.conda_env)
    legacy_python = python_command(args.legacy_python or args.python, args.legacy_conda_env)
    max_image_args = ["--max-images", str(args.max_images)] if args.max_images > 0 else []
    save_vis_args = ["--save-vis", str(args.save_vis)]

    run_step(
        "inspect",
        current_python,
        "inspect_tflite.py",
        ["--model-dir", "models/tflite", "--output", str(output_root / "model_info.json")],
    )
    run_step(
        "palm",
        current_python,
        "eval_palm_tflite.py",
        [
            "--data",
            str(data),
            "--split",
            args.split,
            "--model",
            "models/tflite/mediapipe_legacy_0_10_14_palm_detection_full.tflite",
            "--official-mediapipe",
            str(current_reference),
            "--output-dir",
            str(output_root / "palm_detector"),
            *save_vis_args,
            *max_image_args,
        ],
    )
    run_step(
        "two_stage_current",
        current_python,
        "eval_two_stage_tflite.py",
        [
            "--data",
            str(data),
            "--split",
            args.split,
            "--detector",
            "models/tflite/mediapipe_task_hand_detector_full.tflite",
            "--landmark",
            "models/tflite/mediapipe_task_hand_landmark_full.tflite",
            "--official-mediapipe",
            str(current_reference),
            "--output-dir",
            str(output_root / "two_stage_vs_current_tasks"),
            *save_vis_args,
            *max_image_args,
        ],
    )
    run_step(
        "handlm_manual_gt",
        current_python,
        "eval_handlm_tflite.py",
        [
            "--data",
            str(handlm_data),
            "--output-dir",
            str(output_root / "handlm_manual_gt"),
            *save_vis_args,
            *max_image_args,
        ],
    )

    legacy_predictions = output_root / "legacy_graph" / "legacy_hand_predictions.json"
    if not args.skip_legacy_graph:
        run_step(
            "legacy_graph",
            legacy_python,
            "eval_legacy_graph.py",
            [
                "--data",
                str(data),
                "--split",
                args.split,
                "--current-mediapipe",
                str(current_reference),
                "--two-stage",
                str(output_root / "two_stage_vs_current_tasks" / "predictions.json"),
                "--output-dir",
                str(output_root / "legacy_graph"),
                *save_vis_args,
                *max_image_args,
            ],
        )
    elif not legacy_predictions.exists():
        raise FileNotFoundError(f"--skip-legacy-graph was set, but {legacy_predictions} does not exist.")

    run_step(
        "two_stage_legacy",
        current_python,
        "eval_two_stage_tflite.py",
        [
            "--data",
            str(data),
            "--split",
            args.split,
            "--detector",
            "models/tflite/mediapipe_legacy_0_10_14_palm_detection_full.tflite",
            "--landmark",
            "models/tflite/mediapipe_legacy_0_10_14_hand_landmark_full.tflite",
            "--official-mediapipe",
            str(legacy_predictions),
            "--legacy-rects",
            str(legacy_predictions),
            "--output-dir",
            str(output_root / "two_stage_vs_legacy_graph"),
            *save_vis_args,
            *max_image_args,
        ],
    )
    run_step(
        "legacy_rect",
        current_python,
        "eval_legacy_rect_tflite.py",
        [
            "--legacy-predictions",
            str(legacy_predictions),
            "--landmark",
            "models/tflite/mediapipe_legacy_0_10_14_hand_landmark_full.tflite",
            "--output-dir",
            str(output_root / "legacy_rect_landmark"),
        ],
    )

    if args.run_matrix:
        run_step(
            "matrix",
            current_python,
            "eval_tflite_matrix.py",
            [
                "--data",
                str(data),
                "--split",
                args.split,
                "--reference-current",
                str(current_reference),
                "--reference-legacy-full",
                str(legacy_predictions),
                "--output-dir",
                str(output_root / "tflite_matrix"),
                *save_vis_args,
                *max_image_args,
            ],
        )

    run_step(
        "summary",
        current_python,
        "summarize_baseline.py",
        ["--output-root", str(output_root)],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
