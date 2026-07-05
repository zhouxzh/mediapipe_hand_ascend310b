#!/usr/bin/env python3
"""Print OM model IO metadata through ais_bench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hand_pipeline.runtimes.ascend import AscendModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--device-id", type=int, default=0)
    return parser.parse_args()


def normalize_metadata(value: object) -> object:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def main() -> int:
    args = parse_args()
    model_path = args.model if args.model.is_absolute() else PROJECT_ROOT / args.model
    model = AscendModel(model_path, device_id=args.device_id)
    try:
        print(
            json.dumps(
                {
                    "model": str(model_path),
                    "device_id": args.device_id,
                    "inputs": normalize_metadata(model.inputs),
                    "outputs": normalize_metadata(model.outputs),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

