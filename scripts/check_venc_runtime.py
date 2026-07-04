#!/usr/bin/env python3
"""Check Ascend CANN VENC runtime status.

By default this script only reads runtime/driver state. ``--probe`` creates a
real VENC channel and can trigger driver-side memory pressure on failed CANN 8.0
setups, so it also requires ``--i-understand-venc-probe-risk``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_text(command: list[str], timeout: float = 5.0) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:
        return f"{command[0]} unavailable: {exc}"
    return completed.stdout.strip()


def print_file(path: str, title: str, max_lines: int = 80) -> None:
    print(f"\n== {title} ==")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as file:
            for index, line in enumerate(file):
                if index >= max_lines:
                    print("...")
                    break
                print(line.rstrip())
    except OSError as exc:
        print(f"{path}: {exc}")


def print_meminfo_summary() -> None:
    print("\n== memory summary ==")
    wanted = (
        "MemTotal",
        "MemFree",
        "MemAvailable",
        "CmaTotal",
        "CmaFree",
        "HugePages_Total",
        "HugePages_Free",
        "HugePages_Rsvd",
        "HugePages_Surp",
        "Hugepagesize",
    )
    try:
        with open("/proc/meminfo", "r", encoding="utf-8", errors="replace") as file:
            for line in file:
                if line.startswith(wanted):
                    print(line.rstrip())
    except OSError as exc:
        print(f"/proc/meminfo: {exc}")


def join_usermemory() -> None:
    tasks_path = "/sys/fs/cgroup/memory/usermemory/tasks"
    try:
        with open(tasks_path, "a", encoding="utf-8") as file:
            file.write(f"{os.getpid()}\n")
        print(f"joined {tasks_path}")
    except OSError as exc:
        print(f"join {tasks_path} failed: {exc}")


def show_read_only_status() -> None:
    print("== process ==")
    print(f"python={sys.executable}")
    print(f"pid={os.getpid()}")
    print_file("/proc/self/cgroup", "self cgroup", max_lines=20)

    print("\n== CANN import ==")
    try:
        import acl

        print(f"acl={getattr(acl, '__file__', '?')}")
        print(f"soc={acl.get_soc_name() if hasattr(acl, 'get_soc_name') else '?'}")
        from acl import media

        print(f"has_venc_create={hasattr(media, 'venc_create_channel')}")
        print(f"has_venc_channel_id_setter={hasattr(media, 'venc_set_channel_desc_channel_id')}")
        print(f"has_venc_buf_setter={hasattr(media, 'venc_set_channel_desc_buf_addr')}")
    except Exception as exc:
        print(f"acl import failed: {exc}")

    print("\n== npu-smi ==")
    print(run_text(["bash", "-lc", "source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true; npu-smi info 2>/dev/null || true"]))

    print_meminfo_summary()
    print_file("/proc/umap/venc", "proc umap venc", max_lines=140)
    print_file("/proc/umap/h264e", "proc umap h264e", max_lines=80)
    print_file("/proc/umap/vb", "proc umap vb", max_lines=80)

    print("\n== recent VENC dmesg ==")
    print(run_text(["bash", "-lc", "dmesg | grep -iE 'venc|h264e|h265e|encoder node|rc_' | tail -80"], timeout=5.0))


def run_probe(width: int, height: int, fps: int, bitrate_kbps: int) -> int:
    from webrtc_app.cann_encoder import CannVenc, collect_venc_diagnostics

    print("\n== VENC create probe ==")
    print(f"request={width}x{height}@{fps} bitrate={bitrate_kbps}kbps")
    venc = None
    try:
        venc = CannVenc(width=width, height=height, fps=fps, bitrate=bitrate_kbps)
        print("probe=OK")
        return 0
    except Exception as exc:
        print(f"probe=FAILED: {exc}")
        diagnostics = collect_venc_diagnostics()
        if diagnostics:
            print(f"diagnostics={diagnostics}")
        return 2
    finally:
        if venc is not None:
            venc.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--join-usermemory", action="store_true", help="Join /sys/fs/cgroup/memory/usermemory before checks.")
    parser.add_argument("--probe", action="store_true", help="Create and destroy one CANN VENC channel.")
    parser.add_argument(
        "--i-understand-venc-probe-risk",
        action="store_true",
        help="Required with --probe because failed VENC create can leave driver-side memory pressure.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--bitrate-kbps", type=int, default=4000)
    args = parser.parse_args()

    if args.join_usermemory:
        join_usermemory()
    show_read_only_status()
    if args.probe:
        if not args.i_understand_venc_probe_risk:
            print(
                "\nRefusing --probe without --i-understand-venc-probe-risk. "
                "Use the read-only report first; repeated failed VENC create attempts can increase NPU/CMA memory usage."
            )
            return 3
        return run_probe(args.width, args.height, args.fps, args.bitrate_kbps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
