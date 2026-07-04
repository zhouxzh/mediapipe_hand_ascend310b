#!/usr/bin/env python3
"""Build and optionally run a minimal ACLLite-style CANN VENC probe.

The default mode only writes and compiles the C++ probe. Running the binary
creates a real VENC channel, so it requires an explicit risk flag.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_DIR = ROOT / "build" / "venc_acllite_cpp_probe"


CPP_SOURCE = r"""
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>

#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"

static std::atomic<bool> g_running(true);

static const char *ret_name(aclError ret) {
    return ret == ACL_SUCCESS ? "ACL_SUCCESS" : "ACL_ERROR";
}

static bool check_acl(const char *name, aclError ret) {
    std::cout << name << " -> " << ret << " (" << ret_name(ret) << ")" << std::endl;
    return ret == ACL_SUCCESS;
}

static bool join_usermemory() {
    const char *path = "/sys/fs/cgroup/memory/usermemory/tasks";
    std::ofstream file(path, std::ios::app);
    if (!file.good()) {
        std::cerr << "join usermemory failed: cannot open " << path << std::endl;
        return false;
    }
    file << getpid() << std::endl;
    std::cout << "joined " << path << std::endl;
    return true;
}

static void callback(acldvppPicDesc *input, acldvppStreamDesc *output, void *user_data) {
    (void)user_data;
    uint32_t ret_code = output == nullptr ? 0xffffffffu : acldvppGetStreamDescRetCode(output);
    uint32_t size = output == nullptr ? 0u : acldvppGetStreamDescSize(output);
    std::cout << "callback ret_code=" << ret_code << " size=" << size << std::endl;
    if (input != nullptr) {
        void *input_data = acldvppGetPicDescData(input);
        if (input_data != nullptr) {
            (void)acldvppFree(input_data);
        }
        (void)acldvppDestroyPicDesc(input);
    }
}

static void *report_thread(void *arg) {
    aclrtContext context = reinterpret_cast<aclrtContext>(arg);
    if (context == nullptr) {
        std::cerr << "report thread got null ACL context" << std::endl;
        return reinterpret_cast<void *>(-1);
    }
    aclError ret = aclrtSetCurrentContext(context);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtSetCurrentContext in report thread failed: " << ret << std::endl;
        return reinterpret_cast<void *>(-1);
    }
    while (g_running.load()) {
        (void)aclrtProcessReport(1000);
    }
    return nullptr;
}

static acldvppStreamFormat parse_profile(const std::string &profile) {
    if (profile == "baseline" || profile == "base" || profile == "1") {
        return H264_BASELINE_LEVEL;
    }
    if (profile == "high" || profile == "3") {
        return H264_HIGH_LEVEL;
    }
    return H264_MAIN_LEVEL;
}

int main(int argc, char **argv) {
    uint32_t width = 640;
    uint32_t height = 480;
    uint32_t fps = 30;
    uint32_t bitrate = 10000;
    uint32_t key_frame_interval = 16;
    uint32_t rc_mode = 2;
    bool set_src_rate = false;
    bool do_join_usermemory = false;
    std::string profile = "main";

    for (int index = 1; index < argc; ++index) {
        std::string arg = argv[index];
        auto require_value = [&](const std::string &name) -> const char * {
            if (index + 1 >= argc) {
                std::cerr << name << " requires a value" << std::endl;
                std::exit(2);
            }
            return argv[++index];
        };
        if (arg == "--width") width = static_cast<uint32_t>(std::stoul(require_value(arg)));
        else if (arg == "--height") height = static_cast<uint32_t>(std::stoul(require_value(arg)));
        else if (arg == "--fps") fps = static_cast<uint32_t>(std::stoul(require_value(arg)));
        else if (arg == "--bitrate-kbps") bitrate = static_cast<uint32_t>(std::stoul(require_value(arg)));
        else if (arg == "--key-frame-interval") key_frame_interval = static_cast<uint32_t>(std::stoul(require_value(arg)));
        else if (arg == "--rc-mode") rc_mode = static_cast<uint32_t>(std::stoul(require_value(arg)));
        else if (arg == "--profile") profile = require_value(arg);
        else if (arg == "--set-src-rate") set_src_rate = true;
        else if (arg == "--join-usermemory") do_join_usermemory = true;
        else {
            std::cerr << "unknown argument: " << arg << std::endl;
            return 2;
        }
    }

    if (do_join_usermemory) {
        (void)join_usermemory();
    }

    std::cout << "probe request width=" << width
              << " height=" << height
              << " fps=" << fps
              << " bitrate=" << bitrate
              << " profile=" << profile
              << " set_src_rate=" << (set_src_rate ? "true" : "false")
              << std::endl;

    aclrtContext context = nullptr;
    pthread_t thread_id = 0;
    aclvencChannelDesc *channel_desc = nullptr;
    aclvencFrameConfig *frame_config = nullptr;
    aclrtStream stream = nullptr;
    bool channel_created = false;
    int exit_code = 0;

    if (!check_acl("aclInit", aclInit(nullptr))) return 10;
    if (!check_acl("aclrtSetDevice", aclrtSetDevice(0))) return 11;
    if (!check_acl("aclrtCreateContext", aclrtCreateContext(&context, 0))) return 12;
    if (!check_acl("aclrtSetCurrentContext", aclrtSetCurrentContext(context))) return 13;

    int pthread_ret = pthread_create(&thread_id, nullptr, report_thread, context);
    if (pthread_ret != 0) {
        std::cerr << "pthread_create failed: " << pthread_ret << std::endl;
        exit_code = 14;
        goto cleanup;
    }
    std::cout << "pthread_create -> 0 thread_id=" << static_cast<uint64_t>(thread_id) << std::endl;

    channel_desc = aclvencCreateChannelDesc();
    if (channel_desc == nullptr) {
        std::cerr << "aclvencCreateChannelDesc failed" << std::endl;
        exit_code = 15;
        goto cleanup;
    }
    std::cout << "aclvencCreateChannelDesc -> OK" << std::endl;

    check_acl("aclvencSetChannelDescThreadId", aclvencSetChannelDescThreadId(channel_desc, static_cast<uint64_t>(thread_id)));
    check_acl("aclvencSetChannelDescCallback", aclvencSetChannelDescCallback(channel_desc, callback));
    check_acl("aclvencSetChannelDescEnType", aclvencSetChannelDescEnType(channel_desc, parse_profile(profile)));
    check_acl("aclvencSetChannelDescPicFormat", aclvencSetChannelDescPicFormat(channel_desc, PIXEL_FORMAT_YUV_SEMIPLANAR_420));
    check_acl("aclvencSetChannelDescPicWidth", aclvencSetChannelDescPicWidth(channel_desc, width));
    check_acl("aclvencSetChannelDescPicHeight", aclvencSetChannelDescPicHeight(channel_desc, height));
    check_acl("aclvencSetChannelDescKeyFrameInterval", aclvencSetChannelDescKeyFrameInterval(channel_desc, key_frame_interval));
    check_acl("aclvencSetChannelDescRcMode", aclvencSetChannelDescRcMode(channel_desc, rc_mode));
    check_acl("aclvencSetChannelDescMaxBitRate", aclvencSetChannelDescMaxBitRate(channel_desc, bitrate));
    if (set_src_rate) {
        check_acl("aclvencSetChannelDescSrcRate", aclvencSetChannelDescSrcRate(channel_desc, fps));
    }

    std::cout << "desc entype=" << aclvencGetChannelDescEnType(channel_desc)
              << " format=" << aclvencGetChannelDescPicFormat(channel_desc)
              << " width=" << aclvencGetChannelDescPicWidth(channel_desc)
              << " height=" << aclvencGetChannelDescPicHeight(channel_desc)
              << " key_interval=" << aclvencGetChannelDescKeyFrameInterval(channel_desc)
              << " rc_mode=" << aclvencGetChannelDescRcMode(channel_desc)
              << " max_bitrate=" << aclvencGetChannelDescMaxBitRate(channel_desc)
              << " src_rate=" << aclvencGetChannelDescSrcRate(channel_desc)
              << std::endl;

    {
        aclError ret = aclvencCreateChannel(channel_desc);
        std::cout << "aclvencCreateChannel -> " << ret << " (" << ret_name(ret) << ")" << std::endl;
        if (ret != ACL_SUCCESS) {
            exit_code = 20;
            goto cleanup;
        }
    }
    channel_created = true;

    if (!check_acl("aclrtCreateStream", aclrtCreateStream(&stream))) {
        exit_code = 21;
        goto cleanup;
    }
    if (!check_acl("aclrtSubscribeReport", aclrtSubscribeReport(static_cast<uint64_t>(thread_id), stream))) {
        exit_code = 22;
        goto cleanup;
    }

    frame_config = aclvencCreateFrameConfig();
    if (frame_config == nullptr) {
        std::cerr << "aclvencCreateFrameConfig failed" << std::endl;
        exit_code = 23;
        goto cleanup;
    }
    check_acl("aclvencSetFrameConfigEos", aclvencSetFrameConfigEos(frame_config, 0));
    check_acl("aclvencSetFrameConfigForceIFrame", aclvencSetFrameConfigForceIFrame(frame_config, 1));
    std::cout << "probe=OK" << std::endl;

cleanup:
    if (frame_config != nullptr) {
        (void)aclvencDestroyFrameConfig(frame_config);
    }
    if (stream != nullptr) {
        (void)aclrtUnSubscribeReport(static_cast<uint64_t>(thread_id), stream);
        (void)aclrtDestroyStream(stream);
    }
    if (channel_desc != nullptr) {
        if (channel_created) {
            (void)aclvencDestroyChannel(channel_desc);
        }
        (void)aclvencDestroyChannelDesc(channel_desc);
    }
    g_running.store(false);
    if (thread_id != 0) {
        (void)pthread_join(thread_id, nullptr);
    }
    if (context != nullptr) {
        (void)aclrtDestroyContext(context);
    }
    (void)aclrtResetDevice(0);
    (void)aclFinalize();
    return exit_code;
}
"""


def run(command: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=cwd, env=env, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def source_cann_env() -> dict[str, str]:
    env = dict(os.environ)
    ascend_home = env.get("ASCEND_HOME") or env.get("ASCEND_TOOLKIT_HOME") or "/usr/local/Ascend/ascend-toolkit/latest"
    env["ASCEND_HOME"] = ascend_home
    lib_paths = [
        str(Path(ascend_home) / "runtime" / "lib64"),
        str(Path(ascend_home) / "aarch64-linux" / "lib64"),
    ]
    current = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join([*lib_paths, current]) if current else ":".join(lib_paths)
    return env


def print_text(text: str) -> None:
    if text:
        print(text.rstrip())


def build_probe(build_dir: Path) -> Path:
    env = source_cann_env()
    ascend_home = Path(env["ASCEND_HOME"])
    build_dir.mkdir(parents=True, exist_ok=True)
    source_path = build_dir / "venc_acllite_probe.cpp"
    binary_path = build_dir / "venc_acllite_probe"
    source_path.write_text(CPP_SOURCE, encoding="utf-8")

    include_dirs = [
        ascend_home / "include",
        ascend_home / "runtime" / "include",
    ]
    lib_dirs = [
        ascend_home / "runtime" / "lib64",
        ascend_home / "aarch64-linux" / "lib64",
    ]
    command = ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-pthread", "-DENABLE_DVPP_INTERFACE"]
    for include_dir in include_dirs:
        command.extend(["-I", str(include_dir)])
    command.extend([str(source_path), "-o", str(binary_path)])
    for lib_dir in lib_dirs:
        command.extend(["-L", str(lib_dir), f"-Wl,-rpath,{lib_dir}"])
    command.extend(["-lascendcl", "-lacl_dvpp"])

    completed = run(command, env=env)
    print_text(completed.stdout)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    print(f"built={binary_path}")
    return binary_path


def show_status(title: str) -> None:
    print(f"\n== {title} ==")
    for command in [
        ["bash", "-lc", "source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true; npu-smi info 2>/dev/null || true"],
        ["bash", "-lc", "grep -E 'MemFree|MemAvailable|CmaTotal|CmaFree|HugePages_Total|HugePages_Free' /proc/meminfo || true"],
        ["bash", "-lc", "cat /proc/umap/venc 2>/dev/null | sed -n '/Detail Venc Chn Id Info/,$p' | head -20 || true"],
    ]:
        completed = run(command)
        print_text(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=DEFAULT_BUILD_DIR, type=Path)
    parser.add_argument("--run", action="store_true", help="Run the compiled probe after building.")
    parser.add_argument(
        "--i-understand-venc-probe-risk",
        action="store_true",
        help="Required with --run because failed VENC create can increase driver-side memory usage.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--bitrate-kbps", type=int, default=10000)
    parser.add_argument("--profile", default="main", choices=["baseline", "base", "main", "high", "1", "2", "3"])
    parser.add_argument("--key-frame-interval", type=int, default=16)
    parser.add_argument("--rc-mode", type=int, default=2)
    parser.add_argument("--set-src-rate", action="store_true")
    parser.add_argument("--join-usermemory", action="store_true")
    args = parser.parse_args()

    binary_path = build_probe(args.build_dir)
    if not args.run:
        print("build-only=OK")
        return 0
    if not args.i_understand_venc_probe_risk:
        print("Refusing --run without --i-understand-venc-probe-risk.")
        return 3

    show_status("before")
    command = [
        str(binary_path),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--fps",
        str(args.fps),
        "--bitrate-kbps",
        str(args.bitrate_kbps),
        "--profile",
        args.profile,
        "--key-frame-interval",
        str(args.key_frame_interval),
        "--rc-mode",
        str(args.rc_mode),
    ]
    if args.set_src_rate:
        command.append("--set-src-rate")
    if args.join_usermemory:
        command.append("--join-usermemory")
    completed = run(command, env=source_cann_env())
    print_text(completed.stdout)
    show_status("after")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
