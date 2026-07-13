# Ascend 20T System Instability Record

## Purpose

This document records the current Ascend 20T board instability observed after a
fresh system flash. It is intended to prevent future debugging from incorrectly
attributing the issue to SSH keys, network configuration, `.bashrc`, or project
Python code.

## Board and Image

| Item | Value |
| --- | --- |
| Board | Orange Pi AI Pro 20T / Ascend 310B 20T-class board |
| User | `HwHiAiUser` |
| Hostname observed | `orangepiaipro-20t` |
| System image | `opiaipro_20t_ubuntu22.04_desktop_aarch64_20250211.img` |
| OS family | Ubuntu 22.04 desktop aarch64 |
| Root partition observed in logs | `/dev/nvme0n1p2` |

The user has confirmed that the board is the 20T version and that the SSD is not
believed to be physically faulty. The diagnosis below therefore focuses on the
current root filesystem state, system image/write process, and board-side system
runtime stability rather than assuming an SSD hardware failure.

## Observed Symptoms

- Interactive SSH login is slow or unstable.
- Sometimes the login shell enters the conda `base` environment correctly, but
  sometimes it does not.
- When `base` is active, `conda info --envs` can still terminate with
  `Segmentation fault (core dumped)`.
- `vi .bashrc` previously failed with `Illegal instruction (core dumped)`.
- The board has emitted broadcast messages that PID 1 `systemd` caught a
  segmentation fault and froze execution.

The following conda state was observed during one successful shell
initialization:

```text
CONDA_DEFAULT_ENV=base
CONDA_SHLVL=1
conda is a function
/usr/local/miniconda3/bin/python
conda info --envs
Segmentation fault (core dumped)
```

This means conda shell initialization can succeed, but the conda executable
path is not reliable enough to complete normal operations.

## Key Log Evidence

### PID 1 Crash

The strongest system-level failure is the `systemd` crash:

```text
systemd[1]: Caught <SEGV>, dumped core
systemd[1]: Freezing execution
```

When PID 1 segfaults, the system cannot be treated as a stable Linux runtime.
SSH, login shell behavior, conda activation, and project execution may all
become secondary symptoms.

### ext4 Filesystem Errors

The kernel reported ext4 errors on the root partition:

```text
EXT4-fs (nvme0n1p2): warning: mounting fs with errors, running e2fsck is recommended
EXT4-fs (nvme0n1p2): error count since last fsck: 31
EXT4-fs (nvme0n1p2): initial error ... htree_dirblock_to_tree ... inode ...
EXT4-fs (nvme0n1p2): last error ... ext4_empty_dir ... inode ...
```

These are filesystem consistency errors, not normal SSH or shell startup
messages. They can affect binaries, shared libraries, Python modules, conda
metadata, user files, and service state.

### Ascend Runtime and LPM Faults

The board also reported repeated Ascend-side low-level faults:

```text
[tzdriver] ... cmd_monitor_tick ... pname=dmp_daemon ... timedif=5100 ms
[DRV_LPM_FAULT] receive fault=0xE3A203
[bbox] blackbox receive [LPM] exception ... exception id [0xa6193215]
[bbox] from_module: [lpm]
[bbox] desc: [lpm get current error]
```

These messages indicate board-side firmware/driver/runtime instability around
the Ascend management path. They should not be explained as SSH-key or
`.bashrc` problems.

### Secondary Noise

The logs also contain repeated crash-handler and peripheral messages:

```text
Process ... (apport) has RLIMIT_CORE set to 1; Aborting core
hisi-i2c ... slave address not acknowledged
DRM/HDMI related warnings and errors
```

These are not the first debugging target. The root filesystem errors, PID 1
crash, and Ascend LPM faults are higher priority.

## Current Diagnosis

The current 20T issue should be treated as a system integrity problem rather
than a project deployment problem.

Current most likely causes:

1. The root filesystem was damaged or left inconsistent during image writing,
   first boot expansion, forced shutdown, or later crashes.
2. The `opiaipro_20t_ubuntu22.04_desktop_aarch64_20250211.img` image file,
   decompression, or flashing process may have produced an inconsistent rootfs.
3. The board image may be correct for the 20T hardware, but the resulting
   installed rootfs still needs offline filesystem repair.
4. If ext4 errors are repaired and do not return, remaining `DRV_LPM_FAULT`
   messages should be handled as an Ascend system image, firmware, driver, or
   board-runtime issue.

This record does not conclude that the SSD hardware is bad. It concludes that
the currently mounted root filesystem is not clean and that the running system
is not stable enough for reliable model validation.

## What Not To Do

- Do not keep debugging SSH key setup while PID 1 and ext4 errors are present.
- Do not run `conda init` as a fix for this issue. The shell hook can load, but
  conda itself has crashed.
- Do not edit `.bashrc`, `/etc/profile`, or other startup files unless there is
  a separate, explicit reason after filesystem stability is restored.
- Do not run project Python scripts on this board as acceptance evidence until
  the system can pass basic filesystem and runtime checks.
- Do not install, upgrade, or remove software on the board as part of this
  diagnosis unless explicitly approved for that exact action.

## Recommended Recovery Procedure

First confirm the mounted root source:

```bash
findmnt -no SOURCE,FSTYPE,OPTIONS /
lsblk -f
```

If the root filesystem is `/dev/nvme0n1p2`, repair it only while it is not
mounted as the active root filesystem. Boot from another Linux system, SD card,
USB rescue environment, or attach the NVMe device to another Linux host.

Read-only check:

```bash
sudo e2fsck -f -n /dev/nvme0n1p2
```

Interactive repair:

```bash
sudo e2fsck -f /dev/nvme0n1p2
```

Automatic repair, only when the user accepts automatic fixes:

```bash
sudo e2fsck -f -y /dev/nvme0n1p2
```

After repair and reboot, collect:

```bash
dmesg -T | egrep -i 'EXT4|I/O error|nvme|systemd\[1\]|segv|MATA0|DRV_LPM|dmp_daemon|tzdriver|panic|oops' | tail -200
journalctl -b -p warning..alert --no-pager | tail -200
```

Then verify conda without changing the environment:

```bash
echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
echo "CONDA_SHLVL=$CONDA_SHLVL"
type conda
which python
python -c "import sys; print(sys.executable); print(sys.version)"
python -c "import ssl, sqlite3, json; print('stdlib ok')"
python -c "import conda; print(conda.__version__); print(conda.__file__)"
python -X faulthandler -m conda info --envs
```

## Acceptance Criteria Before Reusing `ascend20t`

The 20T board should not be used as a reliable project validation target until
all of the following are true:

- No new ext4 errors appear after reboot.
- No new `systemd[1]` segmentation fault appears.
- `conda info --envs` completes without segmentation fault.
- Interactive SSH login reaches a shell consistently.
- The conda `base` environment initializes consistently.
- Ascend runtime checks can run repeatedly without new fatal LPM, driver, or
  daemon failures.

Until then, model and pipeline debugging should continue on `ascend8t`, where
the known `npu-smi Health: Alarm` state is already treated as a board-specific
hardware alarm that can be ignored when inference runs normally.
