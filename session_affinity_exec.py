#!/usr/bin/env python3
"""Apply process-level NUMA affinity, emit evidence, then exec Ray."""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path


def parse_cpu_list(value: str) -> set[int]:
    cpus: set[int] = set()
    for item in value.strip().split(","):
        if "-" in item:
            start, end = (int(part) for part in item.split("-", 1))
            cpus.update(range(start, end + 1))
        elif item:
            cpus.add(int(item))
    if not cpus:
        raise RuntimeError("empty CPU list")
    return cpus


def format_cpu_list(cpus: set[int]) -> str:
    ordered = sorted(cpus)
    ranges: list[str] = []
    start = previous = ordered[0]
    for cpu in ordered[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return "+".join(ranges)


def normalize_pci_address(value: str) -> str:
    domain, bus, device_function = value.lower().split(":")
    return f"{domain[-4:].zfill(4)}:{bus.zfill(2)}:{device_function}"


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: session_affinity_exec.py COMMAND [ARG ...]")
    target = os.environ.get("RAY_BENCH_SESSION_CPU_TARGET", "distributed")
    allowed = set(os.sched_getaffinity(0))
    target_pci = "none"
    target_numa = "all"
    target_cpus = allowed
    if target != "distributed":
        device = (Path("/sys/class/net") / target / "device").resolve(strict=True)
        target_pci = normalize_pci_address(device.name)
        target_numa = (device / "numa_node").read_text().strip()
        target_cpus = parse_cpu_list(
            (
                Path("/sys/devices/system/node")
                / f"node{target_numa}"
                / "cpulist"
            ).read_text()
        )
        if not target_cpus <= allowed:
            raise RuntimeError(
                f"{target} CPUs are not contained in the Slurm CPU allocation"
            )
        os.sched_setaffinity(0, target_cpus)
    effective = set(os.sched_getaffinity(0))
    if effective != target_cpus:
        raise RuntimeError("session CPU affinity does not match its target")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "") or "none"
    hostname = os.environ.get("SLURMD_NODENAME", socket.gethostname())
    print(
        f"SESSION_AFFINITY hostname={hostname} target={target} "
        f"target_pci={target_pci} target_numa={target_numa} "
        f"cpus={format_cpu_list(effective)} "
        f"cuda_visible_devices={visible} status=pass",
        flush=True,
    )
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
