#!/usr/bin/env python3
"""Validate benchmark logs and publish canonical artifacts."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any


SIZES_MIB = (1, 64, 1024)
FLOW_COUNTS = (1, 2, 4, 8)
NIXL_FLOW_COUNTS = (1, 2, 4)
NIXL_FOUR_RAIL_FLOW_COUNTS = (1, 2)
DEFAULT_WARMUPS = 3
DEFAULT_ITERATIONS = 10
SERIES = (
    ("object", "cpu"),
    ("nixl", "cpu"),
    ("object", "gpu"),
    ("nccl", "gpu"),
)
FLOW_COUNTS_BY_SERIES = {
    series: NIXL_FLOW_COUNTS if series == ("nixl", "cpu") else FLOW_COUNTS
    for series in SERIES
}
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PCI_ADDRESS = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
CPU_RANGE = re.compile(r"^[0-9]+(?:-[0-9]+)?$")
PATH_EXPECTATIONS = {
    ("object", "cpu"): ("tcp", "ray-object-manager"),
    ("nixl", "cpu"): ("cxi", "libfabric"),
    ("object", "gpu"): ("tcp", "ray-object-manager"),
    ("nccl", "gpu"): ("cxi", "aws-ofi-nccl"),
}
TRANSPORT_DEVICES = {
    "nixl": ("cxi3", "cxi2", "cxi1", "cxi0"),
    "nccl": ("cxi3", "cxi2", "cxi1", "cxi0"),
}
LEGACY_OUT_OF_SCOPE_CASES = frozenset(
    {
        "object-cpu-single-nic-1024mib-16f",
        "object-cuda-single-nic-1024mib-16f",
        "nccl-gpu-single-nic-1024mib-1n-16fpn",
        "nixl-cpu-four-rail-1024mib-4n-4fpn",
    }
)


class ResultError(RuntimeError):
    """A benchmark log or result matrix is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    publish = subparsers.add_parser("publish")
    publish.add_argument("logs", nargs="+", type=Path)
    publish.add_argument("--csv", type=Path, required=True)
    publish.add_argument("--baseline-svg", type=Path, required=True)
    publish.add_argument("--flows-svg", type=Path, required=True)
    publish.add_argument("--nics-svg", type=Path, required=True)
    return parser.parse_args()


def read_log(path: Path) -> str:
    try:
        return ANSI_ESCAPE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    except OSError as exc:
        raise ResultError(f"could not read {path}: {exc}") from exc



def records(text: str, marker: str) -> list[str]:
    found: list[str] = []
    for line in text.splitlines():
        offset = line.find(marker)
        if offset >= 0 and (offset == 0 or line[offset - 1].isspace()):
            found.append(line[offset:].strip())
    return found


def fields(record: str, source: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for token in record.split()[1:]:
        if "=" not in token:
            raise ResultError(f"{source}: malformed record token {token!r}")
        name, value = token.split("=", 1)
        if not name or not value or name in parsed:
            raise ResultError(f"{source}: invalid record field {token!r}")
        parsed[name] = value
    return parsed


def require(mapping: dict[str, str], name: str, source: Path) -> str:
    try:
        return mapping[name]
    except KeyError as exc:
        raise ResultError(f"{source}: record is missing {name}") from exc


def positive_int(mapping: dict[str, str], name: str, source: Path) -> int:
    try:
        value = int(require(mapping, name, source))
    except ValueError as exc:
        raise ResultError(f"{source}: {name} is not an integer") from exc
    if value <= 0:
        raise ResultError(f"{source}: {name} must be positive")
    return value


def nonnegative_int(mapping: dict[str, str], name: str, source: Path) -> int:
    try:
        value = int(require(mapping, name, source))
    except ValueError as exc:
        raise ResultError(f"{source}: {name} is not an integer") from exc
    if value < 0:
        raise ResultError(f"{source}: {name} must be nonnegative")
    return value


def positive_float(mapping: dict[str, str], name: str, source: Path) -> float:
    try:
        value = float(require(mapping, name, source))
    except ValueError as exc:
        raise ResultError(f"{source}: {name} is not numeric") from exc
    if not math.isfinite(value) or value <= 0:
        raise ResultError(f"{source}: {name} must be positive and finite")
    return value


def comma_values(
    mapping: dict[str, str], name: str, source: Path
) -> tuple[str, ...]:
    values = tuple(require(mapping, name, source).split(","))
    if not values or any(not value for value in values):
        raise ResultError(f"{source}: {name} contains an empty value")
    return values


def numa_values(
    mapping: dict[str, str], name: str, source: Path
) -> tuple[int, ...]:
    values: list[int] = []
    for item in comma_values(mapping, name, source):
        try:
            value = int(item)
        except ValueError as exc:
            raise ResultError(f"{source}: {name} is not an integer list") from exc
        if value < 0:
            raise ResultError(f"{source}: {name} must be nonnegative")
        values.append(value)
    return tuple(values)


def parse_cpu_set(value: str, source: Path) -> set[int]:
    items = value.split("+")
    if not items or any(not CPU_RANGE.fullmatch(item) for item in items):
        raise ResultError(f"{source}: invalid CPU set {value!r}")
    cpus: set[int] = set()
    for item in items:
        if "-" in item:
            start, end = (int(part) for part in item.split("-", 1))
            if end < start:
                raise ResultError(f"{source}: invalid CPU range {item!r}")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(item))
    return cpus


def validate_transport_log(
    path: Path, text: str, transport: str
) -> dict[str, tuple[str, int, str]]:
    networks = [fields(item, path) for item in records(text, "NETWORK ")]
    if len(networks) != 2 or any(
        item.get("interface") != "hsn0" or item.get("status") != "pass"
        for item in networks
    ):
        raise ResultError(f"{path}: both Ray nodes were not proven to use hsn0")
    if len({item.get("hostname") for item in networks}) != 2 or len(
        {item.get("ray_ip") for item in networks}
    ) != 2:
        raise ResultError(f"{path}: NETWORK records do not identify two nodes")
    network_topologies: set[tuple[str, int, str]] = set()
    for item in networks:
        pci = require(item, "pci", path)
        if not PCI_ADDRESS.fullmatch(pci):
            raise ResultError(f"{path}: NETWORK record has invalid PCI address")
        numa = nonnegative_int(item, "numa", path)
        cpus = require(item, "cpus", path)
        parse_cpu_set(cpus, path)
        network_topologies.add((pci, numa, cpus))
    if len(network_topologies) != 1:
        raise ResultError(f"{path}: hsn0 topology differs between Ray nodes")
    session_affinities = [
        fields(item, path) for item in records(text, "SESSION_AFFINITY ")
    ]
    if len(session_affinities) != 2 or len(
        {item.get("hostname") for item in session_affinities}
    ) != 2:
        raise ResultError(f"{path}: session affinity was not proven on both nodes")
    for item in session_affinities:
        if item.get("status") != "pass":
            raise ResultError(f"{path}: session affinity did not pass")
        cpus = require(item, "cpus", path)
        parse_cpu_set(cpus, path)
        if transport == "object":
            pci, numa, target_cpus = next(iter(network_topologies))
            if (
                item.get("target") != "hsn0"
                or item.get("target_pci") != pci
                or item.get("target_numa") != str(numa)
                or cpus != target_cpus
                or item.get("cuda_visible_devices") != "3"
            ):
                raise ResultError(f"{path}: Object Store session is not hsn0-local")
        else:
            expected_visible = "none" if transport == "nixl" else "0,1,2,3"
            if (
                item.get("target") != "distributed"
                or item.get("target_pci") != "none"
                or item.get("target_numa") != "all"
                or item.get("cuda_visible_devices") != expected_visible
            ):
                raise ResultError(f"{path}: {transport} session visibility is wrong")
    shutdowns = [fields(item, path) for item in records(text, "SHUTDOWN ")]
    if (
        len(shutdowns) != 1
        or shutdowns[0].get("status") != "clean"
        or shutdowns[0].get("transport") != transport
    ):
        raise ResultError(f"{path}: driver shutdown was not clean")
    if transport == "nccl":
        required_patterns = (
            r"NET/OFI.*Selected Provider is cxi",
            r"Channel .*NET/AWS Libfabric/.*/GDRDMA/",
            r"GPU Direct RDMA Enabled .*read 1,",
        )
        for pattern in required_patterns:
            if not re.search(pattern, text, flags=re.IGNORECASE):
                raise ResultError(f"{path}: missing NCCL evidence matching {pattern}")
        if re.search(
            r"Channel .*NET/AWS Libfabric/[0-9]+/Shared(?:\s|$)",
            text,
            flags=re.IGNORECASE,
        ):
            raise ResultError(f"{path}: NCCL used a plain host-staged Shared path")
    elif transport == "nixl":
        if "NIXL_AGENT backend=LIBFABRIC" not in text:
            raise ResultError(f"{path}: NIXL LIBFABRIC initialization is missing")
        if re.search(r"Selected backend:\s*UCX", text, flags=re.IGNORECASE):
            raise ResultError(f"{path}: NIXL selected UCX")
    return {"hsn0": next(iter(network_topologies))}


def validate_affinity_record(
    source: Path,
    record: dict[str, str],
    path_record: dict[str, str],
    transport: str,
    device: str,
    nic_count: int,
    total_flows: int,
    topology_by_target: dict[str, tuple[str, int, str]],
    gpu_by_target: dict[str, tuple[str, int]],
) -> None:
    if record.get("status") != "pass":
        raise ResultError(f"{source}: affinity validation did not pass")
    if record.get("transport") != transport or record.get("device") != device:
        raise ResultError(f"{source}: affinity transport/device is inconsistent")
    if positive_int(record, "actors", source) != 2 * total_flows:
        raise ResultError(f"{source}: affinity actor count is inconsistent")

    expected_targets = tuple(require(path_record, "devices", source).split(","))
    targets = comma_values(record, "targets", source)
    if targets != expected_targets or len(targets) != nic_count:
        raise ResultError(f"{source}: affinity targets do not match PATH devices")
    target_pcis = comma_values(record, "target_pci", source)
    target_numas = numa_values(record, "target_numa", source)
    target_cpu_sets = tuple(require(record, "target_cpus", source).split("|"))
    policy = record.get("policy", "local")
    if path_record.get("rail_policy") == "striped":
        if policy != "distributed":
            raise ResultError(
                f"{source}: striped NIXL actors are not distributed across NUMA"
            )
        if not (
            len(target_pcis)
            == len(target_numas)
            == len(target_cpu_sets)
            == nic_count
        ):
            raise ResultError(
                f"{source}: striped affinity topology field counts differ"
            )
        cpu_numas_raw = comma_values(record, "cpu_numa", source)
        cpu_sets = tuple(require(record, "cpu_sets", source).split("|"))
        if cpu_numas_raw != ("all",) or len(cpu_sets) != 1:
            raise ResultError(
                f"{source}: striped NIXL actors do not report one distributed CPU set"
            )
        effective_cpu_ids = parse_cpu_set(cpu_sets[0], source)
        target_cpu_ids: set[int] = set()
        for target, pci, numa, target_cpus in zip(
            targets,
            target_pcis,
            target_numas,
            target_cpu_sets,
            strict=True,
        ):
            if not PCI_ADDRESS.fullmatch(pci):
                raise ResultError(
                    f"{source}: affinity target PCI address is invalid"
                )
            target_cpu_ids.update(parse_cpu_set(target_cpus, source))
            topology = (pci, numa, target_cpus)
            previous = topology_by_target.setdefault(target, topology)
            if previous != topology:
                raise ResultError(
                    f"{source}: topology for {target} is inconsistent"
                )
        if effective_cpu_ids != target_cpu_ids:
            raise ResultError(
                f"{source}: striped NIXL actors do not span all rail NUMA CPUs"
            )
        if (
            comma_values(record, "gpu_numa", source) != ("none",)
            or comma_values(record, "gpu_pci", source) != ("none",)
        ):
            raise ResultError(
                f"{source}: striped CPU affinity unexpectedly reports a GPU"
            )
        return

    if policy != "local":
        raise ResultError(f"{source}: locally pinned actors report {policy=}")
    cpu_numas = numa_values(record, "cpu_numa", source)
    cpu_sets = tuple(require(record, "cpu_sets", source).split("|"))
    if not (
        len(target_pcis)
        == len(target_numas)
        == len(target_cpu_sets)
        == len(cpu_numas)
        == len(cpu_sets)
        == nic_count
    ):
        raise ResultError(f"{source}: affinity topology field counts differ")
    if target_numas != cpu_numas:
        raise ResultError(f"{source}: actor CPU NUMA does not match its target")

    for target, pci, numa, target_cpus, cpus in zip(
        targets,
        target_pcis,
        target_numas,
        target_cpu_sets,
        cpu_sets,
        strict=True,
    ):
        if not PCI_ADDRESS.fullmatch(pci):
            raise ResultError(f"{source}: affinity target PCI address is invalid")
        target_cpu_ids = parse_cpu_set(target_cpus, source)
        effective_cpu_ids = parse_cpu_set(cpus, source)
        if not effective_cpu_ids <= target_cpu_ids:
            raise ResultError(f"{source}: actor CPU set escaped its target NUMA")
        topology = (pci, numa, target_cpus)
        previous = topology_by_target.setdefault(target, topology)
        if previous != topology:
            raise ResultError(f"{source}: topology for {target} is inconsistent")

    gpu_numas_raw = comma_values(record, "gpu_numa", source)
    gpu_pcis = comma_values(record, "gpu_pci", source)
    if device == "cpu":
        if gpu_numas_raw != ("none",) or gpu_pcis != ("none",):
            raise ResultError(f"{source}: CPU affinity unexpectedly reports a GPU")
        return

    gpu_numas = numa_values(record, "gpu_numa", source)
    if len(gpu_numas) != nic_count or len(gpu_pcis) != nic_count:
        raise ResultError(f"{source}: GPU affinity field counts differ")
    if gpu_numas != target_numas:
        raise ResultError(f"{source}: GPU and network NUMA domains differ")
    if len(set(gpu_pcis)) != nic_count:
        raise ResultError(f"{source}: distinct targets did not use distinct GPUs")
    for target, pci, numa in zip(targets, gpu_pcis, gpu_numas, strict=True):
        if not PCI_ADDRESS.fullmatch(pci):
            raise ResultError(f"{source}: affinity GPU PCI address is invalid")
        gpu = (pci, numa)
        previous = gpu_by_target.setdefault(target, gpu)
        if previous != gpu:
            raise ResultError(f"{source}: GPU for {target} is inconsistent")


def validate_path_record(
    source: Path,
    record: dict[str, str],
    transport: str,
    device: str,
    nic_count: int,
) -> None:
    series = (transport, device)
    try:
        expected_network, expected_provider = PATH_EXPECTATIONS[series]
    except KeyError as exc:
        raise ResultError(f"{source}: unsupported result series {series}") from exc
    if require(record, "payload_network", source) != expected_network:
        raise ResultError(f"{source}: {series} has the wrong payload network")
    if require(record, "provider", source) != expected_provider:
        raise ResultError(f"{source}: {series} has the wrong provider")

    devices = tuple(require(record, "devices", source).split(","))
    if transport == "object":
        if nic_count != 1 or devices != ("hsn0",):
            raise ResultError(f"{source}: Object Store path must use hsn0")
        expected_staging = "host" if device == "gpu" else "none"
        if record.get("staging") != expected_staging:
            raise ResultError(f"{source}: Object Store staging is inconsistent")
    else:
        expected_devices = TRANSPORT_DEVICES[transport][:nic_count]
        if devices != expected_devices:
            raise ResultError(
                f"{source}: {transport} devices {devices} do not match "
                f"{expected_devices}"
            )
    if record.get("ray_control") != "hsn0":
        raise ResultError(f"{source}: Ray control traffic is not labeled hsn0")
    if transport == "nixl" and record.get("backend") != "LIBFABRIC":
        raise ResultError(f"{source}: NIXL backend is not LIBFABRIC")
    if transport == "nixl":
        expected_policy = "pinned" if nic_count == 1 else "striped"
        if record.get("rail_policy") != expected_policy:
            raise ResultError(
                f"{source}: NIXL rail policy is not {expected_policy}"
            )
    if transport == "nccl" and any(
        record.get(name) != expected
        for name, expected in (
            ("gdr_level", "PHB"),
            ("gdr_read", "1"),
            ("netdevs_policy", "MAX:1"),
        )
    ):
        raise ResultError(f"{source}: NCCL GDR path fields are inconsistent")


def parse_result_records(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], int]]:
    text = read_log(path)
    all_results = [fields(item, path) for item in records(text, "RESULT ")]
    all_paths = [fields(item, path) for item in records(text, "PATH ")]
    all_affinities = [
        fields(item, path) for item in records(text, "AFFINITY ")
    ]
    all_pools = [fields(item, path) for item in records(text, "ACTOR_POOL ")]
    case_records = all_results + all_paths + all_affinities + all_pools
    legacy_cases = {
        require(item, "case", path)
        for item in case_records
        if require(item, "case", path) in LEGACY_OUT_OF_SCOPE_CASES
    }

    def retained_records(items: list[dict[str, str]]) -> list[dict[str, str]]:
        return [
            item
            for item in items
            if require(item, "case", path) not in LEGACY_OUT_OF_SCOPE_CASES
        ]

    raw_results = retained_records(all_results)
    raw_paths = retained_records(all_paths)
    raw_affinities = retained_records(all_affinities)
    raw_pools = retained_records(all_pools)
    raw_cleanups = [fields(item, path) for item in records(text, "ACTOR_CLEANUP ")]
    sessions = [fields(item, path) for item in records(text, "SESSION ")]
    stacks = [fields(item, path) for item in records(text, "STACK ")]
    paths_by_case = {require(item, "case", path): item for item in raw_paths}
    if len(paths_by_case) != len(raw_paths):
        raise ResultError(f"{path}: duplicate PATH case identifiers")
    if not raw_results:
        raise ResultError(f"{path}: no RESULT records")
    result_cases = [require(item, "case", path) for item in raw_results]
    if len(set(result_cases)) != len(result_cases):
        raise ResultError(f"{path}: duplicate RESULT case identifiers")
    if set(result_cases) != set(paths_by_case):
        raise ResultError(f"{path}: PATH and RESULT case identifiers differ")
    affinities_by_case = {
        require(item, "case", path): item for item in raw_affinities
    }
    if len(affinities_by_case) != len(raw_affinities):
        raise ResultError(f"{path}: duplicate AFFINITY case identifiers")
    if set(result_cases) != set(affinities_by_case):
        raise ResultError(f"{path}: AFFINITY and RESULT case identifiers differ")
    pools_by_case = {require(item, "case", path): item for item in raw_pools}
    if len(pools_by_case) != len(raw_pools):
        raise ResultError(f"{path}: duplicate ACTOR_POOL case identifiers")

    transports = {require(item, "transport", path) for item in raw_results}
    if len(transports) != 1:
        raise ResultError(f"{path}: a log must contain one transport")
    transport = next(iter(transports))
    if len(sessions) != 1:
        raise ResultError(f"{path}: expected exactly one SESSION record")
    session = sessions[0]
    profile = require(session, "profile", path)
    expected_profiles = {
        "object": {"full"},
        "nixl": {"single-rail", "four-rail"},
        "nccl": {"sweep", "scaling"},
    }
    if (
        require(session, "transport", path) != transport
        or profile not in expected_profiles.get(transport, set())
    ):
        raise ResultError(
            f"{path}: SESSION profile {profile!r} is invalid for {transport}"
        )
    image = require(session, "image", path)
    session_cpus = positive_int(session, "cpus", path)
    session_gpus = positive_int(session, "gpus", path)
    ray_cpus = positive_int(session, "ray_cpus", path)
    ray_gpus = nonnegative_int(session, "ray_gpus", path)
    if (
        session_cpus != 128
        or session_gpus != 4
        or session.get("cpu_bind") != "none"
        or session.get("mem_bind") != "first-touch"
        or session.get("gpu_bind") != "none"
    ):
        raise ResultError(f"{path}: session resources do not match Perlmutter")
    expected_session = {
        "object": (32, 1, "hsn0", "3"),
        "nixl": (128, 0, "distributed", "none"),
        "nccl": (128, 4, "distributed", "0,1,2,3"),
    }
    if (
        ray_cpus,
        ray_gpus,
        session.get("cpu_target"),
        session.get("cuda_visible_devices"),
    ) != expected_session.get(transport):
        raise ResultError(f"{path}: {transport} Ray visibility is incorrect")
    if len(stacks) != 1 or require(stacks[0], "transport", path) != transport:
        raise ResultError(f"{path}: expected exactly one matching STACK record")
    stack = stacks[0]
    for name in ("ray", "torch", "cuda"):
        require(stack, name, path)
    if transport == "nixl":
        for name in (
            "nixl",
            "cxi_optimized_mrs",
            "cxi_mr_cache_max_count",
            "rail_policy",
        ):
            require(stack, name, path)
        if stack["cxi_mr_cache_max_count"] != "1":
            raise ResultError(
                f"{path}: NIXL did not use the required one-entry MR cache"
            )
        expected_rail_policy = {
            "single-rail": "pinned",
            "four-rail": "striped",
        }[profile]
        if stack["rail_policy"] != expected_rail_policy:
            raise ResultError(
                f"{path}: NIXL STACK rail policy disagrees with its profile"
            )
        expected_suites = {
            "single-rail": {"baseline", "single-nic"},
            "four-rail": {"four-rail"},
        }[profile]
        result_suites = {require(item, "suite", path) for item in raw_results}
        if result_suites != expected_suites:
            raise ResultError(
                f"{path}: NIXL profile contains the wrong suites; "
                f"expected={sorted(expected_suites)} actual={sorted(result_suites)}"
            )
        if profile == "four-rail":
            if stack["cxi_optimized_mrs"].lower() not in {"0", "false"}:
                raise ResultError(
                    f"{path}: four-rail NIXL did not use standard MRs"
                )
            if stack.get("max_bw_per_dram_seg") != "1000":
                raise ResultError(
                    f"{path}: four-rail NIXL has the wrong DRAM rail ceiling"
                )
    elif transport == "nccl":
        for name in (
            "nccl",
            "libfabric",
            "disable_dmabuf_cuda",
            "disable_cuda_sync_memops",
        ):
            require(stack, name, path)
        expected_suites = {
            "sweep": {"baseline", "single-nic"},
            "scaling": {"multi-nic"},
        }[profile]
        result_suites = {require(item, "suite", path) for item in raw_results}
        if result_suites != expected_suites:
            raise ResultError(
                f"{path}: NCCL profile contains the wrong suites; "
                f"expected={sorted(expected_suites)} actual={sorted(result_suites)}"
            )
    topology_by_target = validate_transport_log(path, text, transport)
    gpu_by_target: dict[str, tuple[str, int]] = {}
    expected_devices = {
        device
        for item in raw_paths
        for device in require(item, "devices", path).split(",")
        if device.startswith("cxi")
    }
    rail_policies = {
        require(item, "rail_policy", path) for item in raw_paths
        if require(item, "transport", path) == "nixl"
    }
    if transport == "nixl" and rail_policies == {"striped"}:
        device_list = ",".join(TRANSPORT_DEVICES["nixl"])
        if not re.search(
            rf"NIXL_AGENT .*device={re.escape(device_list)}(?:\s|$)", text
        ):
            raise ResultError(
                f"{path}: NIXL did not initialize the four-rail device list"
            )
        for pattern in (
            r"Created 4 rails using provider=cxi",
            r"Registered memory on 4 rails",
            r"use_striping=true",
        ):
            if not re.search(pattern, text):
                raise ResultError(
                    f"{path}: missing native multi-rail evidence {pattern!r}"
                )
    for device in expected_devices:
        if transport == "nccl" and not re.search(
            rf"HCA [0-9]+ '{re.escape(device)}'", text
        ):
            raise ResultError(f"{path}: NCCL did not initialize {device}")
        if (
            transport == "nixl"
            and rail_policies != {"striped"}
            and not re.search(
                rf"NIXL_AGENT .*device={re.escape(device)}(?:\s|$)", text
            )
        ):
            raise ResultError(f"{path}: NIXL did not initialize {device}")

    parsed_results: list[dict[str, Any]] = []
    nixl_pool_retained: list[int] = []
    nixl_slots_by_device: dict[str, int] = {}
    for result in raw_results:
        case = require(result, "case", path)
        try:
            path_record = paths_by_case[case]
        except KeyError as exc:
            raise ResultError(f"{path}: RESULT {case} has no PATH") from exc
        for name in ("transport", "device", "nic_count"):
            if require(result, name, path) != require(path_record, name, path):
                raise ResultError(f"{path}: {case} disagrees with PATH field {name}")
        if result.get("validation_status") != "pass" or result.get(
            "evidence_status"
        ) != "pass":
            raise ResultError(f"{path}: {case} did not pass validation/evidence")

        size_mib = positive_int(result, "size_mib", path)
        nic_count = positive_int(result, "nic_count", path)
        flows_per_nic = positive_int(result, "flows_per_nic", path)
        total_flows = positive_int(result, "total_flows", path)
        warmups = positive_int(result, "warmups", path)
        iterations = positive_int(result, "iterations", path)
        bytes_per_flow = positive_int(result, "bytes_per_flow", path)
        total_bytes = positive_int(result, "total_bytes", path)
        median_ms = positive_float(result, "median_ms", path)
        p95_ms = positive_float(result, "p95_ms", path)
        median_gbps = positive_float(result, "median_aggregate_GBps", path)
        if bytes_per_flow != size_mib * 1024 * 1024:
            raise ResultError(f"{path}: {case} has inconsistent payload bytes")
        if total_bytes != bytes_per_flow * total_flows:
            raise ResultError(f"{path}: {case} has inconsistent total bytes")
        expected_total_flows = nic_count * flows_per_nic
        if total_flows != expected_total_flows:
            raise ResultError(f"{path}: {case} has inconsistent flow counts")
        affinity_record = affinities_by_case[case]
        validate_affinity_record(
            path,
            affinity_record,
            path_record,
            require(result, "transport", path),
            require(result, "device", path),
            nic_count,
            total_flows,
            topology_by_target,
            gpu_by_target,
        )
        if transport == "nixl":
            try:
                pool_record = pools_by_case[case]
            except KeyError as exc:
                raise ResultError(
                    f"{path}: {case} has no ACTOR_POOL record"
                ) from exc
            active = positive_int(pool_record, "active", path)
            retained = positive_int(pool_record, "retained", path)
            path_devices = require(path_record, "devices", path).split(",")
            striped = path_record.get("rail_policy") == "striped"
            pool_devices = [",".join(path_devices)] if striped else path_devices
            slots_per_pool_device = total_flows if striped else flows_per_nic
            for device_name in pool_devices:
                nixl_slots_by_device[device_name] = max(
                    slots_per_pool_device,
                    nixl_slots_by_device.get(device_name, 0),
                )
            expected_retained = 2 * sum(nixl_slots_by_device.values())
            if (
                pool_record.get("status") != "pass"
                or active != 2 * total_flows
                or retained != expected_retained
            ):
                raise ResultError(
                    f"{path}: {case} has inconsistent NIXL actor-pool evidence"
                )
            nixl_pool_retained.append(retained)
        if warmups != DEFAULT_WARMUPS or iterations != DEFAULT_ITERATIONS:
            raise ResultError(f"{path}: {case} has noncanonical sampling counts")
        expected_gbps = total_bytes / (median_ms / 1000) / 1e9
        if not math.isclose(median_gbps, expected_gbps, rel_tol=2e-5):
            raise ResultError(f"{path}: {case} has inconsistent throughput")
        if p95_ms < median_ms:
            raise ResultError(f"{path}: {case} has p95 below its median")
        validate_path_record(
            path,
            path_record,
            require(result, "transport", path),
            require(result, "device", path),
            nic_count,
        )

        parsed_results.append(
            {
                **result,
                "payload_network": require(path_record, "payload_network", path),
                "provider": require(path_record, "provider", path),
                "devices": require(path_record, "devices", path),
                "rail_policy": path_record.get("rail_policy", ""),
                "target_pci": require(affinity_record, "target_pci", path),
                "target_numa": require(affinity_record, "target_numa", path),
                "target_cpus": require(affinity_record, "target_cpus", path),
                "cpu_sets": require(affinity_record, "cpu_sets", path),
                "gpu_numa": require(affinity_record, "gpu_numa", path),
                "gpu_pci": require(affinity_record, "gpu_pci", path),
                "image": image,
                "session_profile": profile,
                "session_cpus": session_cpus,
                "session_gpus": session_gpus,
                "ray_cpus": ray_cpus,
                "ray_gpus": ray_gpus,
                "session_cpu_bind": require(session, "cpu_bind", path),
                "session_mem_bind": require(session, "mem_bind", path),
                "session_gpu_bind": require(session, "gpu_bind", path),
                "session_cpu_target": require(session, "cpu_target", path),
                "cuda_visible_devices": require(
                    session, "cuda_visible_devices", path
                ),
                "ray_version": stack.get("ray", ""),
                "torch_version": stack.get("torch", ""),
                "cuda_version": stack.get("cuda", ""),
                "nixl_version": stack.get("nixl", ""),
                "cxi_optimized_mrs": stack.get("cxi_optimized_mrs", ""),
                "cxi_mr_cache_max_count": stack.get(
                    "cxi_mr_cache_max_count", ""
                ),
                "nccl_version": stack.get("nccl", ""),
                "libfabric_version": stack.get("libfabric", ""),
                "disable_dmabuf_cuda": stack.get("disable_dmabuf_cuda", ""),
                "disable_cuda_sync_memops": stack.get(
                    "disable_cuda_sync_memops", ""
                ),
            }
        )

    if transport == "nixl":
        if set(pools_by_case) != set(result_cases):
            raise ResultError(
                f"{path}: ACTOR_POOL and RESULT case identifiers differ"
            )
        if any(
            current < previous
            for previous, current in zip(
                nixl_pool_retained, nixl_pool_retained[1:], strict=False
            )
        ):
            raise ResultError(f"{path}: NIXL actor pool shrank during the sweep")
        if len(raw_cleanups) != 1:
            raise ResultError(
                f"{path}: expected one final NIXL ACTOR_CLEANUP record"
            )
        cleanup = raw_cleanups[0]
        graceful = positive_int(cleanup, "graceful", path)
        forced = nonnegative_int(cleanup, "forced", path)
        retained = positive_int(cleanup, "retained", path)
        allowed_retained = {expected_retained}
        if "nixl-cpu-four-rail-1024mib-4n-4fpn" in legacy_cases:
            # The retained actors from this out-of-scope attempt were all
            # cleaned gracefully after its CXI memory-registration failure.
            allowed_retained.add(32)
        expected_retained = max(nixl_pool_retained)
        if (
            cleanup.get("scope") != "nixl-pool"
            or cleanup.get("status") != "pass"
            or forced != 0
            or graceful != retained
            or retained not in allowed_retained
        ):
            raise ResultError(
                f"{path}: final NIXL actor-pool cleanup is inconsistent"
            )

    operating_points: dict[tuple[str, str], int] = {}
    for record in records(text, "OPERATING_POINT "):
        item = fields(record, path)
        key = (require(item, "transport", path), require(item, "device", path))
        if key in operating_points:
            raise ResultError(f"{path}: duplicate operating point for {key}")
        if item.get("criterion") != "95pct_of_sweep_max":
            raise ResultError(f"{path}: operating point criterion is inconsistent")
        operating_points[key] = positive_int(item, "flows_per_nic", path)
    return parsed_results, operating_points


def load_matrix(paths: list[Path]) -> list[dict[str, Any]]:
    if len(paths) != 5:
        raise ResultError(f"expected five transport logs; found {len(paths)}")
    if len({path.resolve() for path in paths}) != len(paths):
        raise ResultError("duplicate input logs")
    rows: list[dict[str, Any]] = []
    operating_points: dict[tuple[str, str], int] = {}
    for path in paths:
        parsed, points = parse_result_records(path)
        rows.extend(parsed)
        for key, value in points.items():
            if key in operating_points:
                raise ResultError(f"duplicate operating point for {key}")
            operating_points[key] = value

    if len({row["image"] for row in rows}) != 1:
        raise ResultError("transport logs used different container images")
    for name in ("ray_version", "torch_version", "cuda_version"):
        if len({row[name] for row in rows}) != 1:
            raise ResultError(f"transport logs disagree on {name}")
    session_profiles = {
        (row["transport"], row["session_profile"]) for row in rows
    }
    expected_profiles = {
        ("object", "full"),
        ("nixl", "single-rail"),
        ("nixl", "four-rail"),
        ("nccl", "sweep"),
        ("nccl", "scaling"),
    }
    if session_profiles != expected_profiles:
        raise ResultError(
            "transport session profiles are inconsistent; "
            f"expected={sorted(expected_profiles)} actual={sorted(session_profiles)}"
        )

    keyed: dict[tuple[str, str, str, int, int, int], dict[str, Any]] = {}
    case_ids: set[str] = set()
    for row in rows:
        if row["case"] in case_ids:
            raise ResultError(f"duplicate case identifier {row['case']}")
        case_ids.add(row["case"])
        key = (
            row["transport"],
            row["device"],
            row["suite"],
            int(row["size_mib"]),
            int(row["nic_count"]),
            int(row["flows_per_nic"]),
        )
        if key in keyed:
            raise ResultError(f"duplicate result {key}")
        keyed[key] = row

    if set(operating_points) != set(SERIES):
        missing = sorted(set(SERIES) - set(operating_points))
        unexpected = sorted(set(operating_points) - set(SERIES))
        raise ResultError(
            f"operating point set is inconsistent; missing={missing}, "
            f"unexpected={unexpected}"
        )
    for series in SERIES:
        sweep = {
            int(row["flows_per_nic"]): float(row["median_aggregate_GBps"])
            for row in rows
            if row["suite"] == "single-nic"
            and (row["transport"], row["device"]) == series
        }
        expected_flow_counts = FLOW_COUNTS_BY_SERIES[series]
        if set(sweep) != set(expected_flow_counts):
            raise ResultError(f"incomplete flow sweep for {series}")
        threshold = max(sweep.values()) * 0.95
        computed = next(
            flows for flows in expected_flow_counts if sweep[flows] >= threshold
        )
        if operating_points[series] != computed:
            raise ResultError(
                f"operating point for {series} is {operating_points[series]}, "
                f"but the flow sweep selects {computed}"
            )

    expected_baseline = {
        (transport, device, "baseline", size, 1, 1)
        for transport, device in SERIES
        for size in SIZES_MIB
    }
    expected_flows = {
        (transport, device, "single-nic", 1024, 1, flows)
        for transport, device in SERIES
        for flows in FLOW_COUNTS_BY_SERIES[(transport, device)]
    }
    expected_nixl_four_rail = {
        ("nixl", "cpu", "four-rail", 1024, 4, flows)
        for flows in NIXL_FOUR_RAIL_FLOW_COUNTS
    }
    nccl_operating_point = operating_points[("nccl", "gpu")]
    expected_nccl_multi_nic = {
        ("nccl", "gpu", "multi-nic", 1024, nics, nccl_operating_point)
        for nics in (1, 2, 4)
    }
    expected = (
        expected_baseline
        | expected_flows
        | expected_nixl_four_rail
        | expected_nccl_multi_nic
    )
    if set(keyed) != expected:
        missing = sorted(expected - set(keyed))
        unexpected = sorted(set(keyed) - expected)
        raise ResultError(
            f"incomplete 32-result matrix; missing={missing}, unexpected={unexpected}"
        )
    if len(rows) != 32:
        raise ResultError(f"expected 32 results; found {len(rows)}")
    return sorted(rows, key=lambda row: row["case"])


CSV_FIELDS = (
    "case",
    "image",
    "session_profile",
    "session_cpus",
    "session_gpus",
    "ray_cpus",
    "ray_gpus",
    "session_cpu_bind",
    "session_mem_bind",
    "session_gpu_bind",
    "session_cpu_target",
    "cuda_visible_devices",
    "suite",
    "transport",
    "device",
    "payload_network",
    "provider",
    "devices",
    "rail_policy",
    "target_pci",
    "target_numa",
    "target_cpus",
    "cpu_sets",
    "gpu_numa",
    "gpu_pci",
    "size_mib",
    "nic_count",
    "flows_per_nic",
    "total_flows",
    "bytes_per_flow",
    "total_bytes",
    "warmups",
    "iterations",
    "median_ms",
    "p95_ms",
    "median_aggregate_GBps",
    "validation_status",
    "evidence_status",
    "ray_version",
    "torch_version",
    "cuda_version",
    "nixl_version",
    "cxi_optimized_mrs",
    "cxi_mr_cache_max_count",
    "nccl_version",
    "libfabric_version",
    "disable_dmabuf_cuda",
    "disable_cuda_sync_memops",
)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in CSV_FIELDS})


def atomic_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_csv(rows, temporary)
    os.replace(temporary, path)


def create_staged_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    return Path(name)


def plot_setup() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        matplotlib.rcParams["svg.hashsalt"] = "ray-perlmutter-transport"
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ResultError("Matplotlib is required to publish SVGs") from exc
    return plt


LABELS = {
    ("object", "cpu"): "Object Store / TCP",
    ("nixl", "cpu"): "NIXL / CXI",
    ("object", "gpu"): "Object Store / TCP",
    ("nccl", "gpu"): "NCCL / CXI",
}
COLORS = {"object": "#666666", "nixl": "#009E73", "nccl": "#0072B2"}


def save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fig.savefig(
        temporary,
        format="svg",
        metadata={"Creator": "plot_benchmark_results.py", "Date": None},
    )
    svg = temporary.read_text(encoding="utf-8")
    temporary.write_text(
        "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def plot_two_panels(
    rows: list[dict[str, Any]], suite: str, x_field: str, path: Path
) -> None:
    plt = plot_setup()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    try:
        for axis, device, title in zip(
            axes, ("cpu", "gpu"), ("CPU tensors", "GPU tensors"), strict=True
        ):
            for transport, series_device in SERIES:
                if series_device != device:
                    continue
                selected = [
                    row
                    for row in rows
                    if row["suite"] == suite
                    and row["transport"] == transport
                    and row["device"] == device
                ]
                selected.sort(key=lambda row: int(row[x_field]))
                axis.plot(
                    [int(row[x_field]) for row in selected],
                    [float(row["median_aggregate_GBps"]) for row in selected],
                    marker="o",
                    linewidth=2,
                    color=COLORS[transport],
                    label=LABELS[(transport, device)],
                )
            axis.set_title(title)
            axis.set_xscale("log", base=2)
            axis.grid(True, color="#dddddd")
            axis.legend(frameon=False)
        axes[0].set_ylabel("Median aggregate throughput (GB/s)")
        if suite == "baseline":
            axes[0].set_xticks(SIZES_MIB, ("1 MiB", "64 MiB", "1 GiB"))
            axes[1].set_xticks(SIZES_MIB, ("1 MiB", "64 MiB", "1 GiB"))
            fig.supxlabel("Payload per flow")
        else:
            for axis in axes:
                axis.set_xticks(FLOW_COUNTS, tuple(str(item) for item in FLOW_COUNTS))
            fig.supxlabel("Concurrent flows through one interface")
        fig.tight_layout()
        save_figure(fig, path)
    finally:
        plt.close(fig)


def plot_multi_nic(rows: list[dict[str, Any]], path: Path) -> None:
    plt = plot_setup()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    try:
        nixl_rows = [
            row
            for row in rows
            if row["suite"] == "four-rail" and row["transport"] == "nixl"
        ]
        nixl_rows.sort(key=lambda row: int(row["flows_per_nic"]))
        axes[0].plot(
            [int(row["flows_per_nic"]) for row in nixl_rows],
            [float(row["median_aggregate_GBps"]) for row in nixl_rows],
            marker="o",
            linewidth=2,
            color=COLORS["nixl"],
            label="4 striped rails",
        )
        axes[0].set_title("NIXL CPU")
        axes[0].set_xlabel("1 GiB flows per NIC")
        axes[0].set_xticks(NIXL_FOUR_RAIL_FLOW_COUNTS)

        nccl_rows = [
            row
            for row in rows
            if row["suite"] == "multi-nic" and row["transport"] == "nccl"
        ]
        nccl_rows.sort(key=lambda row: int(row["nic_count"]))
        nccl_fpn = nccl_rows[0]["flows_per_nic"]
        axes[1].plot(
            [int(row["nic_count"]) for row in nccl_rows],
            [float(row["median_aggregate_GBps"]) for row in nccl_rows],
            marker="o",
            linewidth=2,
            color=COLORS["nccl"],
            label=f"{nccl_fpn} flows/NIC",
        )
        axes[1].set_title("NCCL GPU")
        axes[1].set_xlabel("CXI NICs")
        axes[1].set_xticks((1, 2, 4))

        for axis in axes:
            axis.grid(True, color="#dddddd")
            axis.legend(frameon=False)
        axes[0].set_ylabel("Median aggregate throughput (GB/s)")
        fig.tight_layout()
        save_figure(fig, path)
    finally:
        plt.close(fig)


def publish(args: argparse.Namespace) -> None:
    rows = load_matrix(args.logs)
    final_paths = (
        args.csv,
        args.baseline_svg,
        args.flows_svg,
        args.nics_svg,
    )
    if len({path.resolve() for path in final_paths}) != len(final_paths):
        raise ResultError("canonical output paths must be distinct")
    staged_paths: list[Path] = []
    try:
        staged_paths.extend(create_staged_path(path) for path in final_paths)
        write_csv(rows, staged_paths[0])
        plot_two_panels(rows, "baseline", "size_mib", staged_paths[1])
        plot_two_panels(rows, "single-nic", "flows_per_nic", staged_paths[2])
        plot_multi_nic(rows, staged_paths[3])
        for staged, final in zip(staged_paths, final_paths, strict=True):
            os.replace(staged, final)
    finally:
        for staged in staged_paths:
            try:
                staged.unlink()
            except FileNotFoundError:
                pass
    print(f"ARTIFACT kind=csv path={args.csv} records={len(rows)}")
    print(f"ARTIFACT kind=baseline-svg path={args.baseline_svg}")
    print(f"ARTIFACT kind=flows-svg path={args.flows_svg}")
    print(f"ARTIFACT kind=nics-svg path={args.nics_svg}")


def main() -> None:
    args = parse_args()
    publish(args)


if __name__ == "__main__":
    try:
        main()
    except ResultError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
