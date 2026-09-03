#!/usr/bin/env python3
"""Shared mechanics for the two-node Ray transport benchmarks."""

from __future__ import annotations

import gc
import os
import socket
import time
from pathlib import Path
from typing import Any, Iterable

import ray
import torch
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import (
    NodeAffinitySchedulingStrategy,
    PlacementGroupSchedulingStrategy,
)

from benchmark_stats import Measurements, choose_operating_point, summarize


MIB = 1024 * 1024
BASELINE_SIZES_MIB = (1, 64, 1024)
FLOW_COUNTS = (1, 2, 4, 8)
CXI_DEVICES = ("cxi3", "cxi2", "cxi1", "cxi0")
DEFAULT_WARMUPS = 3
DEFAULT_ITERATIONS = 10
PAYLOAD_VALUE = 17
PAYLOAD_LAST_VALUE = 29
PAYLOAD_EDGE_CHECKSUM = PAYLOAD_VALUE + PAYLOAD_LAST_VALUE
VERIFY_CHUNK_BYTES = 16 * MIB


def parse_cpu_list(value: str) -> set[int]:
    cpus: set[int] = set()
    for item in value.strip().split(","):
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise RuntimeError(f"invalid CPU range {item!r}")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(item))
    if not cpus:
        raise RuntimeError(f"CPU list is empty: {value!r}")
    return cpus


def format_cpu_list(cpus: Iterable[int]) -> str:
    ordered = sorted(set(cpus))
    if not ordered:
        raise RuntimeError("cannot format an empty CPU set")
    ranges: list[str] = []
    start = previous = ordered[0]
    for cpu in ordered[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def normalize_pci_address(value: str | bytes) -> str:
    text = value.decode() if isinstance(value, bytes) else value
    parts = text.strip().lower().split(":")
    if len(parts) != 3 or "." not in parts[2]:
        raise RuntimeError(f"invalid PCI address {text!r}")
    domain, bus, device_function = parts
    return f"{domain[-4:].zfill(4)}:{bus.zfill(2)}:{device_function}"


def linux_device_topology(kind: str, name: str) -> dict[str, Any]:
    class_name = {"cxi": "cxi", "net": "net"}.get(kind)
    if class_name is None:
        raise RuntimeError(f"unsupported Linux device kind {kind!r}")
    link = Path("/sys/class") / class_name / name / "device"
    try:
        device = link.resolve(strict=True)
        numa_node = int((device / "numa_node").read_text().strip())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot resolve NUMA topology for {kind}:{name}") from exc
    if numa_node < 0:
        raise RuntimeError(f"{kind}:{name} has no NUMA node")
    cpu_path = Path("/sys/devices/system/node") / f"node{numa_node}" / "cpulist"
    try:
        cpus = parse_cpu_list(cpu_path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot read CPUs for NUMA node {numa_node}") from exc
    return {
        "target": name,
        "target_pci": normalize_pci_address(device.name),
        "target_numa": numa_node,
        "target_cpus": format_cpu_list(cpus),
    }


def gpu_device_topology() -> dict[str, Any]:
    try:
        import cupy

        pci = normalize_pci_address(cupy.cuda.runtime.deviceGetPCIBusId(0))
        device = (Path("/sys/bus/pci/devices") / pci).resolve(strict=True)
        numa_node = int((device / "numa_node").read_text().strip())
    except Exception as exc:
        raise RuntimeError("cannot resolve the actor GPU's NUMA topology") from exc
    if numa_node < 0:
        raise RuntimeError(f"GPU {pci} has no NUMA node")
    return {"gpu_pci": pci, "gpu_numa": numa_node}


def bind_process_to_device(kind: str, name: str) -> dict[str, Any]:
    topology = linux_device_topology(kind, name)
    target_cpus = parse_cpu_list(topology["target_cpus"])
    allowed_cpus = set(os.sched_getaffinity(0))
    effective_cpus = target_cpus & allowed_cpus
    if not effective_cpus:
        raise RuntimeError(
            f"{kind}:{name} NUMA CPUs {topology['target_cpus']} do not intersect "
            f"the worker's allowed CPUs {format_cpu_list(allowed_cpus)}"
        )

    # Ray actors are processes with pre-existing native threads. Bind every
    # current thread so Ray/NIXL progress threads cannot retain a remote-NUMA
    # mask; later threads inherit the mask of their pinned creator.
    os.sched_setaffinity(0, effective_cpus)
    thread_ids: set[int] = set()
    for _ in range(3):
        for task in Path("/proc/self/task").iterdir():
            try:
                thread_id = int(task.name)
                os.sched_setaffinity(thread_id, effective_cpus)
                thread_ids.add(thread_id)
            except ProcessLookupError:
                continue
        escaped_threads: list[int] = []
        for task in Path("/proc/self/task").iterdir():
            try:
                thread_id = int(task.name)
                if set(os.sched_getaffinity(thread_id)) != effective_cpus:
                    escaped_threads.append(thread_id)
            except ProcessLookupError:
                continue
        if not escaped_threads:
            break
    else:
        raise RuntimeError(
            f"actor threads did not retain {format_cpu_list(effective_cpus)}"
        )
    if not thread_ids:
        raise RuntimeError("could not bind any actor threads")
    effective = set(os.sched_getaffinity(0))
    if effective != effective_cpus:
        raise RuntimeError(
            f"effective affinity {format_cpu_list(effective)} differs from "
            f"requested affinity {format_cpu_list(effective_cpus)}"
        )
    return {
        **topology,
        "effective_cpus": format_cpu_list(effective),
        "bound_threads": len(thread_ids),
    }


def actor_affinity(kind: str, target: str, device: str) -> dict[str, Any]:
    affinity = bind_process_to_device(kind, target)
    gpu = gpu_device_topology() if device == "cuda" else {
        "gpu_pci": "none",
        "gpu_numa": "none",
    }
    if device == "cuda" and gpu["gpu_numa"] != affinity["target_numa"]:
        raise RuntimeError(
            f"GPU {gpu['gpu_pci']} is on NUMA {gpu['gpu_numa']}, but "
            f"{target} is on NUMA {affinity['target_numa']}"
        )
    return {**affinity, **gpu}


def validate_and_emit_affinity(
    case_id: str,
    transport: str,
    device: str,
    infos: list[dict[str, Any]],
    targets: list[str],
) -> None:
    expected_targets = targets * 2
    if len(infos) != len(expected_targets):
        raise RuntimeError(f"{case_id}: affinity actor count is inconsistent")
    target_order = list(dict.fromkeys(targets))
    topology_by_target: dict[str, tuple[str, int, str]] = {}
    effective_by_target: dict[str, str] = {}
    gpu_by_target: dict[str, tuple[str, int]] = {}
    for info, expected_target in zip(infos, expected_targets, strict=True):
        if info.get("target") != expected_target:
            raise RuntimeError(f"{case_id}: actor target ordering changed")
        target_numa = int(info["target_numa"])
        target_cpus = str(info["target_cpus"])
        effective_cpus = parse_cpu_list(str(info["effective_cpus"]))
        if not effective_cpus <= parse_cpu_list(target_cpus):
            raise RuntimeError(f"{case_id}: an actor escaped its target NUMA CPUs")
        topology = (str(info["target_pci"]), target_numa, target_cpus)
        previous_topology = topology_by_target.setdefault(expected_target, topology)
        if previous_topology != topology:
            raise RuntimeError(
                f"{case_id}: {expected_target} topology differs between actors"
            )
        effective_text = format_cpu_list(effective_cpus)
        previous_effective = effective_by_target.setdefault(
            expected_target, effective_text
        )
        if previous_effective != effective_text:
            raise RuntimeError(
                f"{case_id}: {expected_target} actor CPU masks differ"
            )
        if device == "gpu":
            gpu = (str(info["gpu_pci"]), int(info["gpu_numa"]))
            if gpu[1] != target_numa:
                raise RuntimeError(
                    f"{case_id}: GPU and {expected_target} are not NUMA-local"
                )
            previous_gpu = gpu_by_target.setdefault(expected_target, gpu)
            if previous_gpu != gpu:
                raise RuntimeError(
                    f"{case_id}: {expected_target} actors do not share one GPU"
                )
        elif info.get("gpu_pci") != "none" or info.get("gpu_numa") != "none":
            raise RuntimeError(f"{case_id}: CPU actor unexpectedly reported a GPU")
    if device == "gpu" and len(set(gpu_by_target.values())) != len(target_order):
        raise RuntimeError(f"{case_id}: distinct targets did not use distinct GPUs")

    target_numas = [str(topology_by_target[target][1]) for target in target_order]
    target_pcis = [topology_by_target[target][0] for target in target_order]
    target_cpu_sets = [
        topology_by_target[target][2].replace(",", "+") for target in target_order
    ]
    cpu_sets = [
        effective_by_target[target].replace(",", "+") for target in target_order
    ]
    if device == "gpu":
        gpu_numas = [str(gpu_by_target[target][1]) for target in target_order]
        gpu_pcis = [gpu_by_target[target][0] for target in target_order]
    else:
        gpu_numas = ["none"]
        gpu_pcis = ["none"]
    print(
        f"AFFINITY case={case_id} transport={transport} device={device} "
        f"policy=local targets={','.join(target_order)} "
        f"target_pci={','.join(target_pcis)} "
        f"target_numa={','.join(target_numas)} "
        f"target_cpus={'|'.join(target_cpu_sets)} "
        f"cpu_numa={','.join(target_numas)} cpu_sets={'|'.join(cpu_sets)} "
        f"gpu_numa={','.join(gpu_numas)} gpu_pci={','.join(gpu_pcis)} "
        f"actors={len(infos)} status=pass",
        flush=True,
    )


def identity() -> dict[str, Any]:
    context = ray.get_runtime_context()
    return {
        "hostname": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        "node_id": str(context.get_node_id()),
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "accelerator_ids": context.get_accelerator_ids().get("GPU", []),
    }


@ray.remote(num_cpus=0)
def network_identity(interface: str) -> dict[str, Any]:
    record = identity()
    topology = linux_device_topology("net", interface)
    hsn_hostname = f"{record['hostname']}-{interface}"
    try:
        addresses = sorted(
            {
                address[4][0]
                for address in socket.getaddrinfo(
                    hsn_hostname,
                    None,
                    family=socket.AF_INET,
                    type=socket.SOCK_STREAM,
                )
            }
        )
        resolution_error = ""
    except socket.gaierror as error:
        addresses = []
        resolution_error = str(error)
    return {
        **record,
        **topology,
        "interface": interface,
        "hsn_hostname": hsn_hostname,
        "hsn_ipv4": addresses,
        "resolution_error": resolution_error,
    }


def select_nodes() -> tuple[dict[str, Any], dict[str, Any]]:
    nodes = [node for node in ray.nodes() if node.get("Alive")]
    if len(nodes) != 2:
        raise RuntimeError(
            f"this benchmark requires exactly two alive Ray nodes; found {len(nodes)}"
        )
    head_id = str(ray.get_runtime_context().get_node_id())
    head = next(
        (node for node in nodes if str(node.get("NodeID")) == head_id), None
    )
    if head is None:
        raise RuntimeError("could not identify the Ray head node")
    worker = next(node for node in nodes if str(node.get("NodeID")) != head_id)
    return head, worker


def validate_hsn0(
    head: dict[str, Any], worker: dict[str, Any]
) -> list[dict[str, Any]]:
    records = ray.get(
        [
            network_identity.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node["NodeID"], soft=False
                )
            ).remote("hsn0")
            for node in (head, worker)
        ]
    )
    topology = {
        (record["target_pci"], record["target_numa"], record["target_cpus"])
        for record in records
    }
    if len(topology) != 1:
        raise RuntimeError("hsn0 NUMA topology differs between the two Ray nodes")
    for node, record in zip((head, worker), records, strict=True):
        manager_address = str(node.get("NodeManagerAddress", ""))
        if manager_address not in record["hsn_ipv4"]:
            resolution_detail = (
                f"; resolution failed: {record['resolution_error']}"
                if record["resolution_error"]
                else ""
            )
            raise RuntimeError(
                f"Ray node {record['hostname']} uses {manager_address}, but "
                f"{record['hsn_hostname']} resolves to {record['hsn_ipv4']}"
                f"{resolution_detail}"
            )
        print(
            f"NETWORK hostname={record['hostname']} ray_ip={manager_address} "
            f"interface=hsn0 alias={record['hsn_hostname']} "
            f"pci={record['target_pci']} numa={record['target_numa']} "
            f"cpus={record['target_cpus'].replace(',', '+')} status=pass",
            flush=True,
        )
    return records


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_payload(nbytes: int, device: torch.device) -> torch.Tensor:
    payload = torch.full(
        (nbytes,), PAYLOAD_VALUE, dtype=torch.uint8, device=device
    )
    payload[-1] = PAYLOAD_LAST_VALUE
    synchronize(device)
    return payload


def transfer_metadata(payload: torch.Tensor) -> dict[str, Any]:
    nbytes = payload.numel() * payload.element_size()
    edge_checksum = int(payload[0].item()) + int(payload[-1].item())
    synchronize(payload.device)
    return {
        "nbytes": nbytes,
        "edge_checksum": edge_checksum,
        "device": payload.device.type,
    }


def verify_transfer_metadata(
    result: dict[str, Any], expected_nbytes: int, expected_device: str
) -> None:
    if result["nbytes"] != expected_nbytes:
        raise RuntimeError(
            f"receiver saw {result['nbytes']} bytes; expected {expected_nbytes}"
        )
    if result["edge_checksum"] != PAYLOAD_EDGE_CHECKSUM:
        raise RuntimeError("receiver edge checksum is incorrect")
    if result["device"] != expected_device:
        raise RuntimeError(
            f"receiver tensor device was {result['device']}; expected {expected_device}"
        )


def verify_full_payload(
    payload: torch.Tensor, expected_nbytes: int, expected_device: str
) -> dict[str, Any]:
    actual_nbytes = payload.numel() * payload.element_size()
    body_is_valid = True
    # Avoid allocating another full-sized tensor on every concurrent actor.
    # Eight 1 GiB flows already retain 8 GiB on one GPU during the single-NIC
    # test, so verify in bounded chunks outside the timer.
    for offset in range(0, payload.numel() - 1, VERIFY_CHUNK_BYTES):
        end = min(offset + VERIFY_CHUNK_BYTES, payload.numel() - 1)
        if not bool(torch.all(payload[offset:end] == PAYLOAD_VALUE).item()):
            body_is_valid = False
            break
    last_value = int(payload[-1].item())
    synchronize(payload.device)
    if actual_nbytes != expected_nbytes:
        raise RuntimeError(
            f"receiver saw {actual_nbytes} bytes; expected {expected_nbytes}"
        )
    if payload.device.type != expected_device:
        raise RuntimeError(
            f"receiver tensor device was {payload.device}; expected {expected_device}"
        )
    if not body_is_valid or last_value != PAYLOAD_LAST_VALUE:
        raise RuntimeError("full payload verification failed")
    return {"nbytes": actual_nbytes, "verified": True}


def run_batches(
    senders: list[Any],
    receivers: list[Any],
    nbytes: int,
    device: str,
    warmups: int,
    iterations: int,
) -> Measurements:
    if len(senders) != len(receivers) or not senders:
        raise RuntimeError("sender and receiver flow sets are inconsistent")
    durations: list[float] = []
    for run in range(warmups + iterations):
        start = time.perf_counter()
        tensor_refs = [sender.send.remote() for sender in senders]
        result_refs = [
            receiver.receive.remote(tensor_ref)
            for receiver, tensor_ref in zip(receivers, tensor_refs, strict=True)
        ]
        results = ray.get(result_refs)
        duration = time.perf_counter() - start
        for result in results:
            verify_transfer_metadata(result, nbytes, device)
        verified = ray.get(
            [receiver.verify_and_release.remote(nbytes) for receiver in receivers]
        )
        if not all(item["verified"] for item in verified):
            raise RuntimeError("a receiver did not verify its complete payload")
        del tensor_refs, result_refs, results, verified
        gc.collect()
        if run >= warmups:
            durations.append(duration)
    return summarize(durations, len(senders) * nbytes)


def emit_path(
    case_id: str,
    transport: str,
    device: str,
    payload_network: str,
    provider: str,
    nic_count: int,
    devices: Iterable[str],
    **fields: Any,
) -> None:
    extra = " ".join(f"{name}={value}" for name, value in fields.items())
    print(
        f"PATH case={case_id} transport={transport} device={device} "
        f"payload_network={payload_network} provider={provider} "
        f"nic_count={nic_count} devices={','.join(devices) or 'none'} "
        f"{extra}".rstrip(),
        flush=True,
    )


def emit_result(
    case_id: str,
    suite: str,
    transport: str,
    device: str,
    size_mib: int,
    nic_count: int,
    flows_per_nic: int,
    total_flows: int,
    warmups: int,
    iterations: int,
    measurements: Measurements,
    **fields: Any,
) -> None:
    nbytes = size_mib * MIB
    extra = " ".join(f"{name}={value}" for name, value in fields.items())
    print(
        f"RESULT case={case_id} suite={suite} transport={transport} "
        f"device={device} size_mib={size_mib} nic_count={nic_count} "
        f"flows_per_nic={flows_per_nic} total_flows={total_flows} "
        f"bytes_per_flow={nbytes} total_bytes={nbytes * total_flows} "
        f"warmups={warmups} iterations={iterations} "
        f"median_ms={measurements.median_ms:.6f} "
        f"p95_ms={measurements.p95_ms:.6f} "
        f"median_aggregate_GBps={measurements.median_gbps:.6f} "
        f"validation_status=pass evidence_status=pass {extra}".rstrip(),
        flush=True,
    )


def node_resource_key(node: dict[str, Any]) -> str:
    candidates = sorted(
        name for name in node.get("Resources", {}) if name.startswith("node:")
    )
    if not candidates:
        raise RuntimeError(f"Ray node {node.get('NodeID')} lacks its node resource")
    return candidates[0]


def create_gpu_group(node: dict[str, Any], nic_count: int, flows_per_nic: int) -> Any:
    resource = node_resource_key(node)
    bundles = [
        {"CPU": float(flows_per_nic), "GPU": 1.0, resource: 0.001}
        for _ in range(nic_count)
    ]
    group = placement_group(bundles, strategy="PACK")
    ray.get(group.ready(), timeout=120)
    return group


def gpu_actor_options(group: Any, bundle_index: int, flows_per_nic: int) -> dict[str, Any]:
    return {
        "num_cpus": 1,
        "num_gpus": 1 / flows_per_nic,
        "scheduling_strategy": PlacementGroupSchedulingStrategy(
            placement_group=group,
            placement_group_bundle_index=bundle_index,
            placement_group_capture_child_tasks=True,
        ),
    }


def cpu_actor_options(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_cpus": 1,
        "num_gpus": 0,
        "scheduling_strategy": NodeAffinitySchedulingStrategy(
            node_id=node["NodeID"], soft=False
        ),
    }


def cleanup_actors(
    actors: Iterable[Any],
    groups: Iterable[Any] = (),
    *,
    graceful_timeout_s: float = 0,
) -> tuple[int, int]:
    actor_list = list(actors)
    graceful_count = 0
    forced_count = 0
    if graceful_timeout_s > 0 and actor_list:
        pending: dict[Any, Any] = {}
        for actor in actor_list:
            try:
                termination = actor.__ray_terminate__.remote()
                pending[termination] = actor
            except Exception:
                forced_count += 1
                try:
                    ray.kill(actor, no_restart=True)
                except Exception:
                    pass
        if pending:
            try:
                done, not_done = ray.wait(
                    list(pending),
                    num_returns=len(pending),
                    timeout=graceful_timeout_s,
                )
            except Exception:
                done, not_done = [], list(pending)
            graceful_count += len(done)
            for termination in not_done:
                forced_count += 1
                try:
                    ray.kill(pending[termination], no_restart=True)
                except Exception:
                    pass
    else:
        forced_count = len(actor_list)
        for actor in actor_list:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass
    for group in groups:
        try:
            remove_placement_group(group)
        except Exception:
            pass
    return graceful_count, forced_count
