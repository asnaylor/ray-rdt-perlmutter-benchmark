#!/usr/bin/env python3
"""Benchmark CPU tensors striped by one NIXL agent across four CXI rails."""

from __future__ import annotations

import argparse
import gc
import os
from importlib import metadata
from typing import Any

import ray
import torch

from benchmark_common import (
    CXI_DEVICES,
    DEFAULT_ITERATIONS,
    DEFAULT_WARMUPS,
    MIB,
    cleanup_actors,
    cpu_actor_options,
    emit_path,
    emit_result,
    format_cpu_list,
    identity,
    linux_device_topology,
    make_payload,
    parse_cpu_list,
    run_batches,
    select_nodes,
    transfer_metadata,
    validate_hsn0,
    verify_full_payload,
)
from ray_nixl_bench import (
    ACTOR_GRACEFUL_SHUTDOWN_SECONDS,
    EXPECTED_MR_CACHE_MAX_COUNT,
    NIXL_BACKEND,
    environment_setting,
    install_ray_nixl_backend_override,
    require_environment,
)


EXPECTED_MAX_BW_PER_DRAM_SEG = "1000"
STANDARD_MR_VALUES = {"0", "false"}
FOUR_RAIL_FLOWS_PER_NIC = (1, 2)


def selected_devices() -> tuple[str, ...]:
    return tuple(
        item.strip()
        for item in os.environ.get("FI_CXI_DEVICE_NAME", "").split(",")
        if item.strip()
    )


def multirail_affinity() -> dict[str, Any]:
    topologies = [linux_device_topology("cxi", device) for device in CXI_DEVICES]
    allowed_cpus = set(os.sched_getaffinity(0))
    target_cpus = set().union(
        *(parse_cpu_list(topology["target_cpus"]) for topology in topologies)
    )
    if allowed_cpus != target_cpus:
        raise RuntimeError(
            "a native multi-rail actor does not have the complete four-NUMA "
            f"CPU set: allowed={format_cpu_list(allowed_cpus)} "
            f"expected={format_cpu_list(target_cpus)}"
        )
    return {
        "targets": ",".join(CXI_DEVICES),
        "target_pci": ",".join(
            str(topology["target_pci"]) for topology in topologies
        ),
        "target_numa": ",".join(
            str(topology["target_numa"]) for topology in topologies
        ),
        "target_cpus": "|".join(
            str(topology["target_cpus"]).replace(",", "+")
            for topology in topologies
        ),
        "effective_cpus": format_cpu_list(allowed_cpus).replace(",", "+"),
        "gpu_numa": "none",
        "gpu_pci": "none",
    }


def initialize_transport() -> dict[str, Any]:
    from ray.experimental.gpu_object_manager.util import (
        get_tensor_transport_manager,
    )

    manager = get_tensor_transport_manager("NIXL")
    agent = manager.get_nixl_agent()
    return {
        **identity(),
        **multirail_affinity(),
        "manager": type(manager).__name__,
        "agent": str(agent.name),
        "backend": NIXL_BACKEND,
        "devices": ",".join(selected_devices()),
        "mr_cache_max_count": environment_setting("FI_MR_CACHE_MAX_COUNT"),
    }


@ray.remote(enable_tensor_transport=True)
class Sender:
    def __init__(self) -> None:
        install_ray_nixl_backend_override()
        self.payload: torch.Tensor | None = None

    def initialize_transport(self) -> dict[str, Any]:
        return initialize_transport()

    def topology_info(self) -> dict[str, Any]:
        return {**identity(), **multirail_affinity()}

    def prepare_payload(self, nbytes: int) -> dict[str, Any]:
        if nbytes <= 0:
            raise RuntimeError(f"invalid NIXL payload size {nbytes}")
        current_nbytes = (
            self.payload.numel() * self.payload.element_size()
            if self.payload is not None
            else 0
        )
        if current_nbytes != nbytes:
            self.payload = None
            gc.collect()
            self.payload = make_payload(nbytes, torch.device("cpu"))
        return {**identity(), "devices": ",".join(selected_devices()), "nbytes": nbytes}

    @ray.method(tensor_transport="nixl")
    def send(self) -> torch.Tensor:
        if self.payload is None:
            raise RuntimeError("NIXL sender payload was not prepared")
        return self.payload


@ray.remote(enable_tensor_transport=True)
class Receiver:
    def __init__(self) -> None:
        install_ray_nixl_backend_override()
        self.payload: torch.Tensor | None = None

    def initialize_transport(self) -> dict[str, Any]:
        return initialize_transport()

    def topology_info(self) -> dict[str, Any]:
        return {**identity(), **multirail_affinity()}

    def receive(self, payload: torch.Tensor) -> dict[str, Any]:
        if payload.device.type != "cpu":
            raise RuntimeError(f"NIXL CPU receiver got {payload.device}")
        self.payload = payload
        return {**identity(), **transfer_metadata(payload)}

    def verify_and_release(self, expected_nbytes: int) -> dict[str, Any]:
        if self.payload is None:
            raise RuntimeError("receiver has no payload to verify")
        payload = self.payload
        result = verify_full_payload(payload, expected_nbytes, "cpu")
        self.payload = None
        del payload
        gc.collect()
        return result


class NixlMultirailActorPool:
    """Grow one reusable sender/receiver pair per logical striped flow."""

    def __init__(self, head: dict[str, Any], worker: dict[str, Any]) -> None:
        self.head = head
        self.worker = worker
        self.senders: list[Any] = []
        self.receivers: list[Any] = []

    @staticmethod
    def _validate_transport(info: dict[str, Any]) -> None:
        expected_devices = ",".join(CXI_DEVICES)
        if (
            info["manager"] != "NixlTensorTransport"
            or info["backend"] != NIXL_BACKEND
            or info["devices"] != expected_devices
            or info["targets"] != expected_devices
        ):
            raise RuntimeError(f"unexpected native multi-rail state: {info}")
        if info["mr_cache_max_count"] != EXPECTED_MR_CACHE_MAX_COUNT:
            raise RuntimeError("a NIXL actor did not use the one-entry MR cache")
        if info["cuda_visible_devices"] or info["accelerator_ids"]:
            raise RuntimeError("a NIXL CPU actor was assigned a GPU")

    def acquire(self, flow_count: int) -> tuple[list[Any], list[Any]]:
        new_actors: list[Any] = []
        while len(self.senders) < flow_count:
            sender = Sender.options(**cpu_actor_options(self.head)).remote()
            receiver = Receiver.options(**cpu_actor_options(self.worker)).remote()
            self.senders.append(sender)
            self.receivers.append(receiver)
            new_actors.extend((sender, receiver))
        if new_actors:
            infos = ray.get(
                [actor.initialize_transport.remote() for actor in new_actors]
            )
            for info in infos:
                self._validate_transport(info)
        return self.senders[:flow_count], self.receivers[:flow_count]

    def all_actors(self) -> list[Any]:
        return [*self.senders, *self.receivers]

    def actor_count(self) -> int:
        return len(self.senders) + len(self.receivers)

    def close(self) -> tuple[int, int]:
        return cleanup_actors(
            self.all_actors(),
            graceful_timeout_s=ACTOR_GRACEFUL_SHUTDOWN_SECONDS,
        )


def validate_and_emit_affinity(
    case_id: str, infos: list[dict[str, Any]], flow_count: int
) -> None:
    if len(infos) != 2 * flow_count:
        raise RuntimeError(f"{case_id}: affinity actor count is inconsistent")
    expected_devices = ",".join(CXI_DEVICES)
    fields = (
        "targets",
        "target_pci",
        "target_numa",
        "target_cpus",
        "effective_cpus",
        "gpu_numa",
        "gpu_pci",
    )
    reference = {name: infos[0][name] for name in fields}
    if reference["targets"] != expected_devices:
        raise RuntimeError(f"{case_id}: multi-rail target ordering changed")
    for info in infos:
        if any(info[name] != reference[name] for name in fields):
            raise RuntimeError(f"{case_id}: actor multi-rail topology differs")
        if info["cuda_visible_devices"] or info["accelerator_ids"]:
            raise RuntimeError(f"{case_id}: a CPU actor was assigned a GPU")
    print(
        f"AFFINITY case={case_id} transport=nixl device=cpu "
        f"policy=distributed targets={reference['targets']} "
        f"target_pci={reference['target_pci']} "
        f"target_numa={reference['target_numa']} "
        f"target_cpus={reference['target_cpus']} cpu_numa=all "
        f"cpu_sets={reference['effective_cpus']} gpu_numa=none gpu_pci=none "
        f"actors={len(infos)} status=pass",
        flush=True,
    )


def run_case(pool: NixlMultirailActorPool, flows_per_nic: int) -> Any:
    size_mib = 1024
    nbytes = size_mib * MIB
    nic_count = len(CXI_DEVICES)
    total_flows = nic_count * flows_per_nic
    case_id = f"nixl-cpu-four-rail-{size_mib}mib-4n-{flows_per_nic}fpn"
    print(
        f"RUN case={case_id} suite=four-rail transport=nixl device=cpu "
        f"size_mib={size_mib} nic_count={nic_count} "
        f"flows_per_nic={flows_per_nic} total_flows={total_flows} "
        "rail_policy=striped",
        flush=True,
    )
    emit_path(
        case_id,
        "nixl",
        "cpu",
        "cxi",
        "libfabric",
        nic_count,
        CXI_DEVICES,
        backend="LIBFABRIC",
        rail_policy="striped",
        ray_control="hsn0",
    )
    senders, receivers = pool.acquire(total_flows)
    affinity_infos = ray.get(
        [actor.topology_info.remote() for actor in [*senders, *receivers]]
    )
    validate_and_emit_affinity(case_id, affinity_infos, total_flows)
    prepared = ray.get(
        [sender.prepare_payload.remote(nbytes) for sender in senders]
    )
    expected_devices = ",".join(CXI_DEVICES)
    if any(
        info["devices"] != expected_devices or info["nbytes"] != nbytes
        for info in prepared
    ):
        raise RuntimeError(f"{case_id}: sender payload preparation changed")
    measured = run_batches(
        senders,
        receivers,
        nbytes,
        "cpu",
        DEFAULT_WARMUPS,
        DEFAULT_ITERATIONS,
    )
    emit_result(
        case_id,
        "four-rail",
        "nixl",
        "cpu",
        size_mib,
        nic_count,
        flows_per_nic,
        total_flows,
        DEFAULT_WARMUPS,
        DEFAULT_ITERATIONS,
        measured,
        rail_policy="striped",
    )
    print(
        f"ACTOR_POOL case={case_id} active={2 * total_flows} "
        f"retained={pool.actor_count()} policy=striped status=pass",
        flush=True,
    )
    return measured


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("four-rail",), required=True)
    return parser.parse_args()


def benchmark() -> None:
    require_environment()
    devices = selected_devices()
    if devices != CXI_DEVICES:
        raise RuntimeError(
            f"FI_CXI_DEVICE_NAME={devices}; expected {CXI_DEVICES}"
        )
    optimized_mrs = os.environ.get("FI_CXI_OPTIMIZED_MRS", "").lower()
    if optimized_mrs not in STANDARD_MR_VALUES:
        raise RuntimeError(
            "native four-rail NIXL requires FI_CXI_OPTIMIZED_MRS=false"
        )
    max_bw = os.environ.get("NIXL_LIBFABRIC_MAX_BW_PER_DRAM_SEG")
    if max_bw != EXPECTED_MAX_BW_PER_DRAM_SEG:
        raise RuntimeError(
            "native four-rail NIXL requires "
            f"NIXL_LIBFABRIC_MAX_BW_PER_DRAM_SEG={EXPECTED_MAX_BW_PER_DRAM_SEG}"
        )
    print(
        f"STACK transport=nixl ray={ray.__version__} torch={torch.__version__} "
        f"cuda={torch.version.cuda} nixl={metadata.version('nixl')} "
        f"cxi_optimized_mrs={environment_setting('FI_CXI_OPTIMIZED_MRS')} "
        "cxi_mr_cache_max_count="
        f"{environment_setting('FI_MR_CACHE_MAX_COUNT')} "
        f"rail_policy=striped max_bw_per_dram_seg={max_bw}",
        flush=True,
    )
    ray.init(address="auto")
    pool: NixlMultirailActorPool | None = None
    pool_forced = 0
    try:
        head, worker = select_nodes()
        validate_hsn0(head, worker)
        pool = NixlMultirailActorPool(head, worker)
        for flows_per_nic in FOUR_RAIL_FLOWS_PER_NIC:
            run_case(pool, flows_per_nic)
    finally:
        if pool is not None:
            retained = pool.actor_count()
            graceful, pool_forced = pool.close()
            status = "pass" if pool_forced == 0 else "forced"
            print(
                f"ACTOR_CLEANUP scope=nixl-pool graceful={graceful} "
                f"forced={pool_forced} retained={retained} "
                f"policy=striped status={status}",
                flush=True,
            )
        ray.shutdown()
        print("SHUTDOWN status=clean transport=nixl", flush=True)
    if pool_forced:
        raise RuntimeError(
            f"{pool_forced} retained NIXL actors did not exit gracefully"
        )


if __name__ == "__main__":
    try:
        parse_args()
        benchmark()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
