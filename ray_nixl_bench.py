#!/usr/bin/env python3
"""Benchmark CPU tensors through Ray RDT, NIXL, and LIBFABRIC/CXI."""

from __future__ import annotations

import argparse
import functools
import gc
import inspect
import os
import uuid
from importlib import metadata
from typing import Any

import nixl
import ray


NIXL_BACKEND = "LIBFABRIC"
EXPECTED_RAY_VERSION = "2.54.0"
EXPECTED_NIXL_VERSION = "1.3.2"
EXPECTED_MR_CACHE_MAX_COUNT = "1"
ACTOR_GRACEFUL_SHUTDOWN_SECONDS = 30


def environment_setting(name: str) -> str:
    if name not in os.environ:
        return "unset"
    return os.environ[name] or "empty"


def create_libfabric_agent(agent_name: str) -> Any:
    from nixl._api import nixl_agent, nixl_agent_config

    agent = nixl_agent(agent_name, nixl_agent_config(backends=[]))
    agent.create_backend(NIXL_BACKEND, {})
    return agent


def install_ray_nixl_backend_override() -> None:
    """Replace Ray 2.54's hard-coded NIXL UCX agent with LIBFABRIC."""
    ray_version = ray.__version__.split("+", 1)[0]
    nixl_version = metadata.version("nixl")
    if ray_version != EXPECTED_RAY_VERSION or nixl_version != EXPECTED_NIXL_VERSION:
        raise RuntimeError(
            "the NIXL override requires Ray 2.54.0 and NIXL 1.3.2; "
            f"found Ray {ray.__version__} and NIXL {nixl_version}"
        )

    from ray.experimental.gpu_object_manager.nixl_tensor_transport import (
        NixlTensorTransport,
    )

    installed = getattr(NixlTensorTransport, "_ray_bench_forced_backend", None)
    if installed is not None:
        if installed != NIXL_BACKEND:
            raise RuntimeError(f"Ray NIXL was already forced to {installed}")
        return

    original = NixlTensorTransport.get_nixl_agent
    if list(inspect.signature(original).parameters) != ["self"]:
        raise RuntimeError("Ray's NIXL agent factory signature has changed")
    if 'nixl_agent_config(backends=["UCX"])' not in inspect.getsource(original):
        raise RuntimeError("Ray's expected hard-coded UCX NIXL factory has changed")

    @functools.wraps(original)
    def get_libfabric_agent(self: Any) -> Any:
        if self._nixl_agent is None:
            actor_id = ray.get_runtime_context().get_actor_id()
            name = f"RAY-{actor_id or uuid.uuid4()}"
            self._nixl_agent = create_libfabric_agent(name)
            self._ray_bench_backend = NIXL_BACKEND
            print(
                f"NIXL_AGENT backend=LIBFABRIC agent={name} "
                f"device={os.environ.get('FI_CXI_DEVICE_NAME', 'unset')} "
                "mr_cache_max_count="
                f"{environment_setting('FI_MR_CACHE_MAX_COUNT')}",
                flush=True,
            )
        return self._nixl_agent

    NixlTensorTransport.get_nixl_agent = get_libfabric_agent
    NixlTensorTransport._ray_bench_forced_backend = NIXL_BACKEND


install_ray_nixl_backend_override()

import torch  # noqa: E402

from benchmark_common import (  # noqa: E402
    BASELINE_SIZES_MIB,
    DEFAULT_ITERATIONS,
    DEFAULT_WARMUPS,
    MIB,
    actor_affinity,
    cleanup_actors,
    cpu_actor_options,
    emit_path,
    emit_result,
    identity,
    make_payload,
    run_batches,
    select_nodes,
    transfer_metadata,
    validate_and_emit_affinity,
    validate_hsn0,
    verify_full_payload,
)
from benchmark_stats import choose_operating_point  # noqa: E402


NIXL_FLOW_COUNTS = (1, 2, 4)
NIXL_CXI_DEVICE = "cxi3"


def initialize_transport(affinity: dict[str, Any]) -> dict[str, Any]:
    from ray.experimental.gpu_object_manager.util import (
        get_tensor_transport_manager,
    )

    manager = get_tensor_transport_manager("NIXL")
    agent = manager.get_nixl_agent()
    return {
        **identity(),
        **affinity,
        "manager": type(manager).__name__,
        "agent": str(agent.name),
        "rail": os.environ["FI_CXI_DEVICE_NAME"],
        "mr_cache_max_count": environment_setting("FI_MR_CACHE_MAX_COUNT"),
    }


@ray.remote(enable_tensor_transport=True)
class Sender:
    def __init__(self, rail: str) -> None:
        os.environ["FI_PROVIDER"] = "cxi"
        os.environ["FI_CXI_DEVICE_NAME"] = rail
        self.affinity = actor_affinity("cxi", rail, "cpu")
        install_ray_nixl_backend_override()
        self.payload: torch.Tensor | None = None

    def initialize_transport(self) -> dict[str, Any]:
        return initialize_transport(self.affinity)

    def topology_info(self) -> dict[str, Any]:
        return {**identity(), **self.affinity}

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
        return {
            **identity(),
            "rail": os.environ["FI_CXI_DEVICE_NAME"],
            "nbytes": nbytes,
        }

    @ray.method(tensor_transport="nixl")
    def send(self) -> torch.Tensor:
        if self.payload is None:
            raise RuntimeError("NIXL sender payload was not prepared")
        return self.payload


@ray.remote(enable_tensor_transport=True)
class Receiver:
    def __init__(self, rail: str) -> None:
        os.environ["FI_PROVIDER"] = "cxi"
        os.environ["FI_CXI_DEVICE_NAME"] = rail
        self.affinity = actor_affinity("cxi", rail, "cpu")
        install_ray_nixl_backend_override()
        self.payload: torch.Tensor | None = None

    def initialize_transport(self) -> dict[str, Any]:
        return initialize_transport(self.affinity)

    def topology_info(self) -> dict[str, Any]:
        return {**identity(), **self.affinity}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("single-rail", "smoke"),
        default="single-rail",
    )
    return parser.parse_args()


def require_environment() -> None:
    selected = getattr(getattr(nixl, "_pkg", None), "__name__", "unknown")
    if selected != "nixl_cu13":
        raise RuntimeError(f"NIXL selected {selected}; expected nixl_cu13")
    cache_max_count = os.environ.get("FI_MR_CACHE_MAX_COUNT")
    if cache_max_count != EXPECTED_MR_CACHE_MAX_COUNT:
        raise RuntimeError(
            "NIXL requires FI_MR_CACHE_MAX_COUNT=1 to bound each actor's "
            "libfabric MR cache; "
            f"found {environment_setting('FI_MR_CACHE_MAX_COUNT')}"
        )


def rails_for_case(flow_count: int) -> list[str]:
    return [NIXL_CXI_DEVICE] * flow_count


class NixlActorPool:
    """Lazily grow one reusable sender/receiver pool for each CXI rail."""

    def __init__(self, head: dict[str, Any], worker: dict[str, Any]) -> None:
        self.head = head
        self.worker = worker
        self.senders: dict[str, list[Any]] = {}
        self.receivers: dict[str, list[Any]] = {}

    @staticmethod
    def _validate_transport(info: dict[str, Any], expected_rail: str) -> None:
        if info["manager"] != "NixlTensorTransport":
            raise RuntimeError("Ray did not initialize NixlTensorTransport")
        if info["rail"] != expected_rail:
            raise RuntimeError("NIXL actor rail selection changed unexpectedly")
        if info["target"] != expected_rail:
            raise RuntimeError("NIXL actor affinity target changed unexpectedly")
        if info["mr_cache_max_count"] != EXPECTED_MR_CACHE_MAX_COUNT:
            raise RuntimeError(
                "a NIXL actor did not use the one-entry libfabric MR cache"
            )
        if info["cuda_visible_devices"]:
            raise RuntimeError("a NIXL CPU actor was assigned a GPU")

    def acquire(self, rails: list[str]) -> tuple[list[Any], list[Any]]:
        selected_senders: list[Any] = []
        selected_receivers: list[Any] = []
        next_slot: dict[str, int] = {}
        new_actors: list[Any] = []
        new_actor_rails: list[str] = []

        for rail in rails:
            slot = next_slot.get(rail, 0)
            next_slot[rail] = slot + 1
            rail_senders = self.senders.setdefault(rail, [])
            rail_receivers = self.receivers.setdefault(rail, [])
            while len(rail_senders) <= slot:
                sender = Sender.options(**cpu_actor_options(self.head)).remote(rail)
                receiver = Receiver.options(
                    **cpu_actor_options(self.worker)
                ).remote(rail)
                rail_senders.append(sender)
                rail_receivers.append(receiver)
                new_actors.extend((sender, receiver))
                new_actor_rails.extend((rail, rail))
            selected_senders.append(rail_senders[slot])
            selected_receivers.append(rail_receivers[slot])

        if new_actors:
            transport_info = ray.get(
                [actor.initialize_transport.remote() for actor in new_actors]
            )
            for info, expected_rail in zip(
                transport_info, new_actor_rails, strict=True
            ):
                self._validate_transport(info, expected_rail)
        return selected_senders, selected_receivers

    def all_actors(self) -> list[Any]:
        return [
            actor
            for rail in self.senders
            for actor in (*self.senders[rail], *self.receivers[rail])
        ]

    def actor_count(self) -> int:
        return sum(
            len(self.senders[rail]) + len(self.receivers[rail])
            for rail in self.senders
        )

    def close(self) -> tuple[int, int]:
        return cleanup_actors(
            self.all_actors(),
            graceful_timeout_s=ACTOR_GRACEFUL_SHUTDOWN_SECONDS,
        )


def run_case(
    pool: NixlActorPool,
    suite: str,
    size_mib: int,
    flow_count: int,
    warmups: int,
    iterations: int,
) -> Any:
    nic_count = 1
    rails = rails_for_case(flow_count)
    total_flows = len(rails)
    case_id = (
        f"nixl-cpu-{suite}-{size_mib}mib-{nic_count}n-"
        f"{flow_count}fpn"
    )
    print(
        f"RUN case={case_id} suite={suite} transport=nixl device=cpu "
        f"size_mib={size_mib} nic_count={nic_count} "
        f"flows_per_nic={flow_count} total_flows={total_flows}",
        flush=True,
    )
    emit_path(
        case_id,
        "nixl",
        "cpu",
        "cxi",
        "libfabric",
        nic_count,
        (NIXL_CXI_DEVICE,),
        backend="LIBFABRIC",
        rail_policy="pinned",
        ray_control="hsn0",
    )
    senders, receivers = pool.acquire(rails)
    affinity_infos = ray.get(
        [actor.topology_info.remote() for actor in [*senders, *receivers]]
    )
    validate_and_emit_affinity(
        case_id,
        "nixl",
        "cpu",
        affinity_infos,
        rails,
    )
    nbytes = size_mib * MIB
    prepared = ray.get(
        [sender.prepare_payload.remote(nbytes) for sender in senders]
    )
    for info, expected_rail in zip(prepared, rails, strict=True):
        if info["rail"] != expected_rail or info["nbytes"] != nbytes:
            raise RuntimeError(f"{case_id}: sender payload preparation changed")
    measured = run_batches(
        senders,
        receivers,
        nbytes,
        "cpu",
        warmups,
        iterations,
    )
    emit_result(
        case_id,
        suite,
        "nixl",
        "cpu",
        size_mib,
        nic_count,
        flow_count,
        total_flows,
        warmups,
        iterations,
        measured,
    )
    print(
        f"ACTOR_POOL case={case_id} active={2 * total_flows} "
        f"retained={pool.actor_count()} status=pass",
        flush=True,
    )
    return measured


def benchmark(profile: str) -> None:
    require_environment()
    print(
        f"STACK transport=nixl ray={ray.__version__} torch={torch.__version__} "
        f"cuda={torch.version.cuda} nixl={metadata.version('nixl')} "
        "cxi_optimized_mrs="
        f"{environment_setting('FI_CXI_OPTIMIZED_MRS')} "
        "cxi_mr_cache_max_count="
        f"{environment_setting('FI_MR_CACHE_MAX_COUNT')} "
        "rail_policy=pinned",
        flush=True,
    )
    ray.init(address="auto")
    pool: NixlActorPool | None = None
    pool_forced = 0
    try:
        head, worker = select_nodes()
        validate_hsn0(head, worker)
        pool = NixlActorPool(head, worker)
        if profile == "smoke":
            run_case(pool, "smoke", 1, 1, 1, 1)
        else:
            for size_mib in BASELINE_SIZES_MIB:
                run_case(
                    pool,
                    "baseline",
                    size_mib,
                    1,
                    DEFAULT_WARMUPS,
                    DEFAULT_ITERATIONS,
                )
            flow_results = {
                flows: run_case(
                    pool,
                    "single-nic",
                    1024,
                    flows,
                    DEFAULT_WARMUPS,
                    DEFAULT_ITERATIONS,
                )
                for flows in NIXL_FLOW_COUNTS
            }
            operating_point = choose_operating_point(
                flow_results, NIXL_FLOW_COUNTS
            )
            print(
                f"OPERATING_POINT transport=nixl device=cpu "
                f"flows_per_nic={operating_point} criterion=95pct_of_sweep_max",
                flush=True,
            )
    finally:
        if pool is not None:
            retained = pool.actor_count()
            graceful, pool_forced = pool.close()
            cleanup_status = "pass" if pool_forced == 0 else "forced"
            print(
                f"ACTOR_CLEANUP scope=nixl-pool graceful={graceful} "
                f"forced={pool_forced} retained={retained} "
                f"status={cleanup_status}",
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
        arguments = parse_args()
        benchmark(arguments.profile)
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
