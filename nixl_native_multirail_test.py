#!/usr/bin/env python3
"""Probe one NIXL LIBFABRIC agent using four CXI rails."""

from __future__ import annotations

import os
import time
from typing import Any

import ray
import torch

from benchmark_common import (
    MIB,
    cleanup_actors,
    cpu_actor_options,
    format_cpu_list,
    identity,
    make_payload,
    select_nodes,
    transfer_metadata,
    validate_hsn0,
    verify_full_payload,
    verify_transfer_metadata,
)
from ray_nixl_bench import (
    NIXL_BACKEND,
    install_ray_nixl_backend_override,
    require_environment,
)


EXPECTED_DEVICES = ("cxi3", "cxi2", "cxi1", "cxi0")
SIZE_MIB = 1024
OPERATION_TIMEOUT_SECONDS = 120
ACTOR_CLEANUP_SECONDS = 30


def selected_devices() -> tuple[str, ...]:
    return tuple(
        item.strip()
        for item in os.environ.get("FI_CXI_DEVICE_NAME", "").split(",")
        if item.strip()
    )


def initialize_transport(role: str) -> dict[str, Any]:
    from ray.experimental.gpu_object_manager.util import (
        get_tensor_transport_manager,
    )

    manager = get_tensor_transport_manager("NIXL")
    agent = manager.get_nixl_agent()
    return {
        **identity(),
        "role": role,
        "manager": type(manager).__name__,
        "agent": str(agent.name),
        "backend": NIXL_BACKEND,
        "devices": ",".join(selected_devices()),
        "cpus": format_cpu_list(os.sched_getaffinity(0)),
    }


@ray.remote(enable_tensor_transport=True)
class Sender:
    def __init__(self, nbytes: int) -> None:
        install_ray_nixl_backend_override()
        self.payload = make_payload(nbytes, torch.device("cpu"))

    def initialize_transport(self) -> dict[str, Any]:
        return initialize_transport("sender")

    @ray.method(tensor_transport="nixl")
    def send(self) -> torch.Tensor:
        return self.payload


@ray.remote(enable_tensor_transport=True)
class Receiver:
    def __init__(self) -> None:
        install_ray_nixl_backend_override()
        self.payload: torch.Tensor | None = None

    def initialize_transport(self) -> dict[str, Any]:
        return initialize_transport("receiver")

    def receive(self, payload: torch.Tensor) -> dict[str, Any]:
        if payload.device.type != "cpu":
            raise RuntimeError(f"receiver got {payload.device}; expected CPU")
        self.payload = payload
        return {**identity(), **transfer_metadata(payload)}

    def verify_and_release(self, expected_nbytes: int) -> dict[str, Any]:
        if self.payload is None:
            raise RuntimeError("receiver has no payload to verify")
        payload = self.payload
        result = verify_full_payload(payload, expected_nbytes, "cpu")
        self.payload = None
        return result


def benchmark() -> None:
    require_environment()
    devices = selected_devices()
    if devices != EXPECTED_DEVICES:
        raise RuntimeError(
            f"FI_CXI_DEVICE_NAME={devices}; expected {EXPECTED_DEVICES}"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be empty for this CPU test")

    nbytes = SIZE_MIB * MIB
    print(
        "RUN test=nixl-native-multirail "
        f"size_mib={SIZE_MIB} devices={','.join(devices)} "
        "agents_per_node=1",
        flush=True,
    )
    ray.init(address="auto")
    actors: list[Any] = []
    forced = 0
    try:
        head, worker = select_nodes()
        validate_hsn0(head, worker)
        sender = Sender.options(**cpu_actor_options(head)).remote(nbytes)
        receiver = Receiver.options(**cpu_actor_options(worker)).remote()
        actors.extend((sender, receiver))

        infos = ray.get(
            [
                sender.initialize_transport.remote(),
                receiver.initialize_transport.remote(),
            ],
            timeout=OPERATION_TIMEOUT_SECONDS,
        )
        for info in infos:
            if (
                info["manager"] != "NixlTensorTransport"
                or info["backend"] != "LIBFABRIC"
                or info["devices"] != ",".join(EXPECTED_DEVICES)
                or info["accelerator_ids"]
            ):
                raise RuntimeError(f"unexpected NIXL agent state: {info}")
            print(
                f"MULTIRAIL_AGENT role={info['role']} "
                f"hostname={info['hostname']} agent={info['agent']} "
                f"devices={info['devices']} cpus={info['cpus']} status=pass",
                flush=True,
            )

        started = time.perf_counter()
        tensor_ref = sender.send.remote()
        result = ray.get(
            receiver.receive.remote(tensor_ref),
            timeout=OPERATION_TIMEOUT_SECONDS,
        )
        elapsed = time.perf_counter() - started
        verify_transfer_metadata(result, nbytes, "cpu")
        verification = ray.get(
            receiver.verify_and_release.remote(nbytes),
            timeout=OPERATION_TIMEOUT_SECONDS,
        )
        if not verification["verified"]:
            raise RuntimeError("receiver did not verify the complete payload")
        print(
            "RESULT test=nixl-native-multirail "
            f"bytes={nbytes} elapsed_ms={elapsed * 1000:.6f} "
            "validation_status=pass",
            flush=True,
        )
    finally:
        if actors:
            graceful, forced = cleanup_actors(
                actors,
                graceful_timeout_s=ACTOR_CLEANUP_SECONDS,
            )
            print(
                "ACTOR_CLEANUP scope=nixl-native-multirail "
                f"graceful={graceful} forced={forced} status="
                f"{'pass' if forced == 0 else 'forced'}",
                flush=True,
            )
        ray.shutdown()
        print("SHUTDOWN status=clean test=nixl-native-multirail", flush=True)
    if forced:
        raise RuntimeError(f"{forced} NIXL actors required forced cleanup")


if __name__ == "__main__":
    try:
        benchmark()
    except Exception as exc:
        raise SystemExit(f"ERROR: {type(exc).__name__}: {exc}") from exc
