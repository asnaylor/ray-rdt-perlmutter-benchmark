#!/usr/bin/env python3
"""Measure one-way CPU or CUDA tensor transfer through Ray RDT and NIXL."""

from __future__ import annotations

import argparse
import functools
import gc
import inspect
import os
import time
import uuid
from importlib import metadata
from typing import Any

import nixl
import ray

from benchmark_common import (
    PAYLOAD_EDGE_CHECKSUM,
    add_transfer_arguments,
    identity,
    make_payload,
    print_result,
    print_run,
    select_nodes,
    transfer_metadata,
    verify_full_payload,
    verify_placement,
    verify_transfer_metadata,
)


NIXL_BACKEND = os.environ.get("RAY_NIXL_BACKEND", "LIBFABRIC").upper()
EXPECTED_RAY_VERSION = "2.54.0"
EXPECTED_NIXL_VERSION = "1.3.2"


def create_forced_backend_nixl_agent(agent_name: str) -> Any:
    """Create one NIXL agent with explicit backend initialization options."""
    from nixl._api import nixl_agent, nixl_agent_config

    # nixl_agent_config cannot carry backend-specific parameters in NIXL
    # 1.3.2. Construct the agent without an implicit backend, then instantiate
    # the requested backend explicitly.
    agent = nixl_agent(agent_name, nixl_agent_config(backends=[]))
    backend_params: dict[str, str] = {}
    if NIXL_BACKEND == "UCX":
        # NIXL connects its UCX worker to itself during initialization. With
        # peer-failure handling enabled, UCX rejects its self/memory lane and
        # would force this internal connection onto TCP. Disabling endpoint
        # peer-failure handling lets self/memory serve only that local path;
        # UCX_TLS and UCX_NET_DEVICES still constrain inter-node traffic.
        parameter = "ucx_error_handling_mode"
        available_params = agent.get_plugin_params("UCX")
        if parameter not in available_params:
            raise RuntimeError(
                f"NIXL {EXPECTED_NIXL_VERSION} UCX plugin does not expose "
                f"the required {parameter!r} option: {available_params}"
            )
        backend_params[parameter] = "none"

    print(
        f"NIXL backend parameters: backend={NIXL_BACKEND}, "
        f"parameters={backend_params}",
        flush=True,
    )
    agent.create_backend(NIXL_BACKEND, backend_params)
    return agent


def install_ray_nixl_backend_override() -> None:
    """Replace Ray 2.54's hard-coded NIXL-agent backend selection."""
    if NIXL_BACKEND not in {"LIBFABRIC", "UCX"}:
        raise RuntimeError(
            "RAY_NIXL_BACKEND must be LIBFABRIC or UCX; "
            f"found {NIXL_BACKEND!r}"
        )
    ray_version = ray.__version__.split("+", 1)[0]
    nixl_version = metadata.version("nixl")
    if ray_version != EXPECTED_RAY_VERSION:
        raise RuntimeError(
            f"the NIXL override requires Ray {EXPECTED_RAY_VERSION}; "
            f"found {ray.__version__}"
        )
    if nixl_version != EXPECTED_NIXL_VERSION:
        raise RuntimeError(
            f"the NIXL override requires NIXL {EXPECTED_NIXL_VERSION}; "
            f"found {nixl_version}"
        )

    from ray.experimental.gpu_object_manager.nixl_tensor_transport import (
        NixlTensorTransport,
    )

    installed_backend = getattr(
        NixlTensorTransport, "_ray_bench_forced_backend", None
    )
    if installed_backend is not None:
        if installed_backend != NIXL_BACKEND:
            raise RuntimeError(
                "Ray NIXL backend override is already installed for "
                f"{installed_backend}, not {NIXL_BACKEND}"
            )
        return

    original_get_nixl_agent = NixlTensorTransport.get_nixl_agent
    parameters = list(inspect.signature(original_get_nixl_agent).parameters)
    if parameters != ["self"]:
        raise RuntimeError(
            "unexpected Ray NixlTensorTransport.get_nixl_agent signature: "
            f"{inspect.signature(original_get_nixl_agent)}"
        )
    original_source = inspect.getsource(original_get_nixl_agent)
    if 'nixl_agent_config(backends=["UCX"])' not in original_source:
        raise RuntimeError(
            "Ray's NIXL agent factory no longer contains the expected "
            "backends=['UCX'] configuration"
        )

    @functools.wraps(original_get_nixl_agent)
    def get_forced_backend_nixl_agent(self: Any) -> Any:
        if self._nixl_agent is not None:
            if getattr(self, "_ray_bench_backend", None) != NIXL_BACKEND:
                raise RuntimeError(
                    "Ray initialized its NIXL agent before the requested "
                    f"{NIXL_BACKEND} override was applied"
                )
            return self._nixl_agent

        actor_id = ray.get_runtime_context().get_actor_id()
        agent_name = (
            f"RAY-DRIVER-{uuid.uuid4()}"
            if actor_id is None
            else str(actor_id)
        )
        print(
            f"Ray NIXL agent initialization: backend={NIXL_BACKEND}, "
            f"agent={agent_name}",
            flush=True,
        )
        try:
            agent = create_forced_backend_nixl_agent(agent_name)
        except Exception as exc:
            # Ray's availability probe catches every exception, so emit the
            # actual reason before allowing Ray to convert it to False.
            print(
                "Ray NIXL agent initialization failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            raise

        self._nixl_agent = agent
        self._ray_bench_backend = NIXL_BACKEND
        print(
            f"Ray NIXL agent initialized: backend={NIXL_BACKEND}, "
            f"agent={agent_name}",
            flush=True,
        )
        return agent

    NixlTensorTransport.get_nixl_agent = get_forced_backend_nixl_agent
    NixlTensorTransport._ray_bench_forced_backend = NIXL_BACKEND
    print(
        f"Ray NIXL transport override installed: backend={NIXL_BACKEND}",
        flush=True,
    )


# The patch is process-local. This module is imported by the driver and by
# every actor worker, so install it before any actor can initialize NIXL.
install_ray_nixl_backend_override()

import torch  # noqa: E402
from ray.util.scheduling_strategies import (  # noqa: E402
    NodeAffinitySchedulingStrategy,
)


def initialize_ray_nixl_transport() -> dict[str, str]:
    """Eagerly initialize the same process-local NIXL manager Ray RDT uses."""
    from ray.experimental.gpu_object_manager.util import (
        get_tensor_transport_manager,
    )

    manager = get_tensor_transport_manager("NIXL")
    agent = manager.get_nixl_agent()
    return {
        **identity(),
        "manager": type(manager).__name__,
        "agent_name": str(agent.name),
    }


@ray.remote(num_cpus=1, enable_tensor_transport=True)
class Sender:
    def __init__(self, nbytes: int, device_name: str) -> None:
        install_ray_nixl_backend_override()
        self.device = torch.device(device_name)
        self.payload = make_payload(nbytes, self.device)

    def info(self) -> dict[str, str]:
        return {**identity(), "device": str(self.payload.device)}

    def initialize_transport(self) -> dict[str, str]:
        return initialize_ray_nixl_transport()

    @ray.method(tensor_transport="nixl")
    def send(self) -> torch.Tensor:
        return self.payload


@ray.remote(num_cpus=1, enable_tensor_transport=True)
class Receiver:
    def __init__(self, device_name: str) -> None:
        install_ray_nixl_backend_override()
        self.device = torch.device(device_name)
        self.payload: torch.Tensor | None = None

    def info(self) -> dict[str, str]:
        return {**identity(), "device": str(self.device)}

    def initialize_transport(self) -> dict[str, str]:
        return initialize_ray_nixl_transport()

    def receive(self, payload: torch.Tensor) -> dict[str, Any]:
        if payload.device.type != self.device.type:
            raise RuntimeError(
                f"receiver expected a {self.device.type} tensor, "
                f"but got {payload.device}"
            )
        self.payload = payload
        return {**identity(), **transfer_metadata(payload)}

    def verify_and_release(self, expected_nbytes: int) -> dict[str, Any]:
        if self.payload is None:
            raise RuntimeError("receiver has no payload to verify")

        payload = self.payload
        result = verify_full_payload(payload, expected_nbytes, self.device.type)
        self.payload = None
        del payload
        gc.collect()
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark CPU or CUDA tensors through Ray RDT and NIXL"
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    add_transfer_arguments(parser)
    return parser.parse_args()


def require_environment(device_name: str) -> None:
    ray_version = ray.__version__.split("+", 1)[0]
    nixl_version = metadata.version("nixl")
    if ray_version != EXPECTED_RAY_VERSION:
        raise RuntimeError(
            f"this backend override is validated for Ray {EXPECTED_RAY_VERSION}; "
            f"found {ray.__version__}"
        )
    if nixl_version != EXPECTED_NIXL_VERSION:
        raise RuntimeError(
            f"this benchmark is validated for NIXL {EXPECTED_NIXL_VERSION}; "
            f"found {nixl_version}"
        )
    selected_nixl_package = getattr(
        getattr(nixl, "_pkg", None), "__name__", "unknown"
    )
    if selected_nixl_package != "nixl_cu13":
        raise RuntimeError(
            "NIXL did not select its CUDA 13 implementation: "
            f"{selected_nixl_package}"
        )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Torch cannot access CUDA in this container")


def benchmark(args: argparse.Namespace) -> None:
    require_environment(args.device)
    default_mode = (
        f"rdt-nixl-cxi-{args.device}"
        if NIXL_BACKEND == "LIBFABRIC"
        else f"rdt-nixl-ucx-tcp-{args.device}"
    )
    benchmark_mode = os.environ.get("RAY_BENCH_MODE", default_mode)
    print(f"Ray NIXL Direct Transport benchmark ({benchmark_mode})", flush=True)
    print(
        f"Ray: {ray.__version__}; Torch: {torch.__version__}; "
        f"Torch CUDA: {torch.version.cuda}; NIXL: {metadata.version('nixl')}; "
        f"forced NIXL backend: {NIXL_BACKEND}",
        flush=True,
    )

    initialized = False
    try:
        ray.init(address="auto")
        initialized = True
        head, worker = select_nodes()
        nbytes = args.size_mb * 1024 * 1024
        gpu_count = 1 if args.device == "cuda" else 0
        sender = Sender.options(
            num_gpus=gpu_count,
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=head["NodeID"], soft=False
            ),
        ).remote(nbytes, args.device)
        receiver = Receiver.options(
            num_gpus=gpu_count,
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=worker["NodeID"], soft=False
            ),
        ).remote(args.device)

        sender_transport, receiver_transport = ray.get(
            [
                sender.initialize_transport.remote(),
                receiver.initialize_transport.remote(),
            ]
        )
        print(
            "Sender transport initialized: "
            f"manager={sender_transport['manager']}, "
            f"agent={sender_transport['agent_name']}",
            flush=True,
        )
        print(
            "Receiver transport initialized: "
            f"manager={receiver_transport['manager']}, "
            f"agent={receiver_transport['agent_name']}",
            flush=True,
        )

        sender_info, receiver_info = ray.get(
            [sender.info.remote(), receiver.info.remote()]
        )
        verify_placement(sender_info, str(head["NodeID"]), "sender")
        verify_placement(receiver_info, str(worker["NodeID"]), "receiver")
        if sender_info["hostname"] == receiver_info["hostname"]:
            raise RuntimeError("sender and receiver unexpectedly share a hostname")
        if args.device == "cpu":
            for role, actor_info in (
                ("sender", sender_info),
                ("receiver", receiver_info),
            ):
                visible_devices = actor_info["cuda_visible_devices"]
                if visible_devices:
                    raise RuntimeError(
                        f"CPU benchmark {role} has CUDA devices visible: "
                        f"CUDA_VISIBLE_DEVICES={visible_devices!r}"
                    )

        print(
            f"Source: {sender_info['hostname']} ({sender_info['node_id']}), "
            f"device={sender_info['device']}, "
            "CUDA_VISIBLE_DEVICES="
            f"{sender_info['cuda_visible_devices']!r}",
            flush=True,
        )
        print(
            f"Destination: {receiver_info['hostname']} "
            f"({receiver_info['node_id']}), device={receiver_info['device']}, "
            "CUDA_VISIBLE_DEVICES="
            f"{receiver_info['cuda_visible_devices']!r}",
            flush=True,
        )
        print(
            f"Payload: {nbytes} bytes ({args.size_mb} MiB); "
            f"expected edge checksum: {PAYLOAD_EDGE_CHECKSUM}",
            flush=True,
        )

        rates: list[float] = []
        durations: list[float] = []
        for run in range(args.warmup + args.iterations):
            start = time.perf_counter()
            tensor_ref = sender.send.remote()
            result_ref = receiver.receive.remote(tensor_ref)
            result = ray.get(result_ref)
            seconds = time.perf_counter() - start
            verify_transfer_metadata(result, nbytes, args.device)

            verification = ray.get(receiver.verify_and_release.remote(nbytes))
            if not verification["verified"]:
                raise RuntimeError("receiver did not verify the complete payload")
            del tensor_ref, result_ref
            gc.collect()

            rate = print_run(run, args.warmup, seconds, result["nbytes"])
            if rate is not None:
                rates.append(rate)
                durations.append(seconds)

        print_result(benchmark_mode, nbytes, rates, durations, args.warmup)
    finally:
        if initialized:
            ray.shutdown()
            print("DRIVER_SHUTDOWN_STATUS=clean", flush=True)


if __name__ == "__main__":
    try:
        benchmark(parse_args())
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
