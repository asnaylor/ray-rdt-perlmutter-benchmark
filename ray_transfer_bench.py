#!/usr/bin/env python3
"""Benchmark CPU and GPU tensors through Ray's TCP Object Store path."""

from __future__ import annotations

import argparse
import gc
from typing import Any

import ray
import torch

from benchmark_common import (
    BASELINE_SIZES_MIB,
    DEFAULT_ITERATIONS,
    DEFAULT_WARMUPS,
    FLOW_COUNTS,
    MIB,
    actor_affinity,
    choose_operating_point,
    cleanup_actors,
    cpu_actor_options,
    create_gpu_group,
    emit_path,
    emit_result,
    gpu_actor_options,
    identity,
    make_payload,
    run_batches,
    select_nodes,
    transfer_metadata,
    validate_and_emit_affinity,
    validate_hsn0,
    verify_full_payload,
)


@ray.remote
class Sender:
    def __init__(self, nbytes: int, device: str) -> None:
        self.affinity = actor_affinity("net", "hsn0", device)
        self.payload = make_payload(nbytes, torch.device(device))

    def info(self) -> dict[str, Any]:
        return {
            **identity(),
            **self.affinity,
            "device": self.payload.device.type,
        }

    def send(self) -> torch.Tensor:
        return self.payload


@ray.remote
class Receiver:
    def __init__(self, device: str) -> None:
        self.affinity = actor_affinity("net", "hsn0", device)
        self.device = torch.device(device)
        self.payload: torch.Tensor | None = None

    def info(self) -> dict[str, Any]:
        return {**identity(), **self.affinity, "device": self.device.type}

    def receive(self, payload: torch.Tensor) -> dict[str, Any]:
        if payload.device.type != self.device.type:
            payload = payload.to(self.device)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    return parser.parse_args()


def create_flows(
    head: dict[str, Any],
    worker: dict[str, Any],
    device: str,
    nbytes: int,
    flows: int,
) -> tuple[list[Any], list[Any], list[Any]]:
    groups: list[Any] = []
    senders: list[Any] = []
    receivers: list[Any] = []
    if device == "cuda":
        head_group = create_gpu_group(head, 1, flows)
        worker_group = create_gpu_group(worker, 1, flows)
        groups.extend((head_group, worker_group))
        for _ in range(flows):
            senders.append(
                Sender.options(**gpu_actor_options(head_group, 0, flows)).remote(
                    nbytes, device
                )
            )
            receivers.append(
                Receiver.options(**gpu_actor_options(worker_group, 0, flows)).remote(
                    device
                )
            )
    else:
        for _ in range(flows):
            senders.append(
                Sender.options(**cpu_actor_options(head)).remote(nbytes, device)
            )
            receivers.append(
                Receiver.options(**cpu_actor_options(worker)).remote(device)
            )
    return senders, receivers, groups


def run_case(
    head: dict[str, Any],
    worker: dict[str, Any],
    suite: str,
    device: str,
    size_mib: int,
    flows: int,
    warmups: int,
    iterations: int,
) -> Any:
    case_id = f"object-{device}-{suite}-{size_mib}mib-{flows}f"
    public_device = "gpu" if device == "cuda" else "cpu"
    print(
        f"RUN case={case_id} suite={suite} transport=object "
        f"device={public_device} size_mib={size_mib} total_flows={flows}",
        flush=True,
    )
    emit_path(
        case_id,
        "object",
        public_device,
        "tcp",
        "ray-object-manager",
        1,
        ("hsn0",),
        ray_control="hsn0",
        staging="host" if device == "cuda" else "none",
    )
    senders, receivers, groups = create_flows(
        head, worker, device, size_mib * MIB, flows
    )
    try:
        infos = ray.get(
            [actor.info.remote() for actor in [*senders, *receivers]]
        )
        expected_nodes = [str(head["NodeID"])] * flows + [
            str(worker["NodeID"])
        ] * flows
        if any(info["node_id"] != expected for info, expected in zip(
            infos, expected_nodes, strict=True
        )):
            raise RuntimeError("Object Store actor placement did not match the plan")
        if device == "cuda":
            for node_infos in (infos[:flows], infos[flows:]):
                gpu_ids = {
                    str(gpu)
                    for info in node_infos
                    for gpu in info["accelerator_ids"]
                }
                if len(gpu_ids) != 1:
                    raise RuntimeError(
                        "single-interface Object Store flows did not share one GPU"
                    )
        validate_and_emit_affinity(
            case_id,
            "object",
            public_device,
            infos,
            ["hsn0"] * flows,
        )
        measured = run_batches(
            senders,
            receivers,
            size_mib * MIB,
            device,
            warmups,
            iterations,
        )
        emit_result(
            case_id,
            suite,
            "object",
            public_device,
            size_mib,
            1,
            flows,
            flows,
            warmups,
            iterations,
            measured,
        )
        return measured
    finally:
        cleanup_actors([*senders, *receivers], groups)


def benchmark(profile: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Torch cannot access CUDA in this container")
    print(
        f"STACK transport=object ray={ray.__version__} torch={torch.__version__} "
        f"cuda={torch.version.cuda}",
        flush=True,
    )
    ray.init(address="auto")
    try:
        head, worker = select_nodes()
        validate_hsn0(head, worker)
        if profile == "smoke":
            for device in ("cpu", "cuda"):
                run_case(head, worker, "smoke", device, 1, 1, 1, 1)
            return

        for device in ("cpu", "cuda"):
            for size_mib in BASELINE_SIZES_MIB:
                run_case(
                    head,
                    worker,
                    "baseline",
                    device,
                    size_mib,
                    1,
                    DEFAULT_WARMUPS,
                    DEFAULT_ITERATIONS,
                )
            flow_results = {
                flows: run_case(
                    head,
                    worker,
                    "single-nic",
                    device,
                    1024,
                    flows,
                    DEFAULT_WARMUPS,
                    DEFAULT_ITERATIONS,
                )
                for flows in FLOW_COUNTS
            }
            operating_point = choose_operating_point(flow_results)
            public_device = "gpu" if device == "cuda" else "cpu"
            print(
                f"OPERATING_POINT transport=object device={public_device} "
                f"flows_per_nic={operating_point} criterion=95pct_of_sweep_max",
                flush=True,
            )
    finally:
        ray.shutdown()
        print("SHUTDOWN status=clean transport=object", flush=True)


if __name__ == "__main__":
    try:
        benchmark(parse_args().profile)
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
