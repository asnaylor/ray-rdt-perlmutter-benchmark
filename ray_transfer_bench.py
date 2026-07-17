#!/usr/bin/env python3
"""Measure a CUDA tensor transfer through Ray's CPU object store."""

from __future__ import annotations

import argparse
import os
import socket
import statistics
import time
from typing import Any

import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def identity() -> dict[str, str]:
    return {
        "hostname": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        "node_id": str(ray.get_runtime_context().get_node_id()),
    }


@ray.remote(num_cpus=1, num_gpus=1)
class Sender:
    def __init__(self, nbytes: int) -> None:
        self.payload = torch.full(
            (nbytes,), 17, dtype=torch.uint8, device="cuda"
        )
        self.payload[-1] = 29
        torch.cuda.synchronize()

    def info(self) -> dict[str, str]:
        return identity()

    def send(self) -> torch.Tensor:
        # With no tensor_transport annotation, Ray serializes this CUDA tensor
        # through its CPU object store before reconstructing it on the
        # destination GPU.
        return self.payload


@ray.remote(num_cpus=1, num_gpus=1)
class Receiver:
    def info(self) -> dict[str, str]:
        return identity()

    def receive(self, payload: torch.Tensor) -> dict[str, Any]:
        if not payload.is_cuda:
            raise RuntimeError("receiver got a CPU tensor instead of CUDA")
        nbytes = payload.numel() * payload.element_size()
        checksum = int(payload[0].item()) + int(payload[-1].item())
        # Include completion of the object-store-to-GPU reconstruction in the
        # driver's elapsed time.
        torch.cuda.synchronize()
        return {
            **identity(),
            "nbytes": nbytes,
            "checksum": checksum,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a CUDA/Torch tensor through Ray's object store"
    )
    parser.add_argument("--size-mb", type=positive_int, default=1024)
    parser.add_argument("--iterations", type=positive_int, default=5)
    parser.add_argument("--warmup", type=nonnegative_int, default=1)
    return parser.parse_args()


def benchmark(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Torch cannot access CUDA in this container")

    ray.init(address="auto")
    try:
        head, worker = select_nodes()
        nbytes = args.size_mb * 1024 * 1024
        sender = Sender.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=head["NodeID"], soft=False
            )
        ).remote(nbytes)
        receiver = Receiver.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=worker["NodeID"], soft=False
            )
        ).remote()

        # Waiting for actor metadata also guarantees payload preparation is
        # complete before any warmup or timed transfer begins.
        sender_info, receiver_info = ray.get(
            [sender.info.remote(), receiver.info.remote()]
        )
        verify_placement(sender_info, str(head["NodeID"]), "sender")
        verify_placement(receiver_info, str(worker["NodeID"]), "receiver")

        print("Ray Object Store CUDA tensor benchmark", flush=True)
        print(
            f"Ray: {ray.__version__}; Torch: {torch.__version__}; "
            f"CUDA: {torch.version.cuda}",
            flush=True,
        )
        print(
            f"Source: {sender_info['hostname']} ({sender_info['node_id']})",
            flush=True,
        )
        print(
            f"Destination: {receiver_info['hostname']} "
            f"({receiver_info['node_id']})",
            flush=True,
        )
        print(
            f"Payload: {nbytes} bytes ({args.size_mb} MiB); "
            "expected checksum: 46",
            flush=True,
        )

        rates: list[float] = []
        for run in range(args.warmup + args.iterations):
            start = time.perf_counter()
            result = ray.get(receiver.receive.remote(sender.send.remote()))
            seconds = time.perf_counter() - start
            verify_result(result, nbytes)

            if run < args.warmup:
                print(
                    f"Warmup {run + 1}: {seconds:.6f} s "
                    f"({result['nbytes']} bytes, checksum {result['checksum']})",
                    flush=True,
                )
                continue

            gbps = nbytes / seconds / 1e9
            rates.append(gbps)
            print(
                f"Iteration {run - args.warmup + 1}: {seconds:.6f} s, "
                f"{gbps:.3f} GB/s ({result['nbytes']} bytes, "
                f"checksum {result['checksum']})",
                flush=True,
            )

        median_gbps = statistics.median(rates)
        print(f"Median: {median_gbps:.3f} GB/s", flush=True)
        print(
            f"RESULT benchmark=object bytes={nbytes} "
            f"median_GBps={median_gbps:.6f} "
            f"median_Gbitps={median_gbps * 8:.6f}",
            flush=True,
        )
    finally:
        ray.shutdown()


def select_nodes() -> tuple[dict[str, Any], dict[str, Any]]:
    nodes = [node for node in ray.nodes() if node.get("Alive")]
    head_id = str(ray.get_runtime_context().get_node_id())
    head = next(
        (node for node in nodes if str(node.get("NodeID")) == head_id), None
    )
    if head is None:
        raise RuntimeError("could not identify the Ray head node")

    workers = [node for node in nodes if str(node.get("NodeID")) != head_id]
    workers.sort(
        key=lambda node: (
            str(node.get("NodeManagerHostname", "")),
            str(node.get("NodeManagerAddress", "")),
            str(node.get("NodeID", "")),
        )
    )
    if not workers:
        raise RuntimeError("this benchmark requires at least two alive Ray nodes")
    return head, workers[0]


def verify_placement(info: dict[str, str], expected: str, role: str) -> None:
    if info["node_id"] != expected:
        raise RuntimeError(
            f"{role} placement mismatch: expected node {expected}, "
            f"got {info['node_id']}"
        )


def verify_result(result: dict[str, Any], nbytes: int) -> None:
    if result["nbytes"] != nbytes:
        raise RuntimeError(
            f"receiver saw {result['nbytes']} bytes; expected {nbytes}"
        )
    if result["checksum"] != 46:
        raise RuntimeError(
            f"receiver checksum was {result['checksum']}; expected 46"
        )


if __name__ == "__main__":
    benchmark(parse_args())
