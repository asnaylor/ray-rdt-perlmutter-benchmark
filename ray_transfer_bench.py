#!/usr/bin/env python3
"""Measure a CUDA tensor transfer through Ray's CPU object store."""

from __future__ import annotations

import argparse
import gc
import time
from typing import Any

import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

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


@ray.remote(num_cpus=1, num_gpus=1)
class Sender:
    def __init__(self, nbytes: int) -> None:
        self.payload = make_payload(nbytes, torch.device("cuda"))

    def info(self) -> dict[str, str]:
        return {**identity(), "device": str(self.payload.device)}

    def send(self) -> torch.Tensor:
        # Without a tensor_transport annotation, Ray serializes through its
        # CPU object store before reconstructing the tensor on the receiver.
        return self.payload


@ray.remote(num_cpus=1, num_gpus=1)
class Receiver:
    def __init__(self) -> None:
        self.payload: torch.Tensor | None = None

    def info(self) -> dict[str, str]:
        return {**identity(), "device": "cuda"}

    def receive(self, payload: torch.Tensor) -> dict[str, Any]:
        if not payload.is_cuda:
            raise RuntimeError("receiver got a CPU tensor instead of CUDA")
        self.payload = payload
        return {**identity(), **transfer_metadata(payload)}

    def verify_and_release(self, expected_nbytes: int) -> dict[str, Any]:
        if self.payload is None:
            raise RuntimeError("receiver has no payload to verify")
        payload = self.payload
        result = verify_full_payload(payload, expected_nbytes, "cuda")
        self.payload = None
        del payload
        gc.collect()
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a CUDA tensor through Ray's object store"
    )
    add_transfer_arguments(parser)
    return parser.parse_args()


def benchmark(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Torch cannot access CUDA in this container")

    initialized = False
    try:
        ray.init(address="auto")
        initialized = True
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

        sender_info, receiver_info = ray.get(
            [sender.info.remote(), receiver.info.remote()]
        )
        verify_placement(sender_info, str(head["NodeID"]), "sender")
        verify_placement(receiver_info, str(worker["NodeID"]), "receiver")
        if sender_info["hostname"] == receiver_info["hostname"]:
            raise RuntimeError("sender and receiver unexpectedly share a hostname")

        print("Ray Object Store CUDA tensor benchmark", flush=True)
        print(
            f"Ray: {ray.__version__}; Torch: {torch.__version__}; "
            f"CUDA: {torch.version.cuda}",
            flush=True,
        )
        print(
            f"Source: {sender_info['hostname']} ({sender_info['node_id']}), "
            f"device={sender_info['device']}",
            flush=True,
        )
        print(
            f"Destination: {receiver_info['hostname']} "
            f"({receiver_info['node_id']}), device={receiver_info['device']}",
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
            verify_transfer_metadata(result, nbytes, "cuda")

            verification = ray.get(receiver.verify_and_release.remote(nbytes))
            if not verification["verified"]:
                raise RuntimeError("receiver did not verify the complete payload")
            del tensor_ref, result_ref
            gc.collect()

            rate = print_run(run, args.warmup, seconds, result["nbytes"])
            if rate is not None:
                rates.append(rate)
                durations.append(seconds)

        print_result("object", nbytes, rates, durations, args.warmup)
    finally:
        if initialized:
            ray.shutdown()
            print("DRIVER_SHUTDOWN_STATUS=clean", flush=True)


if __name__ == "__main__":
    try:
        benchmark(parse_args())
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
