#!/usr/bin/env python3
"""Measure one-way Ray Direct Transport transfer over NCCL."""

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


# Defining a tensor-transport method fails at import time on older Ray builds.
# Preserve that error and turn it into a concise image/API diagnostic in main.
RDT_API_ERROR: Exception | None = None
try:
    from ray.experimental.collective import create_collective_group

    # Both actors need Ray's private tensor-transport concurrency groups. In
    # particular, the receiver has no decorated producer method from which Ray
    # could infer this and would otherwise block its internal receive task.
    @ray.remote(num_cpus=1, num_gpus=1, enable_tensor_transport=True)
    class Sender:
        def __init__(self, nbytes: int) -> None:
            self.payload = torch.full(
                (nbytes,), 17, dtype=torch.uint8, device="cuda"
            )
            self.payload[-1] = 29
            torch.cuda.synchronize()

        def info(self) -> dict[str, str]:
            return identity()

        @ray.method(tensor_transport="nccl")
        def send(self) -> torch.Tensor:
            return self.payload


    @ray.remote(num_cpus=1, num_gpus=1, enable_tensor_transport=True)
    class Receiver:
        def info(self) -> dict[str, str]:
            return identity()

        def receive(self, payload: torch.Tensor) -> dict[str, Any]:
            if not payload.is_cuda:
                raise RuntimeError("receiver got a CPU tensor instead of CUDA")
            nbytes = payload.numel() * payload.element_size()
            checksum = int(payload[0].item()) + int(payload[-1].item())
            # The driver's timer stops only after this synchronized actor task
            # has returned, so it includes completion of the NCCL transfer.
            torch.cuda.synchronize()
            return {**identity(), "nbytes": nbytes, "checksum": checksum}

except Exception as exc:
    RDT_API_ERROR = exc
    Sender = None  # type: ignore[assignment,misc]
    Receiver = None  # type: ignore[assignment,misc]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a CUDA/Torch Ray Direct Transport NCCL transfer"
    )
    parser.add_argument("--size-mb", type=positive_int, default=1024)
    parser.add_argument("--iterations", type=positive_int, default=5)
    parser.add_argument("--warmup", type=nonnegative_int, default=1)
    return parser.parse_args()


def nccl_version() -> str:
    try:
        return str(torch.cuda.nccl.version())
    except Exception as exc:
        return f"unavailable ({exc})"


def require_rdt() -> None:
    if RDT_API_ERROR is not None:
        raise RuntimeError(
            "this image's Ray build lacks the alpha NCCL Ray Direct "
            "Transport API (ray.method(tensor_transport=...) and "
            "ray.experimental.collective.create_collective_group); choose "
            f"a newer IMAGE. Original error: {RDT_API_ERROR}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Torch cannot access CUDA in this container")
    if nccl_version().startswith("unavailable"):
        raise RuntimeError(f"Torch NCCL support is {nccl_version()}")


def benchmark(args: argparse.Namespace) -> None:
    benchmark_mode = os.environ.get("RAY_BENCH_MODE", "rdt")
    print(f"Ray NCCL Direct Transport benchmark ({benchmark_mode})", flush=True)
    print(
        f"Ray: {ray.__version__}; Torch: {torch.__version__}; "
        f"CUDA: {torch.version.cuda}; NCCL: {nccl_version()}",
        flush=True,
    )
    require_rdt()

    ray.init(address="auto")
    try:
        head, worker = select_nodes()
        nbytes = args.size_mb * 1024 * 1024
        assert Sender is not None and Receiver is not None
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

        # Actor construction allocates and synchronizes the payload. Metadata
        # calls ensure that work is complete before collective setup and timing.
        sender_info, receiver_info = ray.get(
            [sender.info.remote(), receiver.info.remote()]
        )
        verify_placement(sender_info, str(head["NodeID"]), "sender")
        verify_placement(receiver_info, str(worker["NodeID"]), "receiver")
        if sender_info["hostname"] == receiver_info["hostname"]:
            raise RuntimeError("sender and receiver unexpectedly share a hostname")

        # NCCL RDT refs can only be consumed by actors in this collective. Keep
        # the returned group handle alive for the duration of the benchmark.
        try:
            group = create_collective_group([sender, receiver], backend="nccl")
        except Exception as exc:
            raise RuntimeError(
                "could not create the NCCL RDT collective; verify that the "
                "selected IMAGE provides Ray's alpha RDT API and NCCL support: "
                f"{exc}"
            ) from exc

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
            tensor_ref = sender.send.remote()
            result_ref = receiver.receive.remote(tensor_ref)
            result = ray.get(result_ref)
            seconds = time.perf_counter() - start
            verify_result(result, nbytes)

            # Never resolve tensor_ref on the driver: NCCL RDT ObjectRefs must
            # pass directly between actors in their collective group.
            del tensor_ref, result_ref
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
            f"RESULT benchmark={benchmark_mode} bytes={nbytes} "
            f"median_GBps={median_gbps:.6f} "
            f"median_Gbitps={median_gbps * 8:.6f}",
            flush=True,
        )
        del group
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
    try:
        benchmark(parse_args())
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
