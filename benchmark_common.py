#!/usr/bin/env python3
"""Shared mechanics for the Ray transfer benchmark drivers."""

from __future__ import annotations

import argparse
import os
import socket
import statistics
from typing import Any

import ray
import torch


PAYLOAD_VALUE = 17
PAYLOAD_LAST_VALUE = 29
PAYLOAD_EDGE_CHECKSUM = PAYLOAD_VALUE + PAYLOAD_LAST_VALUE


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


def add_transfer_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--size-mb", type=positive_int, default=1024)
    parser.add_argument("--iterations", type=positive_int, default=5)
    parser.add_argument("--warmup", type=nonnegative_int, default=1)


def identity() -> dict[str, str]:
    return {
        "hostname": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        "node_id": str(ray.get_runtime_context().get_node_id()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


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
    """Synchronize transfer completion and return cheap timed metadata."""
    nbytes = payload.numel() * payload.element_size()
    edge_checksum = int(payload[0].item()) + int(payload[-1].item())
    synchronize(payload.device)
    return {
        "nbytes": nbytes,
        "edge_checksum": edge_checksum,
        "device": str(payload.device),
    }


def verify_transfer_metadata(
    result: dict[str, Any], expected_nbytes: int, expected_device: str
) -> None:
    if result["nbytes"] != expected_nbytes:
        raise RuntimeError(
            f"receiver saw {result['nbytes']} bytes; expected {expected_nbytes}"
        )
    if result["edge_checksum"] != PAYLOAD_EDGE_CHECKSUM:
        raise RuntimeError(
            "receiver edge checksum was "
            f"{result['edge_checksum']}; expected {PAYLOAD_EDGE_CHECKSUM}"
        )
    if not str(result["device"]).startswith(expected_device):
        raise RuntimeError(
            f"receiver tensor device was {result['device']}; "
            f"expected {expected_device}"
        )


def verify_full_payload(
    payload: torch.Tensor, expected_nbytes: int, expected_device: str
) -> dict[str, Any]:
    """Validate every payload byte; callers run this outside the timer."""
    actual_nbytes = payload.numel() * payload.element_size()
    body_is_valid = bool(torch.all(payload[:-1] == PAYLOAD_VALUE).item())
    last_value = int(payload[-1].item())
    synchronize(payload.device)

    if actual_nbytes != expected_nbytes:
        raise RuntimeError(
            f"receiver saw {actual_nbytes} bytes; expected {expected_nbytes}"
        )
    if not str(payload.device).startswith(expected_device):
        raise RuntimeError(
            f"receiver tensor device was {payload.device}; "
            f"expected {expected_device}"
        )
    if not body_is_valid or last_value != PAYLOAD_LAST_VALUE:
        raise RuntimeError("full payload verification failed")
    return {"nbytes": actual_nbytes, "verified": True}


def print_run(
    run: int,
    warmup: int,
    seconds: float,
    nbytes: int,
) -> float | None:
    if run < warmup:
        print(
            f"Warmup {run + 1}: {seconds:.6f} s "
            f"({nbytes} bytes, fully verified)",
            flush=True,
        )
        return None

    gbps = nbytes / seconds / 1e9
    print(
        f"Iteration {run - warmup + 1}: {seconds:.6f} s, "
        f"{gbps:.3f} GB/s ({nbytes} bytes, fully verified)",
        flush=True,
    )
    return gbps


def print_result(
    benchmark_mode: str,
    nbytes: int,
    rates: list[float],
    durations: list[float],
    warmup: int,
) -> None:
    if not rates or len(rates) != len(durations):
        raise RuntimeError("measured rates and durations are missing or inconsistent")

    median_gbps = statistics.median(rates)
    median_ms = statistics.median(durations) * 1000
    print(
        f"Median: {median_ms:.3f} ms, {median_gbps:.3f} GB/s",
        flush=True,
    )
    print(
        f"RESULT benchmark={benchmark_mode} bytes={nbytes} "
        f"warmup={warmup} iterations={len(durations)} "
        f"median_ms={median_ms:.6f} "
        f"median_GBps={median_gbps:.6f} "
        f"median_Gbitps={median_gbps * 8:.6f} transfer_status=pass",
        flush=True,
    )
