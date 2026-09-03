"""Dependency-free statistics used by the benchmark and its tests."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable, Mapping


FLOW_COUNTS = (1, 2, 4, 8)


@dataclass(frozen=True)
class Measurements:
    median_ms: float
    p95_ms: float
    median_gbps: float


def nearest_rank(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise RuntimeError("cannot calculate a percentile without samples")
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def summarize(durations: list[float], total_bytes: int) -> Measurements:
    if not durations:
        raise RuntimeError("no measured durations were collected")
    median_duration = statistics.median(durations)
    return Measurements(
        median_ms=median_duration * 1000,
        p95_ms=nearest_rank(durations, 0.95) * 1000,
        median_gbps=total_bytes / median_duration / 1e9,
    )


def choose_operating_point(
    results: Mapping[int, Measurements],
    flow_counts: Iterable[int] = FLOW_COUNTS,
) -> int:
    expected = tuple(flow_counts)
    if not expected or len(expected) != len(set(expected)):
        raise RuntimeError("the operating-point flow sweep is invalid")
    if set(results) != set(expected):
        raise RuntimeError("the operating point requires the complete flow sweep")
    threshold = max(item.median_gbps for item in results.values()) * 0.95
    return next(
        flows for flows in expected if results[flows].median_gbps >= threshold
    )
