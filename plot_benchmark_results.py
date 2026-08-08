#!/usr/bin/env python3
"""Validate benchmark logs, write canonical CSV results, and plot throughput."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SIZES_MIB = (1, 64, 1024)
PROFILE_BY_SIZE = {1: (5, 31), 64: (3, 15), 1024: (2, 7)}
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class ResultError(RuntimeError):
    """A benchmark log is incomplete, malformed, or inconsistent."""


@dataclass(frozen=True)
class Series:
    key: str
    panel: str
    label: str
    mode: str
    rails: str
    payload: str
    network: str
    evidence: tuple[tuple[str, str], ...]
    color: str
    marker: str
    linestyle: str


SERIES = (
    Series(
        "object",
        "gpu",
        "Ray Object Store",
        "object",
        "",
        "cuda",
        "provider not asserted",
        (("path", "ray-object-store"), ("source", "cuda"), ("destination", "cuda")),
        "#666666",
        "o",
        "--",
    ),
    Series(
        "rdt-tcp",
        "gpu",
        "NCCL / TCP (hsn0)",
        "rdt-tcp",
        "",
        "cuda",
        "hsn0",
        (("backend", "NCCL"), ("provider", "tcp"), ("device", "hsn0")),
        "#D55E00",
        "s",
        "--",
    ),
    Series(
        "rdt-cxi-1",
        "gpu",
        "NCCL / CXI (cxi3)",
        "rdt-cxi",
        "1",
        "cuda",
        "cxi3",
        (("backend", "NCCL"), ("provider", "cxi"), ("rails", "1"), ("devices", "cxi3")),
        "#56B4E9",
        "^",
        "-",
    ),
    Series(
        "rdt-nixl-ucx-tcp-cpu",
        "cpu",
        "NIXL / UCX TCP (hsn0)",
        "rdt-nixl-ucx-tcp-cpu",
        "",
        "cpu",
        "hsn0",
        (("backend", "UCX"), ("transport", "tcp"), ("device", "hsn0")),
        "#D55E00",
        "s",
        "--",
    ),
    Series(
        "rdt-nixl-cxi-cpu-1",
        "cpu",
        "NIXL / CXI (cxi3)",
        "rdt-nixl-cxi-cpu",
        "1",
        "cpu",
        "cxi3",
        (("backend", "LIBFABRIC"), ("provider", "cxi"), ("rails", "1"), ("devices", "cxi3")),
        "#56B4E9",
        "^",
        "-",
    ),
)

SERIES_BY_MODE_RAILS = {(item.mode, item.rails): item for item in SERIES}
SERIES_ORDER = {item.key: index for index, item in enumerate(SERIES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="complete matrix logs")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/benchmark-results.csv"),
        help="canonical CSV output",
    )
    parser.add_argument(
        "--svg",
        type=Path,
        default=Path("docs/benchmark-throughput.svg"),
        help="two-panel SVG output",
    )
    return parser.parse_args()


def extract_record(text: str, marker: str, source: Path) -> str:
    records: list[str] = []
    for raw_line in text.splitlines():
        line = ANSI_ESCAPE.sub("", raw_line)
        offset = line.find(marker)
        if offset >= 0 and (offset == 0 or line[offset - 1].isspace()):
            records.append(line[offset:].strip())
    if len(records) != 1:
        raise ResultError(
            f"{source}: expected exactly one {marker.strip()} record, found {len(records)}"
        )
    return records[0]


def parse_fields(record: str, record_name: str, source: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in record.split()[1:]:
        if "=" not in token:
            raise ResultError(f"{source}: malformed {record_name} token {token!r}")
        key, value = token.split("=", 1)
        if not key or not value or key in fields:
            raise ResultError(f"{source}: invalid {record_name} field {token!r}")
        fields[key] = value
    return fields


def required(fields: dict[str, str], name: str, record: str, source: Path) -> str:
    try:
        return fields[name]
    except KeyError as exc:
        raise ResultError(f"{source}: {record} is missing {name}") from exc


def parse_positive_float(
    fields: dict[str, str], name: str, record: str, source: Path
) -> float:
    raw_value = required(fields, name, record, source)
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ResultError(f"{source}: {record} {name} is not numeric") from exc
    if not math.isfinite(value) or value <= 0:
        raise ResultError(f"{source}: {record} {name} must be positive and finite")
    return value


def parse_nonnegative_int(
    fields: dict[str, str], name: str, record: str, source: Path
) -> int:
    raw_value = required(fields, name, record, source)
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ResultError(f"{source}: {record} {name} is not an integer") from exc
    if value < 0:
        raise ResultError(f"{source}: {record} {name} must not be negative")
    return value


def parse_log(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ResultError(f"could not read {path}: {exc}") from exc

    result = parse_fields(extract_record(text, "RESULT ", path), "RESULT", path)
    evidence = parse_fields(
        extract_record(text, "EVIDENCE ", path), "EVIDENCE", path
    )
    shutdown = extract_record(text, "SHUTDOWN_STATUS=", path)
    if shutdown != "SHUTDOWN_STATUS=clean":
        raise ResultError(f"{path}: benchmark shutdown was not clean")
    if required(result, "transfer_status", "RESULT", path) != "pass":
        raise ResultError(f"{path}: payload verification did not pass")
    if required(evidence, "status", "EVIDENCE", path) != "pass":
        raise ResultError(f"{path}: transport evidence did not pass")

    mode = required(result, "benchmark", "RESULT", path)
    rails = evidence.get("rails", "")
    try:
        series = SERIES_BY_MODE_RAILS[(mode, rails)]
    except KeyError as exc:
        raise ResultError(
            f"{path}: unexpected benchmark/evidence combination mode={mode} rails={rails or '<none>'}"
        ) from exc
    for name, expected in series.evidence:
        actual = evidence.get(name)
        if actual != expected:
            raise ResultError(
                f"{path}: evidence {name}={actual!r}; expected {expected!r}"
            )

    raw_bytes = required(result, "bytes", "RESULT", path)
    try:
        nbytes = int(raw_bytes)
    except ValueError as exc:
        raise ResultError(f"{path}: RESULT bytes is not an integer") from exc
    mib, remainder = divmod(nbytes, 1024 * 1024)
    if remainder or mib not in SIZES_MIB:
        raise ResultError(
            f"{path}: payload is {nbytes} bytes; expected 1, 64, or 1024 MiB"
        )

    warmup = parse_nonnegative_int(result, "warmup", "RESULT", path)
    iterations = parse_nonnegative_int(result, "iterations", "RESULT", path)
    expected_warmup, expected_iterations = PROFILE_BY_SIZE[mib]
    if (warmup, iterations) != (expected_warmup, expected_iterations):
        raise ResultError(
            f"{path}: sampling profile is warmup={warmup} iterations={iterations}; "
            f"expected warmup={expected_warmup} iterations={expected_iterations}"
        )

    median_ms = parse_positive_float(result, "median_ms", "RESULT", path)
    median_gbps = parse_positive_float(result, "median_GBps", "RESULT", path)
    median_gbitps = parse_positive_float(result, "median_Gbitps", "RESULT", path)
    expected_gbps = nbytes / (median_ms / 1000) / 1e9
    if not math.isclose(median_gbps, expected_gbps, rel_tol=2e-5):
        raise ResultError(
            f"{path}: median_ms and median_GBps are inconsistent"
        )
    # Both values are independently rounded to six decimal places in RESULT.
    # Multiplying the rounded GB/s value by eight can therefore differ from
    # the rounded Gbit/s value by as much as 4.5e-6.
    if not math.isclose(
        median_gbitps, median_gbps * 8, rel_tol=0, abs_tol=5e-6
    ):
        raise ResultError(f"{path}: median_Gbitps is not 8x median_GBps")

    return {
        "series": series.key,
        "mode": mode,
        "payload": series.payload,
        "network_selection": series.network,
        "size_mib": mib,
        "bytes": nbytes,
        "warmup": warmup,
        "iterations": iterations,
        "median_ms": median_ms,
        "median_GBps": median_gbps,
        "median_Gbitps": median_gbitps,
        "source_log": path.name,
    }


def load_matrix(paths: list[Path]) -> list[dict[str, Any]]:
    unique_paths = {path.resolve() for path in paths}
    if len(unique_paths) != len(paths):
        raise ResultError("the input log list contains duplicate paths")
    rows = [parse_log(path) for path in sorted(unique_paths)]
    keyed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["series"]), int(row["size_mib"]))
        if key in keyed:
            raise ResultError(
                f"duplicate result for series={key[0]} size_mib={key[1]}"
            )
        keyed[key] = row

    expected = {(item.key, size) for item in SERIES for size in SIZES_MIB}
    missing = expected - set(keyed)
    unexpected = set(keyed) - expected
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(
                "missing "
                + ", ".join(f"{series}/{size}MiB" for series, size in sorted(missing))
            )
        if unexpected:
            details.append(
                "unexpected "
                + ", ".join(
                    f"{series}/{size}MiB" for series, size in sorted(unexpected)
                )
            )
        raise ResultError("incomplete benchmark matrix: " + "; ".join(details))

    return sorted(
        rows,
        key=lambda row: (SERIES_ORDER[str(row["series"])], int(row["size_mib"])),
    )


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = (
        "series",
        "mode",
        "payload",
        "network_selection",
        "size_mib",
        "bytes",
        "warmup",
        "iterations",
        "median_ms",
        "median_GBps",
        "median_Gbitps",
        "source_log",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for name in ("median_ms", "median_GBps", "median_Gbitps"):
                output[name] = f"{float(row[name]):.6f}"
            writer.writerow(output)


def write_plot(rows: list[dict[str, Any]], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        matplotlib.rcParams["svg.hashsalt"] = "ray-perlmutter-benchmark"
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ResultError(
            "Matplotlib is required; run this tool inside the benchmark image"
        ) from exc

    row_by_key = {
        (str(row["series"]), int(row["size_mib"])): row for row in rows
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for axis, panel, title in zip(
        axes,
        ("gpu", "cpu"),
        ("CUDA payloads", "CPU payloads"),
        strict=True,
    ):
        for item in (candidate for candidate in SERIES if candidate.panel == panel):
            values = [row_by_key[(item.key, size)] for size in SIZES_MIB]
            axis.plot(
                SIZES_MIB,
                [float(value["median_GBps"]) for value in values],
                label=item.label,
                color=item.color,
                marker=item.marker,
                linestyle=item.linestyle,
                linewidth=2,
                markersize=6,
            )
        axis.set_title(title)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks(SIZES_MIB, ("1 MiB", "64 MiB", "1 GiB"))
        axis.grid(True, which="both", color="#d9d9d9", linewidth=0.7)
        axis.legend(fontsize=8, frameon=False)

    axes[0].set_ylabel("Median effective throughput (GB/s)")
    fig.suptitle("Two-node Ray tensor-transfer scaling on Perlmutter")
    fig.supxlabel("Payload size")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        format="svg",
        metadata={"Creator": "plot_benchmark_results.py", "Date": None},
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_matrix(args.logs)
    write_csv(rows, args.csv)
    write_plot(rows, args.svg)
    print(f"RESULTS_CSV={args.csv}")
    print(f"RESULTS_SVG={args.svg}")
    print(f"RESULTS_RECORDS={len(rows)}")


if __name__ == "__main__":
    try:
        main()
    except ResultError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
