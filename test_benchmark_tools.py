#!/usr/bin/env python3
"""Unit tests for deterministic benchmark statistics and result handling."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import plot_benchmark_results as publisher
from benchmark_stats import (
    Measurements,
    choose_operating_point,
    nearest_rank,
    summarize,
)
from plot_benchmark_results import (
    ResultError,
    atomic_csv,
    load_matrix,
    validate_transport_log,
)


SIZES_MIB = (1, 64, 1024)
FLOW_COUNTS = (1, 2, 4, 8)
NIXL_FLOW_COUNTS = (1, 2, 4)
NIXL_FOUR_RAIL_FLOW_COUNTS = (1, 2)
TRANSPORT_SERIES = {
    "object": (("object", "cpu"), ("object", "gpu")),
    "nixl": (("nixl", "cpu"),),
    "nccl": (("nccl", "gpu"),),
}
RAILS = {
    "nixl": ("cxi3", "cxi2", "cxi1", "cxi0"),
    "nccl": ("cxi3", "cxi2", "cxi1", "cxi0"),
}
TARGET_PCI = {
    "hsn0": "0000:c2:00.0",
    "cxi3": "0000:01:00.0",
    "cxi2": "0000:42:00.0",
    "cxi1": "0000:81:00.0",
    "cxi0": "0000:c2:00.0",
}
TARGET_NUMA = {"hsn0": 0, "cxi3": 3, "cxi2": 2, "cxi1": 1, "cxi0": 0}
CPU_SETS = {
    0: "0-15+64-79",
    1: "16-31+80-95",
    2: "32-47+96-111",
    3: "48-63+112-127",
}
GPU_PCI = {
    "hsn0": "0000:c1:00.0",
    "cxi3": "0000:03:00.0",
    "cxi2": "0000:41:00.0",
    "cxi1": "0000:82:00.0",
    "cxi0": "0000:c1:00.0",
}


def case_records(
    transport: str,
    device: str,
    suite: str,
    size_mib: int,
    nic_count: int,
    flows_per_nic: int,
    gbps: float = 1.0,
    pool_retained: int | None = None,
    case_name: str | None = None,
) -> tuple[str, str, str, str]:
    case = case_name or (
        f"{transport}-{device}-{suite}-{size_mib}mib-"
        f"{nic_count}n-{flows_per_nic}fpn"
    )
    if transport == "object":
        payload_network = "tcp"
        provider = "ray-object-manager"
        devices = "hsn0"
        extra_path = f"staging={'host' if device == 'gpu' else 'none'}"
    else:
        payload_network = "cxi"
        provider = "libfabric" if transport == "nixl" else "aws-ofi-nccl"
        devices = ",".join(RAILS[transport][:nic_count])
        extra_path = (
            "backend=LIBFABRIC rail_policy="
            + ("striped" if suite == "four-rail" else "pinned")
            if transport == "nixl"
            else "gdr_level=PHB gdr_read=1 netdevs_policy=MAX:1"
        )
    path_record = (
        f"PATH case={case} transport={transport} device={device} "
        f"payload_network={payload_network} provider={provider} "
        f"nic_count={nic_count} devices={devices} {extra_path} ray_control=hsn0"
    )
    striped = transport == "nixl" and suite == "four-rail"
    total_flows = nic_count * flows_per_nic
    bytes_per_flow = size_mib * 1024 * 1024
    total_bytes = bytes_per_flow * total_flows
    median_ms = total_bytes / gbps / 1e6
    result_record = (
        f"RESULT case={case} suite={suite} transport={transport} "
        f"device={device} size_mib={size_mib} nic_count={nic_count} "
        f"flows_per_nic={flows_per_nic} total_flows={total_flows} "
        f"bytes_per_flow={bytes_per_flow} total_bytes={total_bytes} "
        f"warmups=3 iterations=10 median_ms={median_ms:.6f} "
        f"p95_ms={median_ms:.6f} median_aggregate_GBps={gbps:.6f} "
        "validation_status=pass evidence_status=pass"
    )
    targets = devices.split(",")
    target_numas = [TARGET_NUMA[target] for target in targets]
    affinity_record = (
        f"AFFINITY case={case} transport={transport} device={device} "
        f"policy={'distributed' if striped else 'local'} targets={devices} "
        f"target_pci={','.join(TARGET_PCI[target] for target in targets)} "
        f"target_numa={','.join(str(value) for value in target_numas)} "
        f"target_cpus={'|'.join(CPU_SETS[value] for value in target_numas)} "
        + (
            "cpu_numa=all cpu_sets=0-127 "
            if striped
            else (
                f"cpu_numa={','.join(str(value) for value in target_numas)} "
                f"cpu_sets={'|'.join(CPU_SETS[value] for value in target_numas)} "
            )
        )
        + (
            "gpu_numa=none gpu_pci=none "
            if device == "cpu"
            else (
                f"gpu_numa={','.join(str(value) for value in target_numas)} "
                f"gpu_pci={','.join(GPU_PCI[target] for target in targets)} "
            )
        )
        + f"actors={2 * total_flows} status=pass"
    )
    if transport == "nixl":
        if pool_retained is None:
            raise ValueError("synthetic NIXL cases require a retained pool size")
        lifecycle_record = (
            f"ACTOR_POOL case={case} active={2 * total_flows} "
            f"retained={pool_retained} status=pass"
        )
    else:
        lifecycle_record = (
            f"ACTOR_CLEANUP case={case} graceful={2 * total_flows} "
            "forced=0 status=pass"
        )
    return path_record, affinity_record, result_record, lifecycle_record


def write_matrix(directory: Path, bad_operating_point: bool = False) -> list[Path]:
    def header(
        transport: str, profile: str, rail_policy: str = ""
    ) -> list[str]:
        session_affinity = (
            "cpus=128 gpus=4 ray_cpus=32 ray_gpus=1 "
            "cpu_bind=none mem_bind=first-touch gpu_bind=none "
            "cpu_target=hsn0 cuda_visible_devices=3"
            if transport == "object"
            else (
                "cpus=128 gpus=4 ray_cpus=128 "
                f"ray_gpus={0 if transport == 'nixl' else 4} "
                "cpu_bind=none mem_bind=first-touch gpu_bind=none "
                "cpu_target=distributed cuda_visible_devices="
                f"{'none' if transport == 'nixl' else '0,1,2,3'}"
            )
        )
        lines = [
            f"SESSION image=test-image transport={transport} profile={profile} "
            f"{session_affinity}",
            (
                f"STACK transport={transport} ray=2.54.0 torch=2.10.0 "
                "cuda=13.0"
                + (
                    " nixl=1.3.2 cxi_optimized_mrs=false "
                    f"cxi_mr_cache_max_count=1 rail_policy={rail_policy}"
                    + (
                        " max_bw_per_dram_seg=1000"
                        if rail_policy == "striped"
                        else ""
                    )
                    if transport == "nixl"
                    else (
                        " nccl=2.29.2 libfabric=1.22 "
                        "disable_dmabuf_cuda=unset "
                        "disable_cuda_sync_memops=unset"
                        if transport == "nccl"
                        else ""
                    )
                )
            ),
            "NETWORK hostname=nid1 ray_ip=1.1.1.1 interface=hsn0 "
            "alias=nid1-hsn0 pci=0000:c2:00.0 numa=0 "
            "cpus=0-15+64-79 status=pass",
            "NETWORK hostname=nid2 ray_ip=1.1.1.2 interface=hsn0 "
            "alias=nid2-hsn0 pci=0000:c2:00.0 numa=0 "
            "cpus=0-15+64-79 status=pass",
        ]
        if transport == "object":
            lines.extend(
                f"SESSION_AFFINITY hostname={host} target=hsn0 "
                "target_pci=0000:c2:00.0 target_numa=0 "
                "cpus=0-15+64-79 cuda_visible_devices=3 status=pass"
                for host in ("nid1", "nid2")
            )
        else:
            visible = "none" if transport == "nixl" else "0,1,2,3"
            lines.extend(
                f"SESSION_AFFINITY hostname={host} target=distributed "
                "target_pci=none target_numa=all cpus=0-127 "
                f"cuda_visible_devices={visible} status=pass"
                for host in ("nid1", "nid2")
            )
        if transport == "nccl":
            lines.extend(
                [
                    "NET/OFI Selected Provider is cxi",
                    "Channel 00 : NET/AWS Libfabric/0/GDRDMA/",
                    "GPU Direct RDMA Enabled for HCA 0 'cxi3', read 1,",
                    *(f"HCA 0 '{rail}'" for rail in RAILS[transport]),
                ]
            )

        return lines

    paths: list[Path] = []

    lines = header("object", "full")
    for series_transport, device in TRANSPORT_SERIES["object"]:
        for size_mib in SIZES_MIB:
            lines.extend(
                case_records(
                    series_transport,
                    device,
                    "baseline",
                    size_mib,
                    1,
                    1,
                )
            )
        for flows in FLOW_COUNTS:
            lines.extend(
                case_records(
                    series_transport,
                    device,
                    "single-nic",
                    1024,
                    1,
                    flows,
                )
            )
        lines.append(
            f"OPERATING_POINT transport={series_transport} device={device} "
            "flows_per_nic=1 criterion=95pct_of_sweep_max"
        )
    lines.append("SHUTDOWN status=clean transport=object")
    path = directory / "object-full.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths.append(path)

    lines = header("nixl", "single-rail", "pinned")
    lines.append("NIXL_AGENT backend=LIBFABRIC agent=test device=cxi3")
    for size_mib in SIZES_MIB:
        lines.extend(
            case_records(
                "nixl",
                "cpu",
                "baseline",
                size_mib,
                1,
                1,
                pool_retained=2,
            )
        )
    for flows in NIXL_FLOW_COUNTS:
        rate = 0.5 if bad_operating_point and flows == 1 else 1.0
        lines.extend(
            case_records(
                "nixl",
                "cpu",
                "single-nic",
                1024,
                1,
                flows,
                rate,
                pool_retained=2 * flows,
            )
        )
    lines.extend(
        (
            "OPERATING_POINT transport=nixl device=cpu flows_per_nic=1 "
            "criterion=95pct_of_sweep_max",
            "ACTOR_CLEANUP scope=nixl-pool graceful=8 forced=0 retained=8 "
            "status=pass",
            "SHUTDOWN status=clean transport=nixl",
        )
    )
    path = directory / "nixl-single-rail.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths.append(path)

    lines = header("nixl", "four-rail", "striped")
    lines.extend(
        (
            "NIXL_AGENT backend=LIBFABRIC agent=test "
            "device=cxi3,cxi2,cxi1,cxi0",
            "Created 4 rails using provider=cxi",
            "Registered memory on 4 rails",
            "use_striping=true",
        )
    )
    for flows in NIXL_FOUR_RAIL_FLOW_COUNTS:
        lines.extend(
            case_records(
                "nixl",
                "cpu",
                "four-rail",
                1024,
                4,
                flows,
                pool_retained=2 * 4 * flows,
            )
        )
    lines.extend(
        (
            "ACTOR_CLEANUP scope=nixl-pool graceful=16 forced=0 retained=16 "
            "status=pass",
            "SHUTDOWN status=clean transport=nixl",
        )
    )
    path = directory / "nixl-four-rail.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths.append(path)

    lines = header("nccl", "sweep")
    for series_transport, device in TRANSPORT_SERIES["nccl"]:
        for size_mib in SIZES_MIB:
            lines.extend(
                case_records(
                    series_transport,
                    device,
                    "baseline",
                    size_mib,
                    1,
                    1,
                )
            )
        for flows in FLOW_COUNTS:
            lines.extend(
                case_records(
                    series_transport,
                    device,
                    "single-nic",
                    1024,
                    1,
                    flows,
                )
            )
        lines.append(
            f"OPERATING_POINT transport={series_transport} device={device} "
            "flows_per_nic=1 criterion=95pct_of_sweep_max"
        )
    lines.append("SHUTDOWN status=clean transport=nccl")
    path = directory / "nccl-sweep.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths.append(path)

    lines = header("nccl", "scaling")
    for series_transport, device in TRANSPORT_SERIES["nccl"]:
        for nic_count in (1, 2, 4):
            lines.extend(
                case_records(
                    series_transport,
                    device,
                    "multi-nic",
                    1024,
                    nic_count,
                    1,
                )
            )
    lines.append("SHUTDOWN status=clean transport=nccl")
    path = directory / "nccl-scaling.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths.append(path)
    return paths


class StatisticsTests(unittest.TestCase):
    def test_nearest_rank_p95_of_ten_is_maximum(self) -> None:
        self.assertEqual(nearest_rank(range(1, 11), 0.95), 10)

    def test_throughput_is_derived_from_median_duration(self) -> None:
        result = summarize([1.0, 2.0], 3_000_000_000)
        self.assertEqual(result.median_ms, 1500)
        self.assertEqual(result.median_gbps, 2.0)
        self.assertEqual(result.p95_ms, 2000)

    def test_operating_point_is_first_value_within_five_percent(self) -> None:
        rates = {1: 4.0, 2: 9.6, 4: 10.0, 8: 9.8}
        results = {
            flows: Measurements(1.0, 1.0, rate) for flows, rate in rates.items()
        }
        self.assertEqual(choose_operating_point(results), 2)

    def test_operating_point_accepts_an_explicit_shorter_sweep(self) -> None:
        rates = {1: 4.0, 2: 9.6, 4: 10.0}
        results = {
            flows: Measurements(1.0, 1.0, rate) for flows, rate in rates.items()
        }
        self.assertEqual(choose_operating_point(results, (1, 2, 4)), 2)


class LogValidationTests(unittest.TestCase):
    def test_hsn_alias_field_does_not_hide_valid_network_record(self) -> None:
        text = (
            "NETWORK hostname=nid1 ray_ip=1.1.1.1 interface=hsn0 "
            "alias=nid1-hsn0 pci=0000:c2:00.0 numa=0 "
            "cpus=0-15+64-79 status=pass\n"
            "NETWORK hostname=nid2 ray_ip=1.1.1.2 interface=hsn0 "
            "alias=nid2-hsn0 pci=0000:c2:00.0 numa=0 "
            "cpus=0-15+64-79 status=pass\n"
            "SESSION_AFFINITY hostname=nid1 target=hsn0 "
            "target_pci=0000:c2:00.0 target_numa=0 "
            "cpus=0-15+64-79 cuda_visible_devices=3 status=pass\n"
            "SESSION_AFFINITY hostname=nid2 target=hsn0 "
            "target_pci=0000:c2:00.0 target_numa=0 "
            "cpus=0-15+64-79 cuda_visible_devices=3 status=pass\n"
            "SHUTDOWN status=clean transport=object\n"
        )
        validate_transport_log(Path("test.log"), text, "object")


class PublicationTests(unittest.TestCase):
    def test_empty_matrix_is_rejected(self) -> None:
        with self.assertRaises(ResultError):
            load_matrix([])

    def test_csv_replaces_an_existing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.csv"
            output.write_text("stale\n", encoding="utf-8")
            atomic_csv([], output)
            self.assertTrue(output.read_text(encoding="utf-8").startswith("case,"))
            header = output.read_text(encoding="utf-8").splitlines()[0]
            self.assertNotIn("run_id", header)
            self.assertNotIn("source_log", header)
            self.assertFalse(output.with_suffix(".csv.tmp").exists())

    def test_complete_32_result_matrix_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = load_matrix(write_matrix(Path(directory)))
            self.assertEqual(len(rows), 32)
            self.assertEqual({row["image"] for row in rows}, {"test-image"})
            nixl_rows = [row for row in rows if row["transport"] == "nixl"]
            self.assertEqual(len(nixl_rows), 8)
            self.assertEqual(
                {row["cxi_optimized_mrs"] for row in nixl_rows}, {"false"}
            )
            self.assertEqual(
                {row["cxi_mr_cache_max_count"] for row in nixl_rows}, {"1"}
            )
            self.assertEqual(
                {
                    int(row["flows_per_nic"])
                    for row in nixl_rows
                    if row["suite"] == "single-nic"
                },
                {1, 2, 4},
            )
            self.assertEqual(
                {
                    (int(row["nic_count"]), int(row["flows_per_nic"]))
                    for row in nixl_rows
                    if row["suite"] == "four-rail"
                },
                {(4, 1), (4, 2)},
            )
            self.assertTrue(
                all(
                    int(row["total_flows"])
                    == int(row["nic_count"]) * int(row["flows_per_nic"])
                    for row in nixl_rows
                    if row["suite"] == "four-rail"
                )
            )
            self.assertEqual(
                {
                    row["rail_policy"]
                    for row in nixl_rows
                    if row["suite"] == "four-rail"
                },
                {"striped"},
            )
            nccl_rows = [row for row in rows if row["transport"] == "nccl"]
            self.assertEqual(
                {row["disable_dmabuf_cuda"] for row in nccl_rows}, {"unset"}
            )
            self.assertEqual(
                {row["disable_cuda_sync_memops"] for row in nccl_rows},
                {"unset"},
            )

    def test_logs_from_different_run_directories_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = write_matrix(root)
            split_paths = []
            for path in paths:
                destination = root / path.stem / path.name
                destination.parent.mkdir()
                path.replace(destination)
                split_paths.append(destination)
            rows = load_matrix(split_paths)
            self.assertEqual(len(rows), 32)
            self.assertTrue(all("run_id" not in row for row in rows))
            self.assertTrue(all("source_log" not in row for row in rows))

    def test_known_legacy_out_of_scope_cases_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = write_matrix(Path(directory))

            def insert_before(path: Path, marker: str, additions: list[str]) -> None:
                text = path.read_text(encoding="utf-8")
                path.write_text(
                    text.replace(
                        marker,
                        "\n".join(additions) + "\n" + marker,
                        1,
                    ),
                    encoding="utf-8",
                )

            object_records: list[str] = []
            for device, case_name in (
                ("cpu", "object-cpu-single-nic-1024mib-16f"),
                ("gpu", "object-cuda-single-nic-1024mib-16f"),
            ):
                object_records.extend(
                    case_records(
                        "object",
                        device,
                        "single-nic",
                        1024,
                        1,
                        16,
                        case_name=case_name,
                    )
                )
            insert_before(
                paths[0],
                "SHUTDOWN status=clean transport=object",
                object_records,
            )

            nccl_records = list(
                case_records("nccl", "gpu", "single-nic", 1024, 1, 16)
            )
            insert_before(
                paths[3],
                "SHUTDOWN status=clean transport=nccl",
                nccl_records,
            )

            legacy_path, legacy_affinity, _, _ = case_records(
                "nixl",
                "cpu",
                "four-rail",
                1024,
                4,
                4,
                pool_retained=32,
            )
            insert_before(
                paths[2],
                "ACTOR_CLEANUP scope=nixl-pool",
                [legacy_path, legacy_affinity],
            )
            text = paths[2].read_text(encoding="utf-8")
            paths[2].write_text(
                text.replace(
                    "graceful=16 forced=0 retained=16",
                    "graceful=32 forced=0 retained=32",
                ),
                encoding="utf-8",
            )

            rows = load_matrix(paths)
            self.assertEqual(len(rows), 32)
            self.assertNotIn(
                16,
                {int(row["flows_per_nic"]) for row in rows},
            )

    def test_unrecognized_out_of_scope_case_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = write_matrix(Path(directory))
            extra_records = case_records(
                "object",
                "cpu",
                "single-nic",
                1024,
                1,
                32,
            )
            text = paths[0].read_text(encoding="utf-8")
            paths[0].write_text(
                text.replace(
                    "SHUTDOWN status=clean transport=object",
                    "\n".join(extra_records)
                    + "\nSHUTDOWN status=clean transport=object",
                    1,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ResultError, "incomplete flow sweep"):
                load_matrix(paths)


    def test_inconsistent_flow_count_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = write_matrix(Path(directory))
            text = paths[0].read_text(encoding="utf-8")
            text = text.replace(
                "total_flows=1 bytes_per_flow=1048576 total_bytes=1048576",
                "total_flows=2 bytes_per_flow=1048576 total_bytes=2097152",
                1,
            )
            paths[0].write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ResultError, "flow counts"):
                load_matrix(paths)

    def test_nonlocal_gpu_affinity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = write_matrix(Path(directory))
            text = paths[3].read_text(encoding="utf-8")
            paths[3].write_text(
                text.replace(
                    "gpu_numa=3 gpu_pci=0000:03:00.0",
                    "gpu_numa=0 gpu_pci=0000:c1:00.0",
                    1,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ResultError, "NUMA domains differ"):
                load_matrix(paths)

    def test_object_session_binding_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = write_matrix(Path(directory))
            text = paths[0].read_text(encoding="utf-8")
            paths[0].write_text(
                text.replace("cpu_target=hsn0", "cpu_target=distributed", 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ResultError, "Ray visibility"):
                load_matrix(paths)

    def test_nixl_mr_cache_must_be_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = write_matrix(Path(directory))
            text = paths[1].read_text(encoding="utf-8")
            paths[1].write_text(
                text.replace(
                    "cxi_mr_cache_max_count=1",
                    "cxi_mr_cache_max_count=1024",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ResultError, "MR cache"):
                load_matrix(paths)

    def test_nixl_actor_pool_evidence_must_match_the_case(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = write_matrix(Path(directory))
            text = paths[1].read_text(encoding="utf-8")
            paths[1].write_text(
                text.replace("active=2 retained=2", "active=4 retained=2", 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ResultError, "actor-pool evidence"):
                load_matrix(paths)

    def test_nixl_actor_pool_cannot_retain_unexpected_actors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = write_matrix(Path(directory))
            text = paths[1].read_text(encoding="utf-8")
            paths[1].write_text(
                text.replace("active=2 retained=2", "active=2 retained=4", 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ResultError, "actor-pool evidence"):
                load_matrix(paths)

    def test_nixl_actor_pool_cleanup_must_be_graceful(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = write_matrix(Path(directory))
            text = paths[1].read_text(encoding="utf-8")
            paths[1].write_text(
                text.replace(
                    "scope=nixl-pool graceful=8 forced=0 retained=8",
                    "scope=nixl-pool graceful=7 forced=1 retained=8",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ResultError, "pool cleanup"):
                load_matrix(paths)

    def test_nixl_four_rail_evidence_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = write_matrix(Path(directory))
            text = paths[2].read_text(encoding="utf-8")
            paths[2].write_text(
                text.replace("use_striping=true", "use_striping=false"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ResultError, "multi-rail evidence"):
                load_matrix(paths)

    def test_nixl_four_rail_counts_flows_on_every_nic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = write_matrix(Path(directory))
            text = paths[2].read_text(encoding="utf-8")
            paths[2].write_text(
                text.replace(
                    "total_flows=4 bytes_per_flow=1073741824 "
                    "total_bytes=4294967296",
                    "total_flows=1 bytes_per_flow=1073741824 "
                    "total_bytes=1073741824",
                    1,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ResultError, "flow counts"):
                load_matrix(paths)

    def test_logged_operating_point_must_match_flow_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = write_matrix(Path(directory), bad_operating_point=True)
            with self.assertRaisesRegex(ResultError, "flow sweep selects 2"):
                load_matrix(paths)

    def test_render_failure_preserves_all_existing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = [
                root / "results.csv",
                root / "baseline.svg",
                root / "flows.svg",
                root / "nics.svg",
            ]
            for output in outputs:
                output.write_text("original\n", encoding="utf-8")
            args = argparse.Namespace(
                logs=[],
                csv=outputs[0],
                baseline_svg=outputs[1],
                flows_svg=outputs[2],
                nics_svg=outputs[3],
            )
            original_load_matrix = publisher.load_matrix
            original_plot_two_panels = publisher.plot_two_panels
            publisher.load_matrix = lambda unused: []

            def fail_render(*unused: object) -> None:
                raise ResultError("render failed")

            publisher.plot_two_panels = fail_render
            try:
                with self.assertRaisesRegex(ResultError, "render failed"):
                    publisher.publish(args)
            finally:
                publisher.load_matrix = original_load_matrix
                publisher.plot_two_panels = original_plot_two_panels
            self.assertTrue(
                all(
                    output.read_text(encoding="utf-8") == "original\n"
                    for output in outputs
                )
            )
            self.assertEqual(list(root.glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
