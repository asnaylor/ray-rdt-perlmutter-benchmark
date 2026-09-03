#!/usr/bin/env python3
"""Benchmark GPU tensors through Ray RDT and NCCL/LIBFABRIC/CXI."""

from __future__ import annotations

import argparse
import ctypes
import gc
import os
import stat
from typing import Any

import ray
import torch

from benchmark_common import (
    BASELINE_SIZES_MIB,
    CXI_DEVICES,
    DEFAULT_ITERATIONS,
    DEFAULT_WARMUPS,
    FLOW_COUNTS,
    MIB,
    actor_affinity,
    cleanup_actors,
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
from benchmark_stats import choose_operating_point


RDT_API_ERROR: Exception | None = None
try:
    from ray.experimental.collective import create_collective_group

    @ray.remote(enable_tensor_transport=True)
    class Sender:
        def __init__(self, nbytes: int, rail: str) -> None:
            os.environ["FI_PROVIDER"] = "cxi"
            os.environ["FI_CXI_DEVICE_NAME"] = rail
            self.rail = rail
            self.affinity = actor_affinity("cxi", rail, "cuda")
            self.payload = make_payload(nbytes, torch.device("cuda"))

        def info(self) -> dict[str, Any]:
            return {
                **identity(),
                **self.affinity,
                "device": "cuda",
                "rail": self.rail,
            }

        @ray.method(tensor_transport="nccl")
        def send(self) -> torch.Tensor:
            return self.payload


    @ray.remote(enable_tensor_transport=True)
    class Receiver:
        def __init__(self, rail: str) -> None:
            os.environ["FI_PROVIDER"] = "cxi"
            os.environ["FI_CXI_DEVICE_NAME"] = rail
            self.rail = rail
            self.affinity = actor_affinity("cxi", rail, "cuda")
            self.payload: torch.Tensor | None = None

        def info(self) -> dict[str, Any]:
            return {
                **identity(),
                **self.affinity,
                "device": "cuda",
                "rail": self.rail,
            }

        def receive(self, payload: torch.Tensor) -> dict[str, Any]:
            if payload.device.type != "cuda":
                raise RuntimeError(f"NCCL receiver got {payload.device}")
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

except Exception as exc:
    RDT_API_ERROR = exc
    Sender = None  # type: ignore[assignment,misc]
    Receiver = None  # type: ignore[assignment,misc]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=("sweep", "scaling", "smoke"), required=True
    )
    parser.add_argument("--flows-per-nic", type=int, choices=FLOW_COUNTS)
    args = parser.parse_args()
    if args.profile == "scaling" and args.flows_per_nic is None:
        parser.error("--profile scaling requires --flows-per-nic")
    if args.profile != "scaling" and args.flows_per_nic is not None:
        parser.error("--flows-per-nic is valid only with --profile scaling")
    return args


def require_environment() -> None:
    if RDT_API_ERROR is not None:
        raise RuntimeError(f"Ray NCCL RDT API is unavailable: {RDT_API_ERROR}")
    if not torch.cuda.is_available():
        raise RuntimeError("Torch cannot access CUDA in this container")
    try:
        mode = os.stat("/dev/gdrdrv").st_mode
    except OSError as exc:
        raise RuntimeError("/dev/gdrdrv is unavailable inside the container") from exc
    if not stat.S_ISCHR(mode):
        raise RuntimeError("/dev/gdrdrv is not a character device")
    required = {
        "NCCL_NET": "AWS Libfabric",
        "NCCL_NET_GDR_LEVEL": "PHB",
        "NCCL_NET_GDR_READ": "1",
        "NCCL_NETDEVS_POLICY": "MAX:1",
        "FI_PROVIDER": "cxi",
    }
    for name, expected in required.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(
                f"{name}={os.environ.get(name)!r}; expected {expected!r}"
            )


def environment_setting(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return "unset"
    if not value:
        return "empty"
    return value


def nccl_version() -> str:
    value = torch.cuda.nccl.version()
    if isinstance(value, tuple):
        return ".".join(str(part) for part in value)
    return str(value)


def libfabric_version() -> str:
    library = ctypes.CDLL("libfabric.so.1")
    library.fi_version.restype = ctypes.c_uint32
    encoded = int(library.fi_version())
    return f"{encoded >> 16}.{encoded & 0xffff}"


def rails_for_case(nic_count: int, flows_per_nic: int) -> list[str]:
    return [
        rail
        for rail in CXI_DEVICES[:nic_count]
        for _ in range(flows_per_nic)
    ]


def create_flows(
    head: dict[str, Any],
    worker: dict[str, Any],
    nbytes: int,
    nic_count: int,
    flows_per_nic: int,
) -> tuple[
    list[Any], list[Any], list[Any], list[Any], list[dict[str, Any]]
]:
    assert Sender is not None and Receiver is not None
    head_group = create_gpu_group(head, nic_count, flows_per_nic)
    worker_group = create_gpu_group(worker, nic_count, flows_per_nic)
    senders: list[Any] = []
    receivers: list[Any] = []
    for bundle_index, rail in enumerate(CXI_DEVICES[:nic_count]):
        for _ in range(flows_per_nic):
            senders.append(
                Sender.options(
                    **gpu_actor_options(head_group, bundle_index, flows_per_nic)
                ).remote(nbytes, rail)
            )
            receivers.append(
                Receiver.options(
                    **gpu_actor_options(worker_group, bundle_index, flows_per_nic)
                ).remote(rail)
            )
    infos = ray.get([actor.info.remote() for actor in [*senders, *receivers]])
    expected_rails = rails_for_case(nic_count, flows_per_nic) * 2
    for info, rail in zip(infos, expected_rails, strict=True):
        if info["rail"] != rail or not info["accelerator_ids"]:
            raise RuntimeError("NCCL GPU/rail actor placement is inconsistent")
    for node_infos in (infos[: len(senders)], infos[len(senders) :]):
        rail_to_gpu: dict[str, set[str]] = {}
        for info in node_infos:
            rail_to_gpu.setdefault(info["rail"], set()).update(
                str(item) for item in info["accelerator_ids"]
            )
        if any(len(gpus) != 1 for gpus in rail_to_gpu.values()):
            raise RuntimeError("flows for one NCCL rail do not share one GPU")
        selected_gpus = {next(iter(gpus)) for gpus in rail_to_gpu.values()}
        if len(selected_gpus) != nic_count:
            raise RuntimeError("different NCCL rails were not assigned distinct GPUs")
    collective_groups = []
    for sender, receiver in zip(senders, receivers, strict=True):
        collective_groups.append(
            create_collective_group([sender, receiver], backend="nccl")
        )
    return (
        senders,
        receivers,
        [head_group, worker_group],
        collective_groups,
        infos,
    )


def run_case(
    head: dict[str, Any],
    worker: dict[str, Any],
    suite: str,
    size_mib: int,
    nic_count: int,
    flows_per_nic: int,
    warmups: int,
    iterations: int,
) -> Any:
    total_flows = nic_count * flows_per_nic
    case_id = (
        f"nccl-gpu-{suite}-{size_mib}mib-{nic_count}n-"
        f"{flows_per_nic}fpn"
    )
    print(
        f"RUN case={case_id} suite={suite} transport=nccl device=gpu "
        f"size_mib={size_mib} nic_count={nic_count} "
        f"flows_per_nic={flows_per_nic} total_flows={total_flows}",
        flush=True,
    )
    emit_path(
        case_id,
        "nccl",
        "gpu",
        "cxi",
        "aws-ofi-nccl",
        nic_count,
        CXI_DEVICES[:nic_count],
        gdr_level="PHB",
        gdr_read=1,
        netdevs_policy="MAX:1",
        ray_control="hsn0",
    )
    senders: list[Any] = []
    receivers: list[Any] = []
    groups: list[Any] = []
    collective_groups: list[Any] = []
    try:
        (
            senders,
            receivers,
            groups,
            collective_groups,
            affinity_infos,
        ) = create_flows(
            head, worker, size_mib * MIB, nic_count, flows_per_nic
        )
        validate_and_emit_affinity(
            case_id,
            "nccl",
            "gpu",
            affinity_infos,
            rails_for_case(nic_count, flows_per_nic),
        )
        measured = run_batches(
            senders,
            receivers,
            size_mib * MIB,
            "cuda",
            warmups,
            iterations,
        )
        emit_result(
            case_id,
            suite,
            "nccl",
            "gpu",
            size_mib,
            nic_count,
            flows_per_nic,
            total_flows,
            warmups,
            iterations,
            measured,
        )
        return measured
    finally:
        collective_groups.clear()
        cleanup_actors([*senders, *receivers], groups)


def benchmark(profile: str, scaling_flows_per_nic: int | None = None) -> None:
    require_environment()
    print(
        f"STACK transport=nccl ray={ray.__version__} torch={torch.__version__} "
        f"cuda={torch.version.cuda} nccl={nccl_version()} "
        f"libfabric={libfabric_version()} "
        "disable_dmabuf_cuda="
        f"{environment_setting('FI_CXI_DISABLE_DMABUF_CUDA')} "
        "disable_cuda_sync_memops="
        f"{environment_setting('FI_CXI_DISABLE_CUDA_SYNC_MEMOPS')}",
        flush=True,
    )
    ray.init(address="auto")
    try:
        head, worker = select_nodes()
        validate_hsn0(head, worker)
        if profile == "smoke":
            run_case(head, worker, "smoke", 1, 1, 1, 1, 1)
            return

        if profile == "sweep":
            for size_mib in BASELINE_SIZES_MIB:
                run_case(
                    head,
                    worker,
                    "baseline",
                    size_mib,
                    1,
                    1,
                    DEFAULT_WARMUPS,
                    DEFAULT_ITERATIONS,
                )
            flow_results = {
                flows: run_case(
                    head,
                    worker,
                    "single-nic",
                    1024,
                    1,
                    flows,
                    DEFAULT_WARMUPS,
                    DEFAULT_ITERATIONS,
                )
                for flows in FLOW_COUNTS
            }
            operating_point = choose_operating_point(flow_results)
            print(
                f"OPERATING_POINT transport=nccl device=gpu "
                f"flows_per_nic={operating_point} criterion=95pct_of_sweep_max",
                flush=True,
            )
            return

        if profile != "scaling" or scaling_flows_per_nic not in FLOW_COUNTS:
            raise RuntimeError("invalid NCCL benchmark profile or flow count")
        for nic_count in (1, 2, 4):
            run_case(
                head,
                worker,
                "multi-nic",
                1024,
                nic_count,
                scaling_flows_per_nic,
                DEFAULT_WARMUPS,
                DEFAULT_ITERATIONS,
            )
    finally:
        ray.shutdown()
        print("SHUTDOWN status=clean transport=nccl", flush=True)


if __name__ == "__main__":
    try:
        arguments = parse_args()
        benchmark(arguments.profile, arguments.flows_per_nic)
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
