# Transport configuration

The launcher applies these settings automatically. This document is a compact
reproducibility checklist and a record of the important constraints.

## Required devices and injection

| Transport | Required container access |
|---|---|
| Object Store | GPU devices when testing GPU tensors; no CXI devices or `--nccl-cu13` |
| NIXL | `--nccl-cu13`, `/dev/cxi0`–`cxi3`, and `/dev/cxi_sbl` |
| NCCL | The NIXL requirements plus `/dev/gdrdrv` |

`--nccl-cu13` supplies the validated Cray libfabric/CXI and aws-ofi-nccl
stack. NIXL uses its libfabric components, not NCCL, for the data path.
`/dev/cxi_sbl` is the CXI service interface rather than a fifth rail.
`/dev/gdrdrv` is required for NCCL's validated PHB/GDRDMA path; mounting
`libgdrapi.so` without the device is insufficient.

Ray's control plane uses TCP in every mode. Object Store payloads use `hsn0`;
NIXL and NCCL payloads use native CXI.

## Locality

The measured local pairs are GPU 0/`cxi3`/NUMA 3, GPU 1/`cxi2`/NUMA 2,
GPU 2/`cxi1`/NUMA 1, and GPU 3/`cxi0`/NUMA 0. Object Store uses `hsn0`
(`cxi0`) and GPU 3 when GPU tensors are tested.

Each Slurm task sees all four GPUs through `--gpu-bind=none`. The launcher then
binds Object Store to the `hsn0` NUMA domain, while each NIXL or NCCL actor
discovers and binds to its assigned CXI device. NCCL additionally verifies that
the assigned GPU and NIC share a NUMA domain. Published rows require matching
affinity evidence.

## Effective transport settings

All NIXL sessions set `FI_PROVIDER=cxi` and `FI_MR_CACHE_MAX_COUNT=1`.
Single-rail mode sets `FI_CXI_DEVICE_NAME=cxi3`; four-rail mode uses one agent
with:

```text
FI_CXI_DEVICE_NAME=cxi3,cxi2,cxi1,cxi0
FI_CXI_OPTIMIZED_MRS=false
NIXL_LIBFABRIC_MAX_BW_PER_DRAM_SEG=1000
```

Every four-rail NIXL flow is striped across all four NICs. Standard memory
regions are required because optimized registration failed on this path. The
bandwidth ceiling allows NIXL to select multiple rails; the one-entry MR cache
and reusable actor pool bound retained CXI registrations.

NCCL uses independent flows pinned to one NIC each:

```text
NCCL_NET=AWS Libfabric
NCCL_NET_GDR_READ=1
NCCL_NETDEVS_POLICY=MAX:1
FI_PROVIDER=cxi
FI_CXI_DEVICE_NAME=<actor NIC>
```

NERSC's injection supplies the remaining PHB, bootstrap, and registration
settings. The launcher removes the obsolete DMA-BUF and CUDA sync-memops
workaround variables before Ray starts.

## Known constraints

- Run each transport session in a fresh allocation. In particular, keep NIXL
  single-rail and four-rail sessions separate; native CXI cleanup can lag Ray
  shutdown.
- NIXL 1.3.2's LIBFABRIC topology does not expose CXI as GPU-capable, so the
  GPU/NIXL path is unavailable in the tested stack.
- NIXL x8 single-rail and x16 total four-rail runs exhausted CXI
  memory-registration resources (`fi_mr_enable: ENOSPC`). This was not host
  OOM or filesystem exhaustion.
- Ray 2.54 teardown can be slow and noisy. The benchmark relies on its
  structured result and shutdown evidence rather than launcher noise.

Use `--smoke` for a quick transport check and `--verbose` for complete Ray and
library output.
