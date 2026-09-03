# Previous Perlmutter transport testing

This document preserves the earlier findings that determined the current
benchmark design. It intentionally omits the chronological debugging log.

## Tested software

The main experiments used the NERSC PyTorch 26.01 image with Ray 2.54,
CUDA 13, NCCL 2.29, NIXL 1.3.2, aws-ofi-nccl 1.6, and Cray libfabric 1.22.
A controlled comparison substituted aws-ofi-nccl 1.19 and libfabric 2.1 while
retaining the same PyTorch and NCCL versions.

## NCCL PHB/GDRDMA

The original PHB failure was caused by a missing container device, not an
incompatible communication stack. Podman-HPC exposed the NVIDIA devices and
injected `libgdrapi.so`, but did not automatically expose `/dev/gdrdrv`.
Passing that device explicitly enabled GDRCopy and full-PHB GDRDMA.

Qualification tested the earlier DMA-BUF and CUDA sync-memops workarounds
together, separately, and disabled. All four configurations passed three
repeated 1 GiB transfers at approximately 21.5–21.8 GB/s, so the current
launcher removes both variables and records their effective values.

The controlled locality and stack comparison was:

| Payload | LOC, aws-ofi 1.6/libfabric 1.22 | PHB, aws-ofi 1.6/libfabric 1.22 | PHB, aws-ofi 1.19/libfabric 2.1 |
|---|---:|---:|---:|
| 64 MiB | 7.202 ms / 9.318 GB/s | 6.810 ms / 9.854 GB/s | Not measured |
| 1 GiB | 62.370 ms / 17.216 GB/s | 49.476 ms / 21.702 GB/s | 49.400 ms / 21.736 GB/s |

At 1 GiB, PHB reduced median latency by 20.7% and increased throughput by
26.1% relative to LOC. The two PHB stacks differed by about 0.15%, so the
simpler NERSC-injected stack became the primary path.

The newer-stack comparison followed the colleague's
[Perlmutter NCCL report](https://github.com/dingp/communication-libraries-image/blob/main/benchmarks/reports/nccl-podman-hpc-8node-aws-ofi-1.19.0-libfabric-2.1.0-20260508.md)
and accompanying
[diagnostic gist](https://gist.github.com/dingp/6934e7bd843d68831a9f029787c1ed94).

## NIXL scope

Ray 2.54 constructs its NIXL agent with UCX. The benchmark therefore uses a
strict, version-checked override to select NIXL 1.3.2's LIBFABRIC backend.
CPU tensors work over LIBFABRIC/CXI, but that version's topology does not
expose CXI as GPU-capable. GPU/NIXL results are consequently unavailable rather
than relabeled as a staged or fallback path.

UCX-over-TCP was useful during diagnosis but duplicated the Object Store TCP
comparison, so it was removed from the current matrix.

## Concurrency and multi-rail decisions

Earlier tests exposed several NICs to a single transfer. NCCL still selected
one NIC, while NIXL striped a payload across multiple NICs; those were different
transfer models. The current benchmark first sweeps concurrent flows on one
interface, then separately evaluates NIC scaling. NIXL uses 4 or 8 total flows,
each striped across four NICs. NCCL uses independent flows pinned one per NIC.

NIXL registration resources set the upper concurrency limit:

- Eight simultaneous flows on one rail failed with optimized-MR PTE-link
  errors and, with standard MRs, `fi_mr_enable: ENOSPC`.
- Sixteen total flows across four striped NICs reached the same registration
  limit.
- Reusing an allocation after a failure caused later registrations to fail,
  showing that native CXI cleanup can lag Ray shutdown.
- Disabling the libfabric MR cache entirely allowed a transfer to complete at
  the provider layer but prevented the receive call from returning to Ray.

The working design uses `FI_MR_CACHE_MAX_COUNT=1`, reuses a bounded actor pool,
and runs single-rail and four-rail tests in separate allocations. Native
four-rail mode uses one NIXL agent with all four CXI devices and standard MRs
(`FI_CXI_OPTIMIZED_MRS=false`). These changes stabilized the published x1/x2/x4
single-rail cases and the 4/8-total-flow four-rail cases; higher concurrency is
excluded rather than reported as a performance result.

Current device, affinity, and environment requirements are summarized in
[transport configuration](transport-configuration.md).
