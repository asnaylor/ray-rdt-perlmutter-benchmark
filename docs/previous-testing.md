# Previous Perlmutter transport testing

This document preserves the findings that directly informed the current
benchmark. Historical modes and their implementation have been removed from
the primary workflow.

## Tested software

The main experiments used the NERSC PyTorch 26.01 image with Ray 2.54, CUDA 13,
NCCL 2.29, NIXL 1.3.2, and the NERSC `--nccl-cu13` injection. That injection
provided aws-ofi-nccl 1.6, Cray libfabric 1.22, and the CXI provider.

A comparison image isolated aws-ofi-nccl 1.19 and libfabric 2.1 while retaining
NCCL 2.29 from the same NERSC PyTorch image.

## Measured Perlmutter locality

Sysfs and `nvidia-smi topo -m` on both nodes showed that `hsn0` is the `cxi0`
PCI function in NUMA domain 0. The complete local pairs are `cxi3`/GPU 0 on
NUMA 3, `cxi2`/GPU 1 on NUMA 2, `cxi1`/GPU 2 on NUMA 1, and `cxi0`/GPU 3 on
NUMA 0. The associated CPU sets are respectively `48-63,112-127`,
`32-47,96-111`, `16-31,80-95`, and `0-15,64-79`.

The current launcher uses one task per node with 128 logical CPUs and
`--gpus-per-task=4 --gpu-bind=none`. For Object Store,
the container entrypoint binds the Ray process tree to NUMA 0 and selects
physical GPU 3 with `CUDA_VISIBLE_DEVICES`; it advertises 32 CPUs and one GPU to
Ray. NIXL actors instead discover and bind themselves to
the selected CXI device's CPUs. NCCL actors use the reversed CXI order
`cxi3,cxi2,cxi1,cxi0`, which aligns Ray's normal GPU 0/1/2/3 assignment with
the physical topology, and fail before measurement if the actual CUDA PCI and
CXI NUMA domains differ. Per-case `AFFINITY` evidence makes this a validated
condition rather than an assumption based on logical Ray resources.

## The PHB failure was a missing device

The host exposed `/dev/gdrdrv`, but Podman-HPC's `--gpu` option did not pass it
into the container. It passed NVIDIA device nodes and `--nccl-cu13` copied
`libgdrapi.so`, but the GDRCopy userspace library could not work without its
kernel character device.

With DMA-BUF disabled and `/dev/gdrdrv` absent, libfabric could not open
GDRCopy. The resulting behavior was initially interpreted as a communication-
stack incompatibility. Explicitly passing `/dev/gdrdrv` allowed both the old
and newer stacks to complete full-PHB GDRDMA.

The successful injected-stack configuration used:

```text
NCCL_NET=AWS Libfabric
NCCL_NET_GDR_LEVEL=PHB
NCCL_NET_GDR_READ=1
FI_PROVIDER=cxi
FI_CXI_DISABLE_DMABUF_CUDA=1
FI_CXI_DISABLE_CUDA_SYNC_MEMOPS=1
```

Follow-up qualification tested both workarounds, each
workaround independently, and neither workaround. Every configuration passed
three 1 GiB repetitions at approximately 21.5--21.8 GB/s. The current
benchmark therefore explicitly removes both workaround variables before Ray
starts and records their effective values in the NCCL stack metadata.

`/dev/cxiN` supplies access to a Cassini NIC. `/dev/cxi_sbl` is the CXI
service/control interface and is not another network rail. `/dev/gdrdrv`
supplies the GPU-memory mapping path used by GDRCopy.

## LOC versus PHB and stack comparison

The controlled comparison used the same allocation, one GPU actor and `cxi3`
per node, complete payload validation, five warmups, and 20 measurements:

| Payload | LOC, aws-ofi 1.6/libfabric 1.22 | PHB, aws-ofi 1.6/libfabric 1.22 | PHB, aws-ofi 1.19/libfabric 2.1 |
|---|---:|---:|---:|
| 64 MiB | 7.202 ms / 9.318 GB/s | 6.810 ms / 9.854 GB/s | Not measured |
| 1 GiB | 62.370 ms / 17.216 GB/s | 49.476 ms / 21.702 GB/s | 49.400 ms / 21.736 GB/s |

At 1 GiB, PHB reduced median latency by 20.7% and increased median throughput
by 26.1% relative to LOC. The two PHB stacks differed by approximately 0.15%,
which was not a meaningful steady-state advantage. The simpler NERSC-injected
stack is therefore the primary path.

The newer-stack investigation followed the colleague's
[Perlmutter NCCL report](https://github.com/dingp/communication-libraries-image/blob/main/benchmarks/reports/nccl-podman-hpc-8node-aws-ofi-1.19.0-libfabric-2.1.0-20260508.md)
and accompanying
[diagnostic gist](https://gist.github.com/dingp/6934e7bd843d68831a9f029787c1ed94).

## NIXL scope

Ray 2.54 hard-codes UCX when it constructs its NIXL agent. The benchmark uses a
strict, version-checked process-local override to create the NIXL 1.3.2
LIBFABRIC backend instead.

CPU tensors work through LIBFABRIC/CXI. GPU tensors are not published because
NIXL 1.3.2's LIBFABRIC topology classifies CXI through a simplified path that
does not expose accelerator support. This is a known stack limitation, not a
reason to relabel a staged or fallback path as GPU/CXI.

UCX-over-TCP was useful while separating backend problems, but it duplicates
the TCP comparison now provided by the Object Store and distracts from the
native-CXI question. It has therefore been removed from the current matrix.

## Why the concurrency design changed

Earlier one-versus-four-NIC experiments exposed several devices to a single
transfer. NCCL still selected one device, while NIXL's explicit striping was a
different transfer model. That comparison did not answer how many independent
Ray flows are needed to fill one NIC.

The current methodology holds the interface count at one while sweeping
concurrent flows. Object Store and NCCL test 1, 2, 4, and 8; NIXL tests 1, 2,
and 4. NIXL then separately measures one and two flows per NIC (four and eight
total logical flows), with each flow natively striped across
`cxi3,cxi2,cxi1,cxi0`. NCCL instead holds its selected flows-per-NIC operating
point constant and scales to 1, 2, and 4 NICs.
The result metadata distinguishes NIXL's `striped` policy from pinned
single-interface and NCCL flows, so these different transfer models are not
silently conflated. Single-NIC NIXL uses `cxi3`, matching the original
benchmark.

An early full-matrix attempt completed NIXL baselines
and the 1/2/4-flow single-NIC cases. At eight concurrent 1 GiB flows, CXI
reported an optimized-memory-region PTE link failure followed by invalid buffer
events, and one RDT object timed out after 60 seconds. There was no OOM evidence.
A follow-up on a different node pair forced standard CXI memory regions with
`FI_CXI_OPTIMIZED_MRS=false`; libfabric confirmed `optimized_mrs=0`. The same
eight-flow case failed again, this time immediately: `fi_mr_enable` returned
`-28` (`No space left on device`) while registering a receive buffer. This is
a CXI registration-resource error, not filesystem or host-memory exhaustion.
The reproducible limit is why the published single-interface NIXL design stops
at four flows and scales aggregate concurrency using all four NICs instead.

Two immediate retries reused the same Slurm allocation after the eight-flow
failure. The first timed out at four flows after an optimized MR fell back to a
standard MR. The next forced standard MRs but failed to register even the first
1 MiB payload with `No space left on device`. This is consistent with depleted
CXI registration resources persisting for that allocation, so those retries
are not valid performance measurements.

An initial mitigation disabled the libfabric MR cache for NIXL with
`FI_MR_CACHE_MAX_COUNT=0`. On a fresh allocation, the one-flow smoke transfer
completed at the NIXL/libfabric layer but the receive call never returned to
Ray. Both actors reported `MR caching disabled`, and the run produced no result
before its timeout. The earlier `Unexpected event PUT` warning was also present
in successful cached runs, so it does not by itself explain the stall.

The benchmark now uses `FI_MR_CACHE_MAX_COUNT=1`. The image's libfabric reports
a default cache limit of 1024 regions. One entry bounds retained CXI
registrations per actor without entering the zero-cache path.

A fresh-node full run showed why that
setting was not sufficient by itself. The three baselines and the one- and
two-flow single-NIC cases passed, but a receiver in the four-flow case failed
to enable its first 1 GiB MR with `No space left on device`. Each earlier case
had created new actors. Ray's termination tasks completed before several
native MR caches emitted their cleanup records, so one cache per old actor
could still overlap the next case.

The driver now lazily creates a fixed actor pool, reuses every rail/flow slot
across cases, and terminates the pool only after the NIXL sweep. With the
one-entry cache, retained registrations are thereby bounded by the largest
requested concurrency rather than the cumulative number of cases. Logs emit
an `ACTOR_POOL` record after every result and one final `ACTOR_CLEANUP` record.

The pooled retry passed every baseline and
the 1/2/4-flow single-NIC cases; four flows reached 10.895 GB/s aggregate. Its
first eight-flow batch completed seven transfers, then stopped after an
optimized-MR PTE-link failure and fallback. This isolated eight simultaneous
registrations as the unstable boundary rather than per-case actor accumulation.
The published single-interface NIXL sweep therefore stops at four flows.

The former four-NIC implementation created a separate NIXL agent per rail and
continued to stall during registration. A minimal native multi-rail probe then
created one NIXL agent per node with all four devices. With optimized MRs it
stalled when the first PTE-link failure entered the provider's standard-MR
fallback. A subsequent probe started directly with
`FI_CXI_OPTIMIZED_MRS=0`: sender and receiver each registered the 1 GiB buffer
on four rails, NIXL reported `use_striping=true`, all four 256 MiB rail requests
completed, and full payload verification passed. The current four-rail suite
uses that native design, forces standard MRs, and runs separately from the
single-rail suite. Four flows per NIC (16 total) later hit the same CXI
registration limit, so the published sweep retains one and two flows per NIC.
