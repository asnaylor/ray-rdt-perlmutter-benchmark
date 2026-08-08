# Perlmutter transport findings

This document records the platform-specific behavior found while building the
Ray transfer benchmarks. It describes the tested software stack rather than a
guarantee about every future Perlmutter image.

## Tested stack

- Ray 2.54.0
- NERSC PyTorch 26.01 image with PyTorch 2.10 development build
- CUDA 13.1 and NCCL 2.29
- NIXL 1.3.2 using `nixl_cu13`
- NIXL-bundled UCX 1.21
- Cray libfabric 1.22 and the `cxi` provider
- AWS OFI NCCL plugin 1.6 for NCCL modes only

## Ray requires a runtime override for NIXL/LIBFABRIC

Ray 2.54's internal `NixlTensorTransport.get_nixl_agent()` constructs its NIXL
agent with `nixl_agent_config(backends=["UCX"])`. Ray exposes no public option
to select NIXL's `LIBFABRIC` backend. Consequently, setting
`FI_PROVIDER=cxi` alone cannot produce a NIXL/LIBFABRIC transfer: libfabric is
never called when Ray creates only a UCX backend.

`ray_nixl_bench.py` installs a process-local replacement for that one Ray
method before the driver or actors initialize NIXL. The replacement creates an
agent without an implicit backend and explicitly creates either `LIBFABRIC` or
`UCX`. For UCX it also supplies `ucx_error_handling_mode=none`, which is needed
for NIXL's local `self/memory` endpoint.

This is deliberately not an unguarded monkeypatch. The benchmark verifies:

- Ray is exactly 2.54.0 and NIXL is exactly 1.3.2.
- The installed method has the expected signature.
- Its source still contains the expected hard-coded UCX configuration.

If any check changes, the run fails instead of silently selecting the wrong
backend. This repository does not modify the installed Ray source or claim the
override is appropriate for another Ray release.

## CXI devices must be passed into the container

The Cassini `/dev/cxi*` character devices exist on a suitable compute host but
are not automatically visible in a Podman-HPC container. Without explicit
`--device=/dev/cxiN` arguments, an NCCL/OFI run can load Cray libfabric and
`libcxi` yet fall back to the libfabric TCP provider.

The launcher resolves and validates the requested host `/dev/cxiN` devices,
passes only those data devices (plus `/dev/cxi_sbl` when present) into the
container, sets `FI_PROVIDER=cxi` and `FI_CXI_DEVICE_NAME`, and fails if a
selected device is unavailable. It runs `fi_info -p cxi` with the same device
filter on each host as an availability preflight. That check is not treated as
transfer proof; the benchmark log must still show the selected `cxi` provider
or NIXL rails created on every requested CXI device.

The launcher currently uses Podman-HPC's `--nccl-cu13` module to inject Cray
libfabric and `libcxi`. NIXL calls libfabric directly and does not use NCCL or
the AWS OFI NCCL plugin merely because the module supplied those libraries.

## Network-device count must be controlled explicitly

The initial NCCL/TCP run exposed six libfabric TCP devices: `hsn0` through
`hsn3`, an `hsn0:chn` endpoint, and `nmn0`. NCCL distributed channels across
those provider devices, so that result was not a single-interface TCP
measurement. The launcher now sets both `FI_TCP_IFACE=hsn0` for libfabric and
`NCCL_SOCKET_IFNAME=hsn0` for NCCL bootstrap. Its evidence check accepts only
`hsn0`/`hsn0:chn` HCA records and fails if another TCP interface appears.

For native CXI, `CXI_RAILS=1` exposes only `cxi3`; the standard benchmark uses
this setting. `CXI_RAILS=4` remains an exploratory option that exposes
`cxi0,cxi1,cxi2,cxi3`. `FI_CXI_DEVICE_NAME` applies the same filter inside
libfabric. The NCCL mode additionally uses
[`NCCL_NETDEVS_POLICY`](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2292/user-guide/docs/env.html#nccl-netdevs-policy)
as `MAX:1` or `ALL`, then requires transfer-channel evidence for every
requested provider device. NCCL 2.29.2 is significant here because its
[release notes](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2307/release-notes/rel_2-29-2.html)
state that send and receive operations obey this policy. The NIXL mode
requires the exact rail count, memory registration on that count, and—on four
rails—a four-request striped transfer. These runtime checks matter because
provider discovery alone does not prove the payload used every visible
device.

The single-device choice is intentionally physical `cxi3`, which NCCL also
selected when all four devices were visible for the benchmark's GPU placement.
This is consistent with Perlmutter's PCIe/NUMA affinity and NERSC's
[documented reverse binding](https://www.nersc.gov/assets/Uploads/NUGcall_GPUaware_Perlmutter.pdf),
where local rank 0 sees physical GPU 3 as logical `cuda:0` through
`CUDA_VISIBLE_DEVICES`. Ray can perform similar logical renumbering,
but these benchmark logs did not record the physical GPU ID, so they prove the
`cxi3` selection rather than that exact GPU mapping.
After `FI_CXI_DEVICE_NAME=cxi3` filters the provider to one device, NCCL reports
that device as HCA 0 and transfer channels as `NET/AWS Libfabric/0/Shared`.
The `/0/` is the provider-local index of `cxi3`; it is not evidence that the
physical `cxi0` device was used.

In the completed one-versus-four experiment, NCCL discovered all four provider
devices but assigned both transfer channels to provider device 3 (`cxi3`). It
therefore did not produce a four-rail transfer. NIXL/LIBFABRIC did submit four
striped requests, but four rails were slower than one: 2.968 versus 3.684 GB/s
at 64 MiB and 4.880 versus 6.009 GB/s at 1 GiB. The result is consistent with
extra request and synchronization overhead dominating before one rail became a
bottleneck; no hardware link counters were collected. The standard matrix
consequently publishes only `CXI_RAILS=1`. Separate `-1cxi` and `-4cxi` log
names preserve exploratory runs without conflating them with standard results.

## Bootstrap traffic does not identify the payload path

Messages such as `Bootstrap: Using hsn0` describe connection setup. They do
not prove that the tensor payload used TCP. For NCCL modes, the meaningful
line is `NET/OFI Selected Provider is tcp` or `cxi`. For NIXL modes, the
meaningful evidence is the selected backend plus its UCX protocol or
LIBFABRIC rail output.

## UCX used TCP in the tested stack

NIXL's UCX backend is not inherently TCP, and this finding should not be read
as a universal Perlmutter default. In the tested wheel/container combination,
UCX reported no supported native IB/CXI device. With TCP available it selected
TCP network interfaces. The working benchmark removes ambiguity with:

```text
UCX_TLS=self,tcp
UCX_NET_DEVICES=hsn0
```

`self/memory` is the same-process lane NIXL needs while initializing its UCX
agent. The inter-node protocol table must separately report `tcp/hsn0`. The
successful CPU benchmark showed `from host memory to host`, empty
`CUDA_VISIBLE_DEVICES`, and actors on different nodes.

The compute nodes expose multiple HSN addresses. DNS and `ip route get`
diagnostics showed bidirectional routes and selected `hsn0` for the tested
peers, so the benchmark pins TCP to that interface. The hostnames are useful
for diagnostics; `UCX_NET_DEVICES` takes interface names, not DNS names.

TCP on `hsn0` still traverses Perlmutter's Slingshot network, but through the
kernel TCP/IP stack. It is not the native `cxi` libfabric/RDMA path.

## The image and NIXL use different UCX builds

The PyTorch image includes `ucx_info` and a general-purpose UCX installation
under `/usr/local/ucx`. It is present even when the container is started
without `--gpu` or `--nccl-cu13`, so it comes from the image rather than the
Perlmutter host or either Podman-HPC option. With GPU runtime access enabled,
this UCX reports the `cuda_copy` and `cuda_ipc` transports.

Those transports have limited meanings: `cuda_copy` allows UCX to stage
between CUDA and host memory, while `cuda_ipc` is an intra-node CUDA transport.
Their presence does not prove an inter-node GPU transfer or GPUDirect RDMA. A
CUDA transfer over `tcp/hsn0` would still use GPU-to-host staging, kernel TCP,
and host-to-GPU staging.

NIXL does not use that `/usr/local/ucx` installation. The `nixl_cu13` wheel
loads its own isolated UCX libraries from `nixl_cu13.libs`. A preflight with
GPU runtime access detected the GPU but reported that `cuda_copy` and
`cuda_ipc` were unavailable and that GPU memory was unsupported. Therefore the
tested NIXL/UCX stack supports the CPU/TCP benchmark, not a CUDA/TCP benchmark.

The `/usr/local/ucx/bin/ucx_info` executable also failed with symbol and ABI
errors when forced to load NIXL's UCX libraries. The two builds cannot safely
be swapped at runtime. Supporting NIXL/UCX CUDA tensors would require rebuilding
NIXL against the image's CUDA-aware UCX, or rebuilding NIXL's matching UCX with
CUDA support enabled.

## Two UCX implementations caused false routing failures

The PyTorch base image loaded HPC-X UCX from `/opt/hpcx/ucx`, while the
`nixl_cu13` wheel loaded a separately bundled UCX 1.21. A single Python process
therefore contained two complete UCX stacks. This correlated with false
`no route` errors during remote NIXL metadata loading and UCX teardown crashes.

`run_with_nixl_ucx.sh` creates conventional UCX SONAME aliases for the wheel's
hashed libraries and puts both the alias directory and `nixl_cu13.libs` first
in `LD_LIBRARY_PATH`. PyTorch and NIXL then resolve to the same UCX
implementation. The supported UCX preflight inspects `/proc/self/maps` and
fails if `/opt/hpcx/ucx` is mapped or more than one UCX library root remains.

The resulting process reported one NIXL UCX library root and established an
inter-node `tcp/hsn0` connection successfully.

## Known limitations

### NCCL GPUDirect RDMA

With `NCCL_NET_GDR_LEVEL=PHB`, NCCL initialized a CXI GDRDMA path but the first
Ray RDT transfer timed out in the tested environment. `LOC` completed by using
CXI with host staging and remains the launcher default. No GPUDirect claim is
made for that result.

### NIXL/LIBFABRIC dependencies and shutdown

The CUDA-13 NIXL LIBFABRIC plugin links `libcuda.so.1`. Its CPU tensor mode
therefore needs the GPU-node runtime mounted even though Ray advertises zero
GPUs and the actors allocate only CPU tensors.

Some valid LIBFABRIC/CXI runs reached `Destroying rail manager` and then hung
during teardown. The launcher accepts that timeout only if full payload
verification and every CXI evidence check already passed. Any timeout before a
valid result remains a failure.

### NIXL/LIBFABRIC CXI does not support CUDA memory in this stack

NIXL/LIBFABRIC with CUDA tensors is not exposed as a benchmark mode. The GPU
runtime and CXI layers themselves were available in the failed experiment:
Ray assigned a CUDA device to each actor, Cray libfabric found four `cxi` rails,
and those rails reported `FI_HMEM` CUDA support. The failure occurred inside
NIXL's compiled LIBFABRIC plugin before a transfer was posted.

NIXL 1.3.2 performs accelerator discovery only for the `efa` provider. Every
other provider, including `cxi`, enters a simplified topology branch intended
for TCP/sockets and explicitly records zero NVIDIA GPUs. The rail manager then
selects `SYSTEM (no accelerators)`, the backend omits `VRAM_SEG` from its
supported memory types, and CUDA tensor registration fails with
`NIXL_ERR_NOT_FOUND`. The relevant implementation is in NIXL's
[`libfabric_topology.cpp`](https://github.com/ai-dynamo/nixl/blob/v1.3.2/src/utils/libfabric/libfabric_topology.cpp)
and
[`libfabric_backend.cpp`](https://github.com/ai-dynamo/nixl/blob/v1.3.2/src/plugins/libfabric/libfabric_backend.cpp).

The CPU benchmark works because `DRAM_SEG` is always supported, including when
NIXL classifies the runtime as `SYSTEM`. Its host buffers register on the CXI
rails and transfer through native libfabric/CXI normally. Only `VRAM_SEG` is
conditional on accelerator discovery.

The observed status 137 was cleanup fallout, not evidence of an out-of-memory
kill: the head task first failed on unsupported `VRAM_SEG`, after which Slurm
killed the remaining task while unwinding the failed step.

Ray's hard-coded backend selection is Python code and can be overridden safely
with version guards. This CXI accelerator classification is C++ inside the NIXL
plugin, so the same Python monkeypatch cannot fix it. Supporting the mode would
require an upstream NIXL fix or a rebuilt/patched NIXL plugin; this benchmark
repository does neither.

## Evidence policy

Configuration and preflights show that a path is available; they do not prove
that the payload used it. A published result requires all three records:

```text
RESULT ... transfer_status=pass
EVIDENCE ... status=pass
SHUTDOWN_STATUS=clean
```

The object-store mode is the exception at the provider layer: it proves the
Ray object-store CUDA-to-CUDA path but does not claim TCP or CXI because its
current runtime output does not identify that lower-level provider.
