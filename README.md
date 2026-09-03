# Ray tensor transport on Perlmutter

This repository measures one-way tensor transfers between two Ray actors on
different Perlmutter GPU nodes. It compares Ray's TCP Object Store path with
Ray Direct Transport (RDT) over NCCL and NIXL.

This is a point-to-point transport benchmark. It is not an all-reduce, DDP,
training-throughput, or MPI benchmark.

## Supported matrix

| Tensor path | Ray Object Store | NCCL RDT | NIXL RDT |
|---|---|---|---|
| CPU to CPU | HSN/TCP | Unsupported | LIBFABRIC/CXI |
| GPU to GPU | HSN/TCP with host staging | LIBFABRIC/CXI with PHB GDRDMA | Currently unavailable |

The NIXL 1.3.2 LIBFABRIC topology implementation does not currently expose CXI
as GPU-capable, so the GPU/NIXL cell is documented as unsupported rather than
silently falling back to another transport. NCCL does not provide a CPU tensor
transport.

Ray's control plane uses TCP in every mode. The network labels in the table
refer to the tensor payload: Object Store payloads use `hsn0`, while NCCL and
NIXL payloads use native CXI.

## Results

Each flow transferred a 1 GiB tensor. These are the best measured aggregate
rates within the in-scope single-interface sweeps:

| Tensor | Transport | Best flow count | Median aggregate GB/s |
|---|---|---:|---:|
| CPU | Ray Object Store over HSN/TCP | 8 | 1.133 |
| CPU | NIXL over LIBFABRIC/CXI | 4 | 8.939 |
| GPU | Ray Object Store with host staging | 8 | 0.888 |
| GPU | NCCL over aws-ofi-nccl/CXI | 1 | 21.700 |

The best multi-rail configurations were:

| Tensor | Transport | Configuration | Total flows | Median aggregate GB/s |
|---|---|---|---:|---:|
| CPU | NIXL, four striped CXI rails | 2 flows per NIC | 8 | 17.272 |
| GPU | NCCL, four CXI NICs | 1 flow per NIC | 4 | 79.764 |

The main outcomes were:

- NCCL performed best with one flow per NIC. Additional flows on a single NIC
  reduced aggregate throughput.
- NCCL reached 79.8 GB/s across four NICs, 3.65x its one-NIC result.
- NIXL CPU transport reached 8.94 GB/s on one interface and 17.27 GB/s with
  four-rail striping.
- Object Store throughput increased with concurrency but was close to its
  observed maximum by eight flows.

Object Store stages GPU payloads through host memory, while NCCL uses GDRDMA,
so their GPU rates do not represent equivalent transfer paths. See the
[full results, tables, and figures](results.md) or download the
[machine-readable CSV](results/benchmark-results.csv).

## Benchmark design

The complete published matrix contains 32 measured scenarios:

- **Baseline:** one flow at 1 MiB, 64 MiB, and 1 GiB for all four supported
  transport/device series (12 results).
- **Single-interface saturation:** Object Store and NCCL use 1, 2, 4, and 8
  concurrent 1 GiB flows; NIXL uses 1, 2, and 4 (15 results total).
- **CXI scaling:** NIXL uses 1 and 2 flows per NIC (4 and 8 total logical
  flows), with every flow natively striped over all four CXI rails. NCCL uses
  1, 2, and 4 NICs with its independently selected single-NIC operating-point
  number of flows (5 results total).

For each single-NIC sweep, the reported operating point is the smallest tested
flow count whose median throughput is within 95% of that sweep's best result.
Each NCCL flow has a separate sender, receiver, and two-actor collective group
and remains limited to one NIC. NCCL multi-NIC throughput therefore comes from
independent flows. Each NIXL four-rail flow instead has one sender and receiver;
NIXL divides that logical payload evenly across the four rails. Thus the
four-rail 1/2-per-NIC cases launch 4/8 sender-receiver pairs and transfer 4/8
GiB per measured batch.

Every scenario uses three warmups and ten measured batches. Actor creation,
payload allocation, NIXL initialization, and NCCL collective creation happen
before the timer. Reported throughput is total bytes completed by all flows
divided by batch duration. Results include median aggregate GB/s and median and
nearest-rank p95 latency. Every received tensor is fully verified outside the
timer. NIXL lazily grows a fixed actor pool and reuses each rail/flow slot for
the rest of the session. This prevents delayed Ray worker teardown from leaving
old per-actor CXI registrations alongside the next scenario. The entire pool is
gracefully terminated after the last NIXL case.

## Build

The image adds CuPy and NIXL 1.3.2 to NERSC's PyTorch image:

```bash
podman-hpc build -t ray-bench-pytorch:26.01.01-nixl1.3.2 .
```

The benchmark deliberately uses NERSC's `--nccl-cu13` injection for the
supported NCCL/libfabric stack. NIXL also uses that option to obtain the Cray
libfabric and CXI dependencies; it does not use NCCL for its data path.

## Run

Allocate exactly two GPU nodes with all four GPUs visible to the one Slurm task
on each node. For example:

```bash
salloc -N 2 -C gpu -q interactive -t 01:30:00 \
  --ntasks-per-node=1 --cpus-per-task=128 --gpus-per-node=4
```

### Run one transport

```bash
./run_benchmarks.sh --transport object \
  --image ray-bench-pytorch:26.01.01-nixl1.3.2
./run_benchmarks.sh --transport nixl --nixl-rails 1 \
  --image ray-bench-pytorch:26.01.01-nixl1.3.2
./run_benchmarks.sh --transport nixl --nixl-rails 4 \
  --image ray-bench-pytorch:26.01.01-nixl1.3.2
./run_benchmarks.sh --transport nccl --nccl-suite sweep \
  --image ray-bench-pytorch:26.01.01-nixl1.3.2
```

Run each command in a fresh allocation. In particular, `--nixl-rails 1` and
`--nixl-rails 4` are deliberately separate sessions so the single-rail actor
pool and native four-rail registrations never coexist. The four-rail mode
automatically uses standard CXI MRs, which are required by the validated native
striping path. NCCL's sweep and NIC scaling are also separate: read the
`flows_per_nic` value from the sweep's `OPERATING_POINT`, then use it in a fresh
allocation:

```bash
./run_benchmarks.sh --transport nccl --nccl-suite scaling \
  --nccl-flows-per-nic <OPERATING_POINT> \
  --image ray-bench-pytorch:26.01.01-nixl1.3.2
```

Benchmark runs create raw logs only; publishing is a separate, explicit step.

A quick check of one transport is available with `--smoke`:

```bash
./run_benchmarks.sh --transport nixl --nixl-rails 1 --smoke \
  --image ray-bench-pytorch:26.01.01-nixl1.3.2 \
  --nixl-device cxi3
```

An independent native-multi-rail probe uses one NIXL agent on each node rather
than one agent per CXI device. It asks each agent to discover all four devices,
uses CXI standard MRs directly, raises the DRAM rail-selection bandwidth ceiling
for the test, transfers and fully verifies one 1 GiB CPU tensor, and then
requires log evidence that NIXL created four rails, registered memory across
four rails, and enabled striping:

```bash
./run_nixl_native_multirail_test.sh \
  --image ray-bench-pytorch:26.01.01-nixl1.3.2
```

Run this diagnostic from a fresh allocation. It does not contribute a row to
the canonical benchmark matrix.

Add `--verbose` to stream complete Ray and library logs. Normally the terminal
shows only concise `RUN`, `STACK`, `NETWORK`, `PATH`, `AFFINITY`, `RESULT`,
`OPERATING_POINT`, `ACTOR_POOL`, `ACTOR_CLEANUP`, `ERROR`, `SHUTDOWN`, and
`ARTIFACT` records. Complete output is retained in a timestamped local
directory that is ignored by Git.

Ray 2.54 container teardown can be slow and noisy after a clean driver
shutdown. Podman's `--init` was tested and shortened that teardown, but it also
propagated Ray's termination statuses as a failed Slurm step. The benchmark
therefore does not use it; transport success is never inferred from a failed
launcher status.

## Transport configuration

Object Store sessions receive GPU devices when needed but no CXI devices and
no `--nccl-cu13` injection. The driver verifies that both Ray node addresses
match their NERSC `<hostname>-hsn0` address.

The launcher and drivers preserve the measured Perlmutter locality instead of
relying on Ray's logical CPU count to pin processes. The topology observed on
both test nodes was:

| Network target | PCI address | NUMA CPUs | Closest GPU |
|---|---|---|---|
| `hsn0` / `cxi0` | `0000:c2:00.0` | NUMA 0: `0-15,64-79` | GPU 3, `0000:c1:00.0` |
| `cxi1` | `0000:81:00.0` | NUMA 1: `16-31,80-95` | GPU 2, `0000:82:00.0` |
| `cxi2` | `0000:42:00.0` | NUMA 2: `32-47,96-111` | GPU 1, `0000:41:00.0` |
| `cxi3` | `0000:01:00.0` | NUMA 3: `48-63,112-127` | GPU 0, `0000:03:00.0` |

Every step has one task per node, requests 128 logical CPUs and four GPUs for
that task, and explicitly sets `--gpu-bind=none`, so each Ray runtime sees all
four GPUs. For Object Store, a small container entrypoint discovers
`hsn0` through sysfs, binds the whole Ray process tree to its NUMA CPU set
before Ray starts, and sets `CUDA_VISIBLE_DEVICES=3`. Linux first-touch memory
placement is therefore local to NUMA 0. Ray advertises 32 CPUs and one GPU for
that session. NIXL and NCCL retain all CPUs and GPUs: every NIXL actor discovers its
selected CXI device through sysfs and pins all existing process threads to
that device's NUMA CPU set; every NCCL actor additionally resolves its assigned
GPU PCI address through CUDA and requires its GPU and CXI NUMA nodes to match.
This is actor-aware, whereas `--gpu-bind=closest` would bind each Slurm task,
not the individual Ray actors sharing that task.

Each case emits an `AFFINITY` record containing target/GPU PCI addresses, NUMA
domains, CPU sets, and actor count. The publisher requires complete, consistent
affinity records and also requires the intended per-session Slurm binding.

NIXL sessions receive `/dev/cxi0` through `/dev/cxi3` plus `/dev/cxi_sbl`.
For `--nixl-rails 1`, each CPU actor sets `FI_PROVIDER=cxi` and
`FI_CXI_DEVICE_NAME=cxi3`, then pins its threads to that rail's NUMA CPUs before
initializing its NIXL agent. For `--nixl-rails 4`, each actor retains the full
four-NUMA CPU set and initializes one NIXL agent with
`FI_CXI_DEVICE_NAME=cxi3,cxi2,cxi1,cxi0`; NIXL stripes each payload across those
four rails. The container retains the mounted CUDA driver needed by `nixl_cu13`,
but Ray advertises zero GPUs and `CUDA_VISIBLE_DEVICES` is empty so the CPU-only
NIXL path cannot initialize a GPU communication stack.

The launcher sets `FI_MR_CACHE_MAX_COUNT=1` for NIXL. Ray/NIXL explicitly
deregisters each receive buffer, but libfabric otherwise caches up to 1024
deregistered regions by default. Repeated 1 GiB buffers can therefore retain
CXI page-table resources until an actor exits. A one-entry cache bounds that
retention per actor while keeping the CXI provider's cached-registration path
enabled; disabling the cache entirely caused a completed smoke transfer to
stall before returning to Ray. Reusing a bounded actor pool limits retained
registrations to the largest requested concurrency instead of accumulating a
new cache for every case. The driver emits per-case `ACTOR_POOL` evidence and a
single final `ACTOR_CLEANUP`; the publisher requires both. This is limited to
NIXL and does not alter the NCCL stack. Run only one NIXL session per allocation
because final native CXI cleanup can lag Ray's session exit.

The four-rail mode always sets `FI_CXI_OPTIMIZED_MRS=false`: optimized MR
registration failed before the native striped transfer, while direct standard
MR registration completed on all four rails. `--nixl-standard-mrs` applies the
same setting to an optional controlled single-rail comparison. The NIXL `STACK`
record captures the effective value and rail policy so the results remain
distinguishable.

NCCL sessions additionally receive `/dev/gdrdrv`. The launcher inherits the
general NERSC settings from `--nccl-cu13` and adds only the settings that define
this experiment:

```text
NCCL_NET=AWS Libfabric
NCCL_NET_GDR_READ=1
NCCL_NETDEVS_POLICY=MAX:1
FI_PROVIDER=cxi
FI_CXI_DEVICE_NAME=<actor rail>
```

`--nccl-cu13` already supplies PHB, the HSN bootstrap selection, the memory
registration configuration, and the injected libraries. The benchmark does
not set the earlier DMA-BUF or CUDA sync-memops compatibility workarounds; the
default configuration passed three repeated 1 GiB qualification transfers.
`/dev/gdrdrv` is the GDRCopy kernel interface; having `libgdrapi.so` without
that character device is insufficient for the validated PHB path.

The launcher explicitly removes `FI_CXI_DISABLE_DMABUF_CUDA` and
`FI_CXI_DISABLE_CUDA_SYNC_MEMOPS` from the container environment before Ray
starts. The NCCL driver's `STACK` record captures their effective values for
auditability without treating inherited settings as a driver error.

## Publish results

The publisher consumes five locally retained transport logs and requires
matching images and Ray, PyTorch, and CUDA versions. Its command-line interface
is documented by `./publish_results.sh --help`.

A successful publication creates:

- [The canonical 32-row CSV](results/benchmark-results.csv)
- [Payload-size baseline figure](docs/baseline-throughput.svg)
- [Single-interface flow-scaling figure](docs/single-nic-flow-scaling.svg)
- [Multi-rail scaling figure](docs/multi-nic-scaling.svg)

The publisher validates every in-scope result and its path, payload, affinity,
stack, and lifecycle evidence. It renders all four outputs to staging files
before atomically replacing any canonical artifact, so missing or invalid
in-scope measurements cannot partially update the published results.

The public CSV contains measurement, configuration, software-version, and
validation fields, but no local log paths or run-directory identifiers. Raw
logs and their provenance index remain local and are ignored by Git.

The main implementation is split into one launcher, four transport drivers,
shared benchmark/statistics code, and one strict result publisher. See the
[detailed results](results.md) and [previous testing](docs/previous-testing.md)
for the findings and technical background.

## Local checks

The dependency-free checks can run on a login node with Python 3.11:

```bash
python3.11 -m unittest -v test_benchmark_tools.py
python3.11 -m py_compile \
  benchmark_stats.py benchmark_common.py ray_transfer_bench.py \
  ray_nccl_bench.py ray_nixl_bench.py ray_nixl_multirail_bench.py \
  plot_benchmark_results.py
bash -n run_benchmarks.sh publish_results.sh
```

## References

- [Ray Direct Transport](https://docs.ray.io/en/latest/ray-core/direct-transport.html)
- [Ray on Slurm](https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html)
- [NERSC Perlmutter architecture](https://docs.nersc.gov/systems/perlmutter/architecture/)
- [NERSC GPU affinity](https://docs.nersc.gov/jobs/affinity/)
- [AWS OFI NCCL](https://github.com/aws/aws-ofi-nccl)
- [libfabric CXI provider](https://ofiwg.github.io/libfabric/main/man/fi_cxi.7.html)
- [NIXL](https://github.com/ai-dynamo/nixl)
