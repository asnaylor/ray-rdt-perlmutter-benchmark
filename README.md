# Diagnosing Ray transfer performance on Perlmutter

This benchmark moves the same CUDA PyTorch payload between Ray actors on two
NERSC Perlmutter A100 40 GB nodes. It supports three transfer paths:

| Mode | Transfer path |
|---|---|
| `object` | Ray Object Store: GPU → CPU object store → network → GPU |
| `rdt-tcp` | Ray Direct Transport with NCCL → AWS OFI → libfabric `tcp` |
| `rdt-cxi` | Ray Direct Transport with NCCL → AWS OFI → Slingshot `cxi` |

The goal is to verify which provider NCCL actually uses and show the performance
impact when CXI is unavailable and the AWS OFI plugin selects TCP.

## Observed performance

One-way transfer of a 1 GiB CUDA `torch.uint8` tensor, with one warmup and five
measured iterations:

| Mode | Median | Equivalent payload rate |
|---|---:|---:|
| `object` | 0.490 GB/s | 3.917 Gbit/s |
| `rdt-tcp` | 0.935 GB/s | 7.480 Gbit/s |
| `rdt-cxi` | **16.876 GB/s** | **135.012 Gbit/s** |

On the tested node pair, RDT over CXI was approximately 18.1× faster than RDT
over libfabric TCP and 34.5× faster than the Ray Object Store path. RDT over
TCP was approximately 1.9× faster than the Object Store path. The CXI result
used `NCCL_NET_GDR_LEVEL=LOC`, which uses CXI with host staging rather than GPU
Direct RDMA.

This is a one-way, one-GPU-per-node microbenchmark. It does not measure DDP,
all-reduce, multi-GPU scaling, or end-to-end training performance.

## The problem and fix

The NERSC `podman-hpc --nccl-cu13` module supplied the AWS OFI NCCL plugin,
Cray libfabric, and `libcxi`, but in the tested configuration the host
`/dev/cxi*` devices were not present inside the container. Without access to the
CXI devices, NCCL/OFI selected libfabric TCP:

```text
NET/OFI Selected Provider is tcp
```

The launcher in this repository fixes the provider-selection problem by passing
all host `/dev/cxi*` character devices into the container and setting
`FI_PROVIDER=cxi`. It fails instead of silently falling back when CXI is not
available. A working CXI run reports:

```text
NET/OFI Selected Provider is cxi
```

`Bootstrap: Using hsn0` does not mean the payload is using TCP. NCCL can use the
HSN IP interface for bootstrap while transferring the payload through CXI. The
`Selected Provider` line identifies the data provider.

### GPU Direct RDMA note

With CXI selected and NERSC's default `NCCL_NET_GDR_LEVEL=PHB`, NCCL initialized
channels as `GDRDMA/Shared`, but the first Ray RDT transfer timed out after 60
seconds. Setting `NCCL_NET_GDR_LEVEL=LOC` made the transfer complete by using
CXI with host staging. Specify `LOC` explicitly when running this benchmark
because the NERSC environment may already export `PHB`. The underlying cause of
the PHB/GDRDMA timeout has not been isolated.

## Test environment

- Ray 2.54.0
- PyTorch 2.10.0a0 from the NERSC `26.01` image
- CUDA 13.1
- NCCL 2.29.2
- AWS OFI NCCL plugin 1.6.0
- Cray libfabric 1.22
- CuPy 14.1.1
- NVIDIA A100 40 GB

All modes use the same one-dimensional CUDA tensor and receiver checksum.
Payload creation is outside the timer; receiver completion and CUDA
synchronization are inside.

## Run the benchmark

Allocate two A100 40 GB nodes:

```bash
export NERSC_ACCOUNT=<your_gpu_account>

salloc \
  --nodes=2 \
  --qos=interactive \
  --time=00:30:00 \
  --constraint="gpu&hbm40g" \
  --account="${NERSC_ACCOUNT}" \
  --ntasks-per-node=1 \
  --gpus-per-node=4 \
  --cpus-per-task=128
```

Build the image:

```bash
podman-hpc build \
  -t ray-bench-pytorch:26.01.01-cupy14.1.1 \
  -f Containerfile .

export IMAGE=ray-bench-pytorch:26.01.01-cupy14.1.1
```

Run one mode. Start with 64 MiB, then use `--size-mb 1024` for the full test.

```bash
# Ray Object Store
BENCH=object \
BENCH_ARGS="--size-mb 64 --warmup 1 --iterations 2" \
  ./run_ray_symmetric_bench_interactive.sh

# RDT over libfabric TCP
NCCL_DEBUG=INFO \
BENCH=rdt-tcp \
BENCH_ARGS="--size-mb 64 --warmup 1 --iterations 2" \
  ./run_ray_symmetric_bench_interactive.sh

# RDT over native CXI with host staging
NCCL_NET_GDR_LEVEL=LOC \
NCCL_DEBUG=INFO \
BENCH=rdt-cxi \
BENCH_ARGS="--size-mb 64 --warmup 1 --iterations 2" \
  ./run_ray_symmetric_bench_interactive.sh
```

Full 1 GiB CXI benchmark:

```bash
NCCL_NET_GDR_LEVEL=LOC \
NCCL_DEBUG=INFO \
BENCH=rdt-cxi \
BENCH_ARGS="--size-mb 1024 --warmup 1 --iterations 5" \
  ./run_ray_symmetric_bench_interactive.sh
```

Each run ends with a machine-readable summary:

```text
RESULT benchmark=rdt-cxi bytes=1073741824 median_GBps=16.876438 median_Gbitps=135.011503
```

To reproduce the PHB/GDRDMA timeout:

```bash
NCCL_NET_GDR_LEVEL=PHB \
NCCL_DEBUG=INFO \
BENCH=rdt-cxi \
BENCH_ARGS="--size-mb 64 --warmup 1 --iterations 2" \
  ./run_ray_symmetric_bench_interactive.sh
```

For detailed diagnostics, set `FI_LOG_LEVEL=debug` and `NCCL_DEBUG=TRACE`.

## Files

- `ray_transfer_bench.py`: Object Store transfer of a CUDA tensor.
- `ray_nccl_bench.py`: RDT/NCCL transfer of the same CUDA tensor.
- `run_ray_symmetric_bench_interactive.sh`: Slurm/Podman-HPC launcher.
- `Containerfile`: image based on NERSC PyTorch `26.01.01`.

## References

- [Ray Direct Transport](https://docs.ray.io/en/latest/ray-core/direct-transport.html)
- [NERSC Podman-HPC](https://docs.nersc.gov/development/containers/podman-hpc/overview/)
- [libfabric CXI provider](https://ofiwg.github.io/libfabric/v1.21.1/man/fi_cxi.7.html)
