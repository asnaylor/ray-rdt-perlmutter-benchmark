#!/usr/bin/env bash
# Run from a fresh two-node Perlmutter GPU allocation.

set -euo pipefail

die() {
  echo "ERROR: $*" >&2
  exit 2
}

if (( $# != 2 )) || [[ "$1" != "--image" ]]; then
  die "usage: $0 --image IMAGE"
fi
IMAGE="$2"

[[ -n "${SLURM_JOB_ID:-}" ]] || die "run inside a Slurm allocation"
[[ -n "${SLURM_JOB_NODELIST:-}" ]] || die "SLURM_JOB_NODELIST is not set"
for command_name in scontrol srun podman-hpc timeout tee rg; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || die "required command not found: ${command_name}"
done

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
(( ${#NODES[@]} == 2 )) \
  || die "the test requires exactly two nodes; found ${#NODES[@]}"

WORKDIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-128}"
[[ "${CPUS_PER_TASK}" == "128" ]] \
  || die "the allocation requires --cpus-per-task=128"

numeric_job_id="${SLURM_JOB_ID%%[^0-9]*}"
[[ "${numeric_job_id}" =~ ^[1-9][0-9]*$ ]] \
  || die "cannot derive a port from SLURM_JOB_ID=${SLURM_JOB_ID}"
port="$((35000 + numeric_job_id % 15000))"
ray_address="${NODES[0]}:${port}"
run_stamp="${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)"
log_dir="${WORKDIR}/logs/${run_stamp}"
log="${log_dir}/nixl-native-multirail.log"
mkdir -p -- "${log_dir}"

srun_args=(
  --nodes=2
  --ntasks=2
  --ntasks-per-node=1
  --exact
  --cpus-per-task=128
  --cpu-bind=none
  --gpus-per-task=4
  --gpu-bind=none
)

required_devices=(/dev/cxi0 /dev/cxi1 /dev/cxi2 /dev/cxi3 /dev/cxi_sbl)
echo "RUN validation=host-devices test=nixl-native-multirail"
srun "${srun_args[@]}" bash -lc '
  for device in "$@"; do
    [[ -c "$device" ]] || {
      echo "ERROR: $(hostname) is missing character device $device" >&2
      exit 1
    }
  done
  echo "PATH hostname=$(hostname) devices=$* status=pass"
' nixl-native-multirail-device-check "${required_devices[@]}"

cat_session="SESSION image=${IMAGE} test=nixl-native-multirail cpus=128 gpus=4 ray_cpus=128 ray_gpus=0 devices=cxi3,cxi2,cxi1,cxi0 optimized_mrs=0 max_bw_per_dram_seg=1000"
echo "${cat_session}" >"${log}"
echo "RUN test=nixl-native-multirail log=${log}"

command=(
  podman-hpc run
  --rm
  --net host
  --shm-size=40GB
  --gpu
  -v "${WORKDIR}:/workdir"
  -w /workdir
  --nccl-cu13
  --device=/dev/cxi0
  --device=/dev/cxi1
  --device=/dev/cxi2
  --device=/dev/cxi3
  --device=/dev/cxi_sbl
  --env "RAY_ADDRESS=${ray_address}"
  --env "RAY_SYMMETRIC_RUN_CLUSTER_WAIT_TIMEOUT=180"
  --env "RAY_BENCH_SESSION_CPU_TARGET=distributed"
  --env "CUDA_VISIBLE_DEVICES="
  --env "PYTHONUNBUFFERED=1"
  --env "RAY_DEDUP_LOGS=0"
  --env "FI_PROVIDER=cxi"
  --env "FI_CXI_DEVICE_NAME=cxi3,cxi2,cxi1,cxi0"
  --env "FI_CXI_OPTIMIZED_MRS=0"
  --env "FI_MR_CACHE_MAX_COUNT=1"
  --env "NIXL_LIBFABRIC_MAX_BW_PER_DRAM_SEG=1000"
  --env "FI_LOG_LEVEL=info"
  --env "FI_LOG_PROV=cxi"
  --env "NIXL_LOG_LEVEL=DEBUG"
  --env "SLURM_JOB_ID=${SLURM_JOB_ID}"
  --env "SLURM_JOB_NUM_NODES=2"
  --env "SLURM_CPUS_PER_TASK=128"
  --env "SLURM_GPUS_PER_TASK=4"
  --env SLURMD_NODENAME
  "${IMAGE}"
  bash -lc 'exec env \
    -u FI_CXI_DISABLE_DMABUF_CUDA \
    -u FI_CXI_DISABLE_CUDA_SYNC_MEMOPS \
    python -u /workdir/session_affinity_exec.py ray symmetric-run \
    --address "$RAY_ADDRESS" \
    --min-nodes "$SLURM_JOB_NUM_NODES" \
    --num-cpus 128 \
    --num-gpus 0 \
    -- \
    python -u /workdir/nixl_native_multirail_test.py'
)

set +e
timeout --signal=TERM --kill-after=15s 600s \
  srun "${srun_args[@]}" "${command[@]}" 2>&1 \
  | tee -a "${log}" \
  | awk '/(^|[[:space:]])(RUN|SESSION_AFFINITY|NETWORK|MULTIRAIL_AGENT|RESULT|ACTOR_CLEANUP|ERROR|SHUTDOWN)[[:space:]]/ {print; fflush()}'
status=${PIPESTATUS[0]}
set -e
(( status == 0 )) \
  || die "native multi-rail session failed with status ${status}; log=${log}"

created="$(rg -c 'Created 4 rails using provider=cxi' "${log}" || true)"
registered="$(rg -c 'Registered memory on 4 rails' "${log}" || true)"
striped="$(rg -c 'use_striping=true' "${log}" || true)"
created="${created:-0}"
registered="${registered:-0}"
striped="${striped:-0}"
(( created >= 2 )) \
  || die "NIXL did not create four rails in both agents; log=${log}"
(( registered >= 2 )) \
  || die "NIXL did not register the payload on four rails; log=${log}"
(( striped >= 1 )) \
  || die "NIXL did not report multi-rail striping; log=${log}"

echo "MULTIRAIL_EVIDENCE created=${created} registered=${registered} striped=${striped} status=pass"
echo "ARTIFACT kind=diagnostic-log path=${log}"
