#!/usr/bin/env bash
# Launch from the shell returned by a NERSC interactive salloc.

set -euo pipefail

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ -n "${SLURM_JOB_ID:-}" ]] \
  || die "run this script inside a Slurm allocation (start one with salloc)"
[[ -n "${SLURM_JOB_NODELIST:-}" ]] || die "SLURM_JOB_NODELIST is not set"

for command_name in scontrol srun podman-hpc; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || die "required command not found: ${command_name}"
done

PODMANHPC_IMAGE="${IMAGE:-nersc/pytorch:26.01.01}"
BENCH="${BENCH:-object}"
BENCH_ARGS="${BENCH_ARGS:-}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-128}"

case "${BENCH}" in
  object)
    BENCH_SCRIPT=ray_transfer_bench.py
    BENCH_DESCRIPTION="Ray Object Store (CUDA/Torch ObjectRef)"
    FI_PROVIDER_MODE=""
    NCCL_GDR_MODE=""
    ;;
  rdt-tcp)
    BENCH_SCRIPT=ray_nccl_bench.py
    BENCH_DESCRIPTION="Ray Direct Transport (NCCL over OFI/TCP)"
    FI_PROVIDER_MODE=tcp
    NCCL_GDR_MODE=LOC
    ;;
  rdt-cxi)
    BENCH_SCRIPT=ray_nccl_bench.py
    BENCH_DESCRIPTION="Ray Direct Transport (NCCL over native CXI)"
    FI_PROVIDER_MODE=cxi
    # LOC is the known-working host-staged path on the tested Perlmutter
    # software stack. Set NCCL_NET_GDR_LEVEL=PHB explicitly to investigate
    # GPU Direct RDMA; that path hung in the July 2026 test environment.
    NCCL_GDR_MODE="${NCCL_NET_GDR_LEVEL:-LOC}"
    ;;
  nccl)
    die "BENCH=nccl is ambiguous; use BENCH=rdt-tcp or BENCH=rdt-cxi"
    ;;
  *)
    die "BENCH must be 'object', 'rdt-tcp', or 'rdt-cxi' (got ${BENCH})"
    ;;
esac

derive_gpus_per_task() {
  local value

  value="${SLURM_TRES_PER_TASK:-}"
  if [[ "${value}" =~ gres/gpu(:[^=,]+)?=([0-9]+) ]]; then
    echo "${BASH_REMATCH[2]}"
    return
  fi

  for value in "${SLURM_GPUS_PER_TASK:-}" "${SLURM_GPUS_ON_NODE:-}"; do
    if [[ "${value}" =~ ^[0-9]+$ ]]; then
      echo "${value}"
      return
    fi
  done

  echo 4
}

GPUS_PER_TASK="$(derive_gpus_per_task)"
[[ "${CPUS_PER_TASK}" =~ ^[1-9][0-9]*$ ]] \
  || die "invalid SLURM_CPUS_PER_TASK: ${CPUS_PER_TASK}"
[[ "${GPUS_PER_TASK}" =~ ^[1-9][0-9]*$ ]] \
  || die "could not derive a valid GPU count per task: ${GPUS_PER_TASK}"

mapfile -t nodes_array < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
(( ${#nodes_array[@]} >= 2 )) \
  || die "the benchmark requires at least two allocated nodes"

NUM_NODES="${SLURM_JOB_NUM_NODES:-${#nodes_array[@]}}"
[[ "${NUM_NODES}" =~ ^[1-9][0-9]*$ ]] || die "invalid node count: ${NUM_NODES}"
if (( NUM_NODES != ${#nodes_array[@]} )); then
  die "SLURM_JOB_NUM_NODES=${NUM_NODES}, but node list has ${#nodes_array[@]} nodes"
fi

head_node="${nodes_array[0]}"
if [[ "${SLURM_JOB_ID}" =~ ^([0-9]+) ]]; then
  numeric_job_id="${BASH_REMATCH[1]}"
else
  die "cannot derive a numeric port from SLURM_JOB_ID=${SLURM_JOB_ID}"
fi
port="$((20000 + numeric_job_id % 20000))"
ip_head="${head_node}:${port}"

export RAY_ADDRESS="${ip_head}"
export RAY_SYMMETRIC_RUN_CLUSTER_WAIT_TIMEOUT="${RAY_SYMMETRIC_RUN_CLUSTER_WAIT_TIMEOUT:-180}"
export PYTHONUNBUFFERED=1

WORKDIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
read -r -a BENCH_ARGV <<< "${BENCH_ARGS}"

PODMAN_RUN_ARGS=(
  run
  --rm
  --gpu
  --net host
  --shm-size=40GB
  -v "${WORKDIR}:/workdir"
  -w /workdir
  --env "RAY_BENCH_MODE=${BENCH}"
  --env "RAY_ADDRESS=${RAY_ADDRESS}"
  --env "RAY_SYMMETRIC_RUN_CLUSTER_WAIT_TIMEOUT=${RAY_SYMMETRIC_RUN_CLUSTER_WAIT_TIMEOUT}"
  --env "PYTHONUNBUFFERED=1"
  --env "SLURM_JOB_ID=${SLURM_JOB_ID}"
  --env "SLURM_JOB_NUM_NODES=${NUM_NODES}"
  --env "SLURM_CPUS_PER_TASK=${CPUS_PER_TASK}"
  --env "SLURM_TRES_PER_TASK=${SLURM_TRES_PER_TASK:-}"
  --env "SLURM_GPUS_PER_TASK=${GPUS_PER_TASK}"
  --env SLURMD_NODENAME
)

if [[ -n "${FI_PROVIDER_MODE}" ]]; then
  PODMAN_RUN_ARGS+=(
    --nccl-cu13
    --env "NCCL_NET=AWS Libfabric"
    --env "FI_PROVIDER=${FI_PROVIDER_MODE}"
    --env "NCCL_NET_GDR_LEVEL=${NCCL_GDR_MODE}"
  )
fi

if [[ "${BENCH}" == "rdt-cxi" ]]; then
  shopt -s nullglob
  CXI_DEVICES=(/dev/cxi*)
  shopt -u nullglob
  (( ${#CXI_DEVICES[@]} > 0 )) \
    || die "BENCH=rdt-cxi requires host CXI devices, but /dev/cxi* is absent"
  for device in "${CXI_DEVICES[@]}"; do
    [[ -c "${device}" ]] || die "CXI path is not a character device: ${device}"
    PODMAN_RUN_ARGS+=(--device="${device}")
  done
fi

DIAGNOSTIC_ENV_VARS=(
  NCCL_DEBUG
  NCCL_DEBUG_SUBSYS
  FI_LOG_LEVEL
  RAY_DEDUP_LOGS
)
for env_name in "${DIAGNOSTIC_ENV_VARS[@]}"; do
  if [[ -v "${env_name}" && -n "${!env_name}" ]]; then
    PODMAN_RUN_ARGS+=(--env "${env_name}=${!env_name}")
  fi
done
if [[ -n "${SCRATCH:-}" && -d "${SCRATCH}" ]]; then
  PODMAN_RUN_ARGS+=(-v "${SCRATCH}:${SCRATCH}")
fi

echo "Interactive Slurm job: ${SLURM_JOB_ID}"
echo "Ray nodes (${NUM_NODES}):"
printf '  %s\n' "${nodes_array[@]}"
echo "Ray head: ${RAY_ADDRESS}"
echo "Resources per Ray node: ${CPUS_PER_TASK} CPUs, ${GPUS_PER_TASK} GPUs"
echo "Image: ${PODMANHPC_IMAGE}"
echo "Benchmark: ${BENCH_DESCRIPTION}"
echo "Driver: ${BENCH_SCRIPT}"
echo "Benchmark args: ${BENCH_ARGS:-<defaults>}"
if [[ -n "${FI_PROVIDER_MODE}" ]]; then
  echo "NCCL_NET: AWS Libfabric"
  echo "FI_PROVIDER: ${FI_PROVIDER_MODE}"
  echo "NCCL_NET_GDR_LEVEL: ${NCCL_GDR_MODE}"
fi
if [[ "${BENCH}" == "rdt-cxi" ]]; then
  printf 'CXI devices:'
  printf ' %s' "${CXI_DEVICES[@]}"
  printf '\n'
fi
for env_name in "${DIAGNOSTIC_ENV_VARS[@]}"; do
  if [[ -v "${env_name}" && -n "${!env_name}" ]]; then
    echo "${env_name}: ${!env_name}"
  fi
done

# Exactly one Slurm task/container is started per node. The login shell
# initializes the NERSC image before symmetric-run starts Ray on every node.
srun \
  --nodes="${NUM_NODES}" \
  --ntasks="${NUM_NODES}" \
  --ntasks-per-node=1 \
  --gpus-per-task="${GPUS_PER_TASK}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  podman-hpc "${PODMAN_RUN_ARGS[@]}" \
  "${PODMANHPC_IMAGE}" \
  bash -lc 'exec ray symmetric-run \
    --address "$RAY_ADDRESS" \
    --min-nodes "$SLURM_JOB_NUM_NODES" \
    --num-cpus "$SLURM_CPUS_PER_TASK" \
    --num-gpus "$SLURM_GPUS_PER_TASK" \
    -- \
    python -u "/workdir/$1" "${@:2}"' \
  ray-bench-entrypoint "${BENCH_SCRIPT}" "${BENCH_ARGV[@]}"
