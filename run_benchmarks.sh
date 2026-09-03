#!/usr/bin/env bash
# Run from the shell returned by a two-node Perlmutter GPU allocation.

set -euo pipefail

die() {
  echo "ERROR: $*" >&2
  exit 2
}

IMAGE=""
PROFILE=full
TRANSPORT=""
VERBOSE=0
NIXL_RAILS=""
NCCL_SUITE=""
NCCL_FLOWS_PER_NIC=""
while (( $# )); do
  case "$1" in
    --image)
      (( $# >= 2 )) || die "--image requires a value"
      IMAGE="$2"
      shift 2
      ;;
    --smoke)
      PROFILE=smoke
      shift
      ;;
    --transport)
      (( $# >= 2 )) || die "--transport requires a value"
      TRANSPORT="$2"
      shift 2
      ;;
    --nixl-rails)
      (( $# >= 2 )) || die "--nixl-rails requires a value"
      NIXL_RAILS="$2"
      shift 2
      ;;
    --nccl-suite)
      (( $# >= 2 )) || die "--nccl-suite requires a value"
      NCCL_SUITE="$2"
      shift 2
      ;;
    --nccl-flows-per-nic)
      (( $# >= 2 )) || die "--nccl-flows-per-nic requires a value"
      NCCL_FLOWS_PER_NIC="$2"
      shift 2
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    -h|--help)
      echo "Usage: $0 --image IMAGE --transport object|nixl|nccl [--smoke] [--nixl-rails 1|4] [--nccl-suite sweep|scaling] [--nccl-flows-per-nic N] [--verbose]"
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -n "${IMAGE}" ]] || die "--image is required"
[[ -n "${TRANSPORT}" ]] || die "--transport is required"
case "${TRANSPORT}" in
  object|nixl|nccl) ;;
  *) die "--transport must be one of object, nixl, or nccl" ;;
esac
if [[ -n "${NIXL_RAILS}" ]]; then
  [[ "${NIXL_RAILS}" =~ ^(1|4)$ ]] \
    || die "--nixl-rails must be 1 or 4"
  [[ "${TRANSPORT}" == "nixl" ]] \
    || die "--nixl-rails requires a NIXL transport run"
fi
if [[ "${TRANSPORT}" == "nixl" ]]; then
  if [[ "${PROFILE}" == "full" && -z "${NIXL_RAILS}" ]]; then
    die "full NIXL runs require --nixl-rails 1 or --nixl-rails 4"
  fi
  if [[ "${PROFILE}" == "smoke" ]]; then
    [[ -z "${NIXL_RAILS}" || "${NIXL_RAILS}" == "1" ]] \
      || die "--smoke supports only --nixl-rails 1"
    NIXL_RAILS=1
  fi
fi
if [[ -n "${NCCL_SUITE}" ]]; then
  [[ "${NCCL_SUITE}" =~ ^(sweep|scaling)$ ]] \
    || die "--nccl-suite must be sweep or scaling"
  [[ "${TRANSPORT}" == "nccl" && "${PROFILE}" == "full" ]] \
    || die "--nccl-suite requires a full NCCL transport run"
fi
if [[ -n "${NCCL_FLOWS_PER_NIC}" ]]; then
  [[ "${NCCL_FLOWS_PER_NIC}" =~ ^(1|2|4|8)$ ]] \
    || die "--nccl-flows-per-nic must be one of 1, 2, 4, or 8"
  [[ "${TRANSPORT}" == "nccl" && "${NCCL_SUITE}" == "scaling" ]] \
    || die "--nccl-flows-per-nic requires --transport nccl --nccl-suite scaling"
fi
if [[ "${TRANSPORT}" == "nccl" && "${PROFILE}" == "full" ]]; then
  [[ -n "${NCCL_SUITE}" ]] \
    || die "full NCCL runs require --nccl-suite sweep or --nccl-suite scaling"
  if [[ "${NCCL_SUITE}" == "scaling" ]]; then
    [[ -n "${NCCL_FLOWS_PER_NIC}" ]] \
      || die "NCCL scaling requires --nccl-flows-per-nic"
  else
    [[ -z "${NCCL_FLOWS_PER_NIC}" ]] \
      || die "NCCL sweep does not accept --nccl-flows-per-nic"
  fi
fi
if [[ "${TRANSPORT}" == "nccl" && "${PROFILE}" == "smoke" ]]; then
  [[ -z "${NCCL_SUITE}" && -z "${NCCL_FLOWS_PER_NIC}" ]] \
    || die "NCCL smoke does not accept suite or flow-count options"
fi
[[ -n "${SLURM_JOB_ID:-}" ]] || die "run inside a Slurm allocation"
[[ -n "${SLURM_JOB_NODELIST:-}" ]] || die "SLURM_JOB_NODELIST is not set"
for command_name in scontrol srun podman-hpc timeout tee tail; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || die "required command not found: ${command_name}"
done

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
(( ${#NODES[@]} == 2 )) \
  || die "the benchmark requires exactly two nodes; found ${#NODES[@]}"

WORKDIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-128}"
GPUS_FOR_RAY_TASK=4
[[ "${CPUS_PER_TASK}" =~ ^[1-9][0-9]*$ ]] \
  || die "invalid SLURM_CPUS_PER_TASK=${CPUS_PER_TASK}"
(( CPUS_PER_TASK == 128 )) \
  || die "the benchmark allocation requires --cpus-per-task=128"

numeric_job_id="${SLURM_JOB_ID%%[^0-9]*}"
[[ "${numeric_job_id}" =~ ^[1-9][0-9]*$ ]] \
  || die "cannot derive a port from SLURM_JOB_ID=${SLURM_JOB_ID}"
PORT_BASE="$((20000 + numeric_job_id % 15000))"
SESSION_INDEX=0
RUN_STAMP="${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${WORKDIR}/logs/${RUN_STAMP}"
mkdir -p -- "${LOG_DIR}"

PREFLIGHT_SRUN_ARGS=(
  --nodes=2
  --ntasks=2
  --ntasks-per-node=1
  --exact
  --cpus-per-task="${CPUS_PER_TASK}"
  --cpu-bind=none
  --gpus-per-task="${GPUS_FOR_RAY_TASK}"
  --gpu-bind=none
)

required_devices=()
if [[ "${TRANSPORT}" == "nixl" || "${TRANSPORT}" == "nccl" ]]; then
  required_devices+=(/dev/cxi0 /dev/cxi1 /dev/cxi2 /dev/cxi3 /dev/cxi_sbl)
fi
if [[ "${TRANSPORT}" == "nccl" ]]; then
  required_devices+=(/dev/gdrdrv)
fi
if (( ${#required_devices[@]} )); then
  echo "RUN validation=host-devices nodes=2 transport=${TRANSPORT}"
  srun "${PREFLIGHT_SRUN_ARGS[@]}" bash -lc '
    for device in "$@"; do
      [[ -c "$device" ]] || {
        echo "ERROR: $(hostname) is missing character device $device" >&2
        exit 1
      }
    done
    printf -v device_list "%s," "$@"
    echo "PATH hostname=$(hostname) devices=${device_list%,} status=pass"
  ' ray-bench-device-check "${required_devices[@]}"
fi

run_session() {
  local transport="$1"
  local driver="$2"
  local driver_profile="$3"
  local timeout_seconds=3600
  if [[ "${driver_profile}" == "smoke" ]]; then
    timeout_seconds=600
  fi
  (( SESSION_INDEX += 1 ))
  local port="$((PORT_BASE + SESSION_INDEX))"
  local ray_address="${NODES[0]}:${port}"
  local suffix="${transport}-${driver_profile}"
  local log="${LOG_DIR}/${suffix}.log"
  local session_cpus="${CPUS_PER_TASK}"
  local session_gpus="${GPUS_FOR_RAY_TASK}"
  local ray_num_cpus="${CPUS_PER_TASK}"
  local ray_num_gpus="${GPUS_FOR_RAY_TASK}"
  local cpu_bind="none"
  local mem_bind="first-touch"
  local gpu_bind="none"
  local cpu_target="distributed"
  local cuda_visible_devices="0,1,2,3"
  local -a session_srun_args=(
    --nodes=2
    --ntasks=2
    --ntasks-per-node=1
    --exact
    --cpus-per-task="${session_cpus}"
    --cpu-bind=none
    --gpus-per-task="${session_gpus}"
    --gpu-bind=none
  )
  if [[ "${transport}" == "object" ]]; then
    # hsn0 and physical GPU 3 are both on NUMA node 0 on Perlmutter.
    # Use all four GPUs for the one task on each node with --gpu-bind=none.
    # The container entrypoint then binds the whole Ray
    # process tree to hsn0 and selects physical GPU 3.
    ray_num_cpus=32
    ray_num_gpus=1
    cpu_target="hsn0"
    cuda_visible_devices=3
  fi
  if [[ "${transport}" == "nixl" ]]; then
    ray_num_gpus=0
    cuda_visible_devices=""
  fi

  local -a podman_args=(
    run
    --rm
    --net host
    --shm-size=40GB
    --gpu
    -v "${WORKDIR}:/workdir"
    -w /workdir
    --env "RAY_ADDRESS=${ray_address}"
    --env "RAY_SYMMETRIC_RUN_CLUSTER_WAIT_TIMEOUT=180"
    --env "RAY_NUM_CPUS=${ray_num_cpus}"
    --env "RAY_NUM_GPUS=${ray_num_gpus}"
    --env "RAY_BENCH_SESSION_CPU_TARGET=${cpu_target}"
    --env "CUDA_VISIBLE_DEVICES=${cuda_visible_devices}"
    --env "PYTHONUNBUFFERED=1"
    --env "RAY_DEDUP_LOGS=0"
    --env "SLURM_JOB_ID=${SLURM_JOB_ID}"
    --env "SLURM_JOB_NUM_NODES=2"
    --env "SLURM_CPUS_PER_TASK=${session_cpus}"
    --env "SLURM_GPUS_PER_TASK=${session_gpus}"
    --env SLURMD_NODENAME
  )

  if [[ "${transport}" == "nixl" || "${transport}" == "nccl" ]]; then
    podman_args+=(
      --nccl-cu13
      --device=/dev/cxi0
      --device=/dev/cxi1
      --device=/dev/cxi2
      --device=/dev/cxi3
      --device=/dev/cxi_sbl
      --env "FI_PROVIDER=cxi"
      --env "FI_CXI_DEVICE_NAME=cxi3,cxi2,cxi1,cxi0"
      --env "FI_LOG_LEVEL=info"
      --env "FI_LOG_PROV=cxi"
    )
  fi
  if [[ "${transport}" == "nixl" ]]; then
    podman_args+=(
      --env "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0"
      --env "RAY_NIXL_BACKEND=LIBFABRIC"
      --env "FI_MR_CACHE_MAX_COUNT=1"
      --env "NIXL_LOG_LEVEL=DEBUG"
    )
    if [[ "${NIXL_RAILS}" == "4" ]]; then
      podman_args+=(
        --env "FI_CXI_OPTIMIZED_MRS=false"
        --env "NIXL_LIBFABRIC_MAX_BW_PER_DRAM_SEG=1000"
      )
    fi
  elif [[ "${transport}" == "nccl" ]]; then
    podman_args+=(
      --device=/dev/gdrdrv
      --env "NCCL_NET=AWS Libfabric"
      --env "NCCL_DEBUG=INFO"
      --env "NCCL_DEBUG_SUBSYS=NET"
      --env "NCCL_NET_GDR_READ=1"
      --env "NCCL_NETDEVS_POLICY=MAX:1"
    )
  fi

  local -a driver_args=(--profile "${driver_profile}")
  if [[ "${transport}" == "nccl" && "${driver_profile}" == "scaling" ]]; then
    driver_args+=(--flows-per-nic "${NCCL_FLOWS_PER_NIC}")
  fi

  local -a command=(
    podman-hpc "${podman_args[@]}" "${IMAGE}"
    bash -lc 'exec env \
      -u FI_CXI_DISABLE_DMABUF_CUDA \
      -u FI_CXI_DISABLE_CUDA_SYNC_MEMOPS \
      python -u /workdir/session_affinity_exec.py ray symmetric-run \
      --address "$RAY_ADDRESS" \
      --min-nodes "$SLURM_JOB_NUM_NODES" \
      --num-cpus "$RAY_NUM_CPUS" \
      --num-gpus "$RAY_NUM_GPUS" \
      -- \
      python -u "/workdir/$1" "${@:2}"'
    ray-bench-entrypoint "${driver}" "${driver_args[@]}"
  )

  {
    local recorded_cuda_devices="${cuda_visible_devices:-none}"
    echo "SESSION image=${IMAGE} transport=${transport} profile=${driver_profile} cpus=${session_cpus} gpus=${session_gpus} ray_cpus=${ray_num_cpus} ray_gpus=${ray_num_gpus} cpu_bind=${cpu_bind} mem_bind=${mem_bind} gpu_bind=${gpu_bind} cpu_target=${cpu_target} cuda_visible_devices=${recorded_cuda_devices}"
  } >"${log}"

  echo "RUN session=${transport}-${driver_profile} log=${log}"
  set +e
  if (( VERBOSE )); then
    timeout --signal=TERM --kill-after=15s "${timeout_seconds}s" \
      srun "${session_srun_args[@]}" "${command[@]}" \
      2>&1 | tee -a "${log}"
  else
    timeout --signal=TERM --kill-after=15s "${timeout_seconds}s" \
      srun "${session_srun_args[@]}" "${command[@]}" \
      2>&1 | tee -a "${log}" \
      | awk '/(^|[[:space:]])(RUN|SESSION_AFFINITY|STACK|NETWORK|PATH|AFFINITY|RESULT|OPERATING_POINT|ACTOR_POOL|ACTOR_CLEANUP|ERROR|SHUTDOWN|ARTIFACT)[[:space:]]/ {print; fflush()}'
  fi
  local status=${PIPESTATUS[0]}
  set -e
  if (( status != 0 )); then
    echo "ERROR: session=${transport}-${driver_profile} status=${status} log=${log}" >&2
    tail -n 40 "${log}" >&2
  fi
  return "${status}"
}

if [[ "${TRANSPORT}" == "object" ]]; then
  run_session object ray_transfer_bench.py "${PROFILE}"
fi
if [[ "${TRANSPORT}" == "nixl" ]]; then
  if [[ "${PROFILE}" == "smoke" ]]; then
    run_session nixl ray_nixl_bench.py smoke
  elif [[ "${NIXL_RAILS}" == "1" ]]; then
    run_session nixl ray_nixl_bench.py single-rail
  else
    run_session nixl ray_nixl_multirail_bench.py four-rail
  fi
fi
if [[ "${TRANSPORT}" == "nccl" ]]; then
  if [[ "${PROFILE}" == "smoke" ]]; then
    run_session nccl ray_nccl_bench.py smoke
  else
    run_session nccl ray_nccl_bench.py "${NCCL_SUITE}"
  fi
fi

echo "ARTIFACT kind=raw-logs path=${LOG_DIR} canonical_results=unchanged"
