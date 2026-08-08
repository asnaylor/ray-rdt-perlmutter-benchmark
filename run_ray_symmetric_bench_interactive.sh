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

[[ -v IMAGE && -n "${IMAGE}" ]] \
  || die "IMAGE must be exported (for example: export IMAGE=ray-bench-pytorch:26.01.01-nixl1.3.2)"
PODMANHPC_IMAGE="${IMAGE}"
BENCH="${BENCH:-object}"
BENCH_ARGS="${BENCH_ARGS:-}"
NCCL_DEBUG_MODE="${NCCL_DEBUG:-INFO}"
NCCL_DEBUG_SUBSYS_MODE="${NCCL_DEBUG_SUBSYS:-NET}"
NEEDS_GPU_RUNTIME=1
RAY_USES_GPU=1
USES_NCCL=0
USES_NIXL=0
USES_NIXL_CXI=0
NEEDS_CXI=0
NIXL_BACKEND=""
NIXL_DEVICE=""
UCX_NET_DEVICE=""
TCP_NET_DEVICE=""
CXI_RAIL_COUNT=""
CXI_DEVICE_NAMES=""
NCCL_NETDEVS_POLICY_MODE=""
BENCH_FIXED_ARGV=()

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
    USES_NCCL=1
    FI_PROVIDER_MODE=tcp
    NCCL_GDR_MODE=LOC
    TCP_NET_DEVICE=hsn0
    ;;
  rdt-cxi)
    BENCH_SCRIPT=ray_nccl_bench.py
    BENCH_DESCRIPTION="Ray Direct Transport (NCCL over native CXI)"
    USES_NCCL=1
    NEEDS_CXI=1
    FI_PROVIDER_MODE=cxi
    # LOC is the known-working host-staged path on the tested Perlmutter
    # software stack. Set NCCL_NET_GDR_LEVEL=PHB explicitly to investigate
    # GPU Direct RDMA; that path hung in the July 2026 test environment.
    NCCL_GDR_MODE="${NCCL_NET_GDR_LEVEL:-LOC}"
    ;;
  rdt-nixl-cxi-cpu)
    BENCH_SCRIPT=ray_nixl_bench.py
    BENCH_DESCRIPTION="Ray Direct Transport (CPU NIXL/LIBFABRIC over CXI)"
    # The payload and Ray actors are CPU-only, but the nixl_cu13 LIBFABRIC
    # plugin has an ELF dependency on libcuda.so.1. Run on a GPU node and
    # expose the driver to Podman so the plugin can be loaded.
    RAY_USES_GPU=0
    USES_NIXL=1
    USES_NIXL_CXI=1
    NEEDS_CXI=1
    NIXL_BACKEND=LIBFABRIC
    NIXL_DEVICE=cpu
    BENCH_FIXED_ARGV=(--device cpu)
    FI_PROVIDER_MODE=cxi
    NCCL_GDR_MODE=""
    ;;
  rdt-nixl-ucx-tcp-cpu)
    BENCH_SCRIPT=ray_nixl_bench.py
    BENCH_DESCRIPTION="Ray Direct Transport (CPU NIXL/UCX over TCP/HSN)"
    NEEDS_GPU_RUNTIME=0
    RAY_USES_GPU=0
    USES_NIXL=1
    NIXL_BACKEND=UCX
    NIXL_DEVICE=cpu
    # NIXL's required intra-agent endpoint uses UCX self/memory. Pin network
    # traffic to hsn0, the interface selected by the nodes' route tables. No
    # management or loopback network device is exposed to UCX.
    UCX_NET_DEVICE="${UCX_NET_DEVICES:-hsn0}"
    [[ "${UCX_NET_DEVICE}" =~ ^hsn[0-9]+$ ]] \
      || die "UCX_NET_DEVICES must name exactly one HSN interface"
    BENCH_FIXED_ARGV=(--device cpu)
    FI_PROVIDER_MODE=""
    NCCL_GDR_MODE=""
    ;;
  nccl)
    die "BENCH=nccl is ambiguous; use BENCH=rdt-tcp or BENCH=rdt-cxi"
    ;;
  *)
    die "BENCH must be 'object', 'rdt-tcp', 'rdt-cxi', "\
"'rdt-nixl-cxi-cpu', or 'rdt-nixl-ucx-tcp-cpu' (got ${BENCH})"
    ;;
esac

if (( NEEDS_CXI )); then
  CXI_RAIL_COUNT="${CXI_RAILS:-1}"
  case "${CXI_RAIL_COUNT}" in
    1)
      # NCCL selected cxi3 in the original single-NIC run. Keep that device
      # fixed so the one-vs-four comparison does not change between runs.
      CXI_DEVICE_NAMES=cxi3
      NCCL_NETDEVS_POLICY_MODE=MAX:1
      ;;
    4)
      CXI_DEVICE_NAMES=cxi0,cxi1,cxi2,cxi3
      NCCL_NETDEVS_POLICY_MODE=ALL
      ;;
    *)
      die "CXI_RAILS must be 1 or 4 for BENCH=${BENCH} (got ${CXI_RAIL_COUNT})"
      ;;
  esac
  IFS=, read -ra SELECTED_CXI_DEVICES <<< "${CXI_DEVICE_NAMES}"
fi

if (( NEEDS_GPU_RUNTIME )); then
  CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-128}"
else
  CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-4}"
fi

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

[[ "${CPUS_PER_TASK}" =~ ^[1-9][0-9]*$ ]] \
  || die "invalid SLURM_CPUS_PER_TASK: ${CPUS_PER_TASK}"
if (( NEEDS_GPU_RUNTIME )); then
  GPUS_PER_TASK="$(derive_gpus_per_task)"
  if [[ ! "${GPUS_PER_TASK}" =~ ^[1-9][0-9]*$ ]]; then
    if [[ "${BENCH}" == "rdt-nixl-cxi-cpu" ]]; then
      die "${BENCH} uses CPU tensors but needs a GPU allocation because "\
"the nixl_cu13 LIBFABRIC plugin links libcuda.so.1"
    fi
    die "could not derive a valid GPU count per task: ${GPUS_PER_TASK}"
  fi
else
  GPUS_PER_TASK=0
fi
if (( RAY_USES_GPU )); then
  RAY_NUM_GPUS="${GPUS_PER_TASK}"
else
  RAY_NUM_GPUS=0
fi

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
BENCH_SIZE_MB=1024
for (( arg_index = 0; arg_index < ${#BENCH_ARGV[@]}; arg_index++ )); do
  case "${BENCH_ARGV[arg_index]}" in
    --size-mb)
      (( arg_index + 1 < ${#BENCH_ARGV[@]} )) \
        || die "--size-mb requires a value in BENCH_ARGS"
      (( arg_index += 1 ))
      BENCH_SIZE_MB="${BENCH_ARGV[arg_index]}"
      ;;
    --size-mb=*)
      BENCH_SIZE_MB="${BENCH_ARGV[arg_index]#--size-mb=}"
      ;;
  esac
done
[[ "${BENCH_SIZE_MB}" =~ ^[1-9][0-9]*$ ]] \
  || die "--size-mb must be a positive integer (got ${BENCH_SIZE_MB})"

PODMAN_RUN_ARGS=(
  run
  --rm
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
  --env "RAY_NUM_GPUS=${RAY_NUM_GPUS}"
  --env SLURMD_NODENAME
)

if (( NEEDS_GPU_RUNTIME )); then
  PODMAN_RUN_ARGS+=(--gpu)
fi

if (( USES_NCCL )); then
  PODMAN_RUN_ARGS+=(
    --nccl-cu13
    --env "NCCL_NET=AWS Libfabric"
    --env "NCCL_DEBUG=${NCCL_DEBUG_MODE}"
    --env "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS_MODE}"
    --env "FI_PROVIDER=${FI_PROVIDER_MODE}"
    --env "NCCL_NET_GDR_LEVEL=${NCCL_GDR_MODE}"
  )
  if [[ "${BENCH}" == "rdt-tcp" ]]; then
    PODMAN_RUN_ARGS+=(
      --env "FI_TCP_IFACE=${TCP_NET_DEVICE}"
      --env "NCCL_SOCKET_IFNAME=${TCP_NET_DEVICE}"
    )
  elif [[ "${BENCH}" == "rdt-cxi" ]]; then
    PODMAN_RUN_ARGS+=(
      --env "NCCL_NETDEVS_POLICY=${NCCL_NETDEVS_POLICY_MODE}"
    )
  fi
fi

if (( USES_NIXL_CXI )); then
  # NERSC's nccl-cu13 module also supplies the Cray libfabric and libcxi
  # libraries needed by NIXL. NCCL itself is not used by this data path.
  PODMAN_RUN_ARGS+=(--nccl-cu13)
fi

if (( NEEDS_CXI )); then
  CXI_DEVICES=()
  for device_name in "${SELECTED_CXI_DEVICES[@]}"; do
    device="/dev/${device_name}"
    [[ -c "${device}" ]] \
      || die "BENCH=${BENCH} requires CXI character device ${device}"
    CXI_DEVICES+=("${device}")
  done
  if [[ -e /dev/cxi_sbl ]]; then
    [[ -c /dev/cxi_sbl ]] \
      || die "CXI path is not a character device: /dev/cxi_sbl"
    CXI_DEVICES+=(/dev/cxi_sbl)
  fi
  for device in "${CXI_DEVICES[@]}"; do
    PODMAN_RUN_ARGS+=(--device="${device}")
  done
  PODMAN_RUN_ARGS+=(--env "FI_CXI_DEVICE_NAME=${CXI_DEVICE_NAMES}")
fi

FORWARDED_ENV_VARS=(
  NCCL_GDRCOPY_ENABLE
  OFI_NCCL_DISABLE_DMABUF
  FI_LOG_LEVEL
  FI_LOG_PROV
  NIXL_LOG_LEVEL
  RAY_DEDUP_LOGS
)
for env_name in "${FORWARDED_ENV_VARS[@]}"; do
  if (( USES_NIXL )) && [[ "${env_name}" == "NIXL_LOG_LEVEL" \
      || "${env_name}" == "RAY_DEDUP_LOGS" ]]; then
    continue
  fi
  if (( USES_NIXL_CXI )) && [[ "${env_name}" == "FI_LOG_LEVEL" \
      || "${env_name}" == "FI_LOG_PROV" ]]; then
    continue
  fi
  if [[ -v "${env_name}" && -n "${!env_name}" ]]; then
    PODMAN_RUN_ARGS+=(--env "${env_name}=${!env_name}")
  fi
done
if (( USES_NIXL )); then
  PODMAN_RUN_ARGS+=(
    --env "RAY_NIXL_BACKEND=${NIXL_BACKEND}"
    --env "NIXL_LOG_LEVEL=DEBUG"
    --env "RAY_DEDUP_LOGS=0"
  )
  if [[ "${NIXL_DEVICE}" == "cpu" ]]; then
    # Keep CUDA driver libraries available to the CUDA-13 NIXL package while
    # ensuring the benchmark actors can allocate only CPU tensors.
    PODMAN_RUN_ARGS+=(--env "CUDA_VISIBLE_DEVICES=")
  fi
fi
if (( USES_NIXL_CXI )); then
  PODMAN_RUN_ARGS+=(
    --env "FI_PROVIDER=cxi"
    --env "FI_LOG_LEVEL=info"
    --env "FI_LOG_PROV=cxi"
  )
fi
if [[ "${NIXL_BACKEND}" == "UCX" ]]; then
  PODMAN_RUN_ARGS+=(
    --env "UCX_TLS=self,tcp"
    --env "UCX_NET_DEVICES=${UCX_NET_DEVICE}"
    --env "UCX_LOG_LEVEL=info"
    --env "UCX_PROTO_INFO=y"
    --env "UCX_VFS_ENABLE=n"
  )
fi
if [[ -n "${SCRATCH:-}" && -d "${SCRATCH}" ]]; then
  PODMAN_RUN_ARGS+=(-v "${SCRATCH}:${SCRATCH}")
fi

NIXL_CONTAINER_PREFIX=()
if [[ "${NIXL_BACKEND}" == "UCX" ]]; then
  NIXL_CONTAINER_PREFIX=(/workdir/run_with_nixl_ucx.sh)
fi

echo "Interactive Slurm job: ${SLURM_JOB_ID}"
echo "Ray nodes (${NUM_NODES}):"
printf '  %s\n' "${nodes_array[@]}"
echo "Ray head: ${RAY_ADDRESS}"
echo "Slurm/container resources per node: ${CPUS_PER_TASK} CPUs, ${GPUS_PER_TASK} GPUs"
echo "Ray resources per node: ${CPUS_PER_TASK} CPUs, ${RAY_NUM_GPUS} GPUs"
echo "Image: ${PODMANHPC_IMAGE}"
echo "Benchmark: ${BENCH_DESCRIPTION}"
echo "Driver: ${BENCH_SCRIPT}"
echo "Benchmark args: ${BENCH_FIXED_ARGV[*]} ${BENCH_ARGS:-<defaults>}"
if (( USES_NCCL )); then
  echo "NCCL_NET: AWS Libfabric"
  echo "NCCL_DEBUG: ${NCCL_DEBUG_MODE}"
  echo "NCCL_DEBUG_SUBSYS: ${NCCL_DEBUG_SUBSYS_MODE}"
  echo "FI_PROVIDER: ${FI_PROVIDER_MODE}"
  echo "NCCL_NET_GDR_LEVEL: ${NCCL_GDR_MODE}"
  if [[ "${BENCH}" == "rdt-tcp" ]]; then
    echo "FI_TCP_IFACE: ${TCP_NET_DEVICE}"
    echo "NCCL_SOCKET_IFNAME: ${TCP_NET_DEVICE}"
  fi
fi
if (( USES_NIXL )); then
  echo "NIXL backend: ${NIXL_BACKEND} (forced by benchmark)"
  echo "NIXL tensor device: ${NIXL_DEVICE}"
  echo "NIXL_LOG_LEVEL: DEBUG"
  echo "RAY_DEDUP_LOGS: 0"
fi
if (( USES_NIXL_CXI )); then
  echo "FI_PROVIDER: cxi"
  echo "FI_LOG_LEVEL: info"
  echo "FI_LOG_PROV: cxi"
fi
if [[ "${NIXL_BACKEND}" == "UCX" ]]; then
  echo "UCX_TLS: self,tcp (self/memory is intra-process only)"
  echo "UCX_NET_DEVICES: ${UCX_NET_DEVICE} (inter-node TCP is HSN-only)"
  echo "NIXL UCX error handling: none (required for self/memory)"
  echo "UCX_LOG_LEVEL: info"
  echo "UCX_PROTO_INFO: y"
fi
if (( USES_NIXL )) && [[ "${NIXL_DEVICE}" == "cpu" ]]; then
  if (( NEEDS_GPU_RUNTIME )); then
    echo "GPU use: CUDA-13 NIXL runtime dependency only; CUDA devices are hidden"
  else
    echo "GPU use: none; GPU runtime and CUDA devices are not passed through"
  fi
  echo "CUDA_VISIBLE_DEVICES: <empty>"
fi
if (( NEEDS_CXI )); then
  echo "CXI_RAILS: ${CXI_RAIL_COUNT}"
  echo "FI_CXI_DEVICE_NAME: ${CXI_DEVICE_NAMES}"
  if [[ "${BENCH}" == "rdt-cxi" ]]; then
    echo "NCCL_NETDEVS_POLICY: ${NCCL_NETDEVS_POLICY_MODE}"
  fi
  printf 'Passed CXI device nodes:'
  printf ' %s' "${CXI_DEVICES[@]}"
  printf '\n'
fi
for env_name in "${FORWARDED_ENV_VARS[@]}"; do
  if (( USES_NIXL )) && [[ "${env_name}" == "NIXL_LOG_LEVEL" \
      || "${env_name}" == "RAY_DEDUP_LOGS" ]]; then
    continue
  fi
  if (( USES_NIXL_CXI )) && [[ "${env_name}" == "FI_LOG_LEVEL" \
      || "${env_name}" == "FI_LOG_PROV" ]]; then
    continue
  fi
  if [[ -v "${env_name}" && -n "${!env_name}" ]]; then
    echo "${env_name}: ${!env_name}"
  fi
done

SRUN_ARGS=(
  --nodes="${NUM_NODES}" \
  --ntasks="${NUM_NODES}" \
  --ntasks-per-node=1 \
  --cpus-per-task="${CPUS_PER_TASK}"
)
if (( NEEDS_GPU_RUNTIME )); then
  SRUN_ARGS+=(--gpus-per-task="${GPUS_PER_TASK}")
fi

CONTAINER_COMMAND=(
  podman-hpc "${PODMAN_RUN_ARGS[@]}" \
  "${PODMANHPC_IMAGE}" \
)
CONTAINER_COMMAND+=("${NIXL_CONTAINER_PREFIX[@]}")
CONTAINER_COMMAND+=(
  bash -lc 'exec ray symmetric-run \
    --address "$RAY_ADDRESS" \
    --min-nodes "$SLURM_JOB_NUM_NODES" \
    --num-cpus "$SLURM_CPUS_PER_TASK" \
    --num-gpus "$RAY_NUM_GPUS" \
    -- \
    python -u "/workdir/$1" "${@:2}"' \
  ray-bench-entrypoint "${BENCH_SCRIPT}" \
  "${BENCH_FIXED_ARGV[@]}" "${BENCH_ARGV[@]}"
)

# Exactly one Slurm task/container is started per node. The login shell
# initializes the NERSC image before symmetric-run starts Ray on every node.
for command_name in timeout tee grep; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || die "required command not found for evidence capture: ${command_name}"
done

if (( NEEDS_CXI )); then
  echo "Checking ${CXI_RAIL_COUNT} selected host CXI rail(s) on every node..."
  srun "${SRUN_ARGS[@]}" \
    env "FI_CXI_DEVICE_NAME=${CXI_DEVICE_NAMES}" \
    bash -lc 'command -v fi_info >/dev/null && fi_info -p cxi >/dev/null'
  echo "The selected host CXI rail(s) are available on every node."
fi

if [[ "${NIXL_BACKEND}" == "UCX" ]]; then
  echo "Checking UCX TCP interfaces ${UCX_NET_DEVICE} on every node..."
  srun "${SRUN_ARGS[@]}" \
    env "UCX_NET_DEVICES=${UCX_NET_DEVICE}" \
    bash -lc '
      IFS=, read -ra devices <<< "$UCX_NET_DEVICES"
      for device in "${devices[@]}"; do
        [[ -d "/sys/class/net/$device" ]] || exit 1
      done
    '
  echo "All requested UCX HSN interfaces are available on every node."
fi

if (( USES_NIXL )); then
  preflight_backend="${NIXL_BACKEND,,}"
  echo "Checking the NIXL ${NIXL_BACKEND} runtime on every node..."
  srun "${SRUN_ARGS[@]}" \
    podman-hpc "${PODMAN_RUN_ARGS[@]}" \
    "${PODMANHPC_IMAGE}" \
    "${NIXL_CONTAINER_PREFIX[@]}" \
    python -u /workdir/benchmark_preflight.py "${preflight_backend}"
  echo "The NIXL ${NIXL_BACKEND} runtime loaded on every node."
fi

RAY_BENCH_TIMEOUT_SECONDS="${RAY_BENCH_TIMEOUT_SECONDS:-180}"
[[ "${RAY_BENCH_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]] \
  || die "RAY_BENCH_TIMEOUT_SECONDS must be a positive integer"
LOG_ROOT="${SCRATCH:-/tmp}"
LOG_MODE="${BENCH}"
if (( NEEDS_CXI )); then
  LOG_MODE="${BENCH}-${CXI_RAIL_COUNT}cxi"
fi
LOG_MODE="${LOG_MODE}-${BENCH_SIZE_MB}MiB"
RAY_BENCH_LOG="${RAY_BENCH_LOG:-${LOG_ROOT}/ray-${LOG_MODE}-${SLURM_JOB_ID}.log}"
mkdir -p -- "$(dirname -- "${RAY_BENCH_LOG}")"
echo "Benchmark log: ${RAY_BENCH_LOG}"
echo "Benchmark timeout: ${RAY_BENCH_TIMEOUT_SECONDS} seconds"

# Persist the image selection because the launcher summary above is otherwise
# outside the benchmark's tee pipeline.
printf 'IMAGE=%s\n' "${PODMANHPC_IMAGE}" | tee "${RAY_BENCH_LOG}"

set +e
timeout --signal=TERM --kill-after=10s "${RAY_BENCH_TIMEOUT_SECONDS}s" \
  srun "${SRUN_ARGS[@]}" "${CONTAINER_COMMAND[@]}" \
  2>&1 | tee -a "${RAY_BENCH_LOG}"
run_status=${PIPESTATUS[0]}
set -e

evidence_status=pass
missing_evidence=()
require_evidence() {
  local pattern="$1"
  local description="$2"
  if ! grep -Eiq -- "${pattern}" "${RAY_BENCH_LOG}"; then
    evidence_status=fail
    missing_evidence+=("${description}")
  fi
}
reject_evidence() {
  local pattern="$1"
  local description="$2"
  if grep -Eiq -- "${pattern}" "${RAY_BENCH_LOG}"; then
    evidence_status=fail
    missing_evidence+=("${description}")
  fi
}
emit_record() {
  printf '%s\n' "$1" | tee -a "${RAY_BENCH_LOG}"
}

require_evidence \
  "RESULT benchmark=${BENCH} .*transfer_status=pass" \
  "the tensor transfer completed and fully verified"

case "${BENCH}" in
  object)
    require_evidence "Ray Object Store CUDA tensor benchmark" \
      "the object-store benchmark driver ran"
    require_evidence "Source: .*device=cuda" "the source tensor was CUDA"
    require_evidence "Destination: .*device=cuda" \
      "the destination tensor was CUDA"
    evidence_summary="path=ray-object-store source=cuda destination=cuda"
    ;;
  rdt-tcp|rdt-cxi)
    require_evidence "Ray NCCL Direct Transport benchmark \(${BENCH}\)" \
      "the NCCL Direct Transport benchmark ran"
    require_evidence \
      "NET/OFI.*Selected Provider is ${FI_PROVIDER_MODE}([^[:alnum:]_]|$)" \
      "AWS OFI NCCL selected ${FI_PROVIDER_MODE}"
    if [[ "${FI_PROVIDER_MODE}" == "cxi" ]]; then
      require_evidence \
        "Selected Provider is cxi \(found ${CXI_RAIL_COUNT} nics\)" \
        "AWS OFI NCCL exposed exactly ${CXI_RAIL_COUNT} CXI network device(s)"
      for device_name in "${SELECTED_CXI_DEVICES[@]}"; do
        require_evidence \
          "HCA [0-9]+ '${device_name}'" \
          "AWS OFI NCCL initialized ${device_name}"
      done
      for (( rail_id = 0; rail_id < CXI_RAIL_COUNT; rail_id++ )); do
        require_evidence \
          "Channel .*via NET/AWS Libfabric/${rail_id}/" \
          "an NCCL transfer channel used CXI network device ${rail_id}"
      done
      reject_evidence "NET/OFI.*Selected Provider is tcp([^[:alnum:]_]|$)" \
        "AWS OFI NCCL selected TCP instead of CXI"
      evidence_summary="backend=NCCL plugin=AWS_OFI provider=cxi transport=slingshot rails=${CXI_RAIL_COUNT} devices=${CXI_DEVICE_NAMES}"
    else
      require_evidence \
        "HCA [0-9]+ '${TCP_NET_DEVICE}(:chn)?'" \
        "AWS OFI NCCL initialized TCP only on ${TCP_NET_DEVICE}"
      reject_evidence "NET/OFI.*Selected Provider is cxi([^[:alnum:]_]|$)" \
        "AWS OFI NCCL selected CXI instead of TCP"
      unexpected_tcp_devices="$({ grep -Eio -- "HCA [0-9]+ '[^']+'" \
        "${RAY_BENCH_LOG}" || true; } | sed -E "s/.*'([^']+)'.*/\\1/" \
        | sort -u | grep -Ev -- "^${TCP_NET_DEVICE}(:chn)?$" || true)"
      if [[ -n "${unexpected_tcp_devices}" ]]; then
        evidence_status=fail
        missing_evidence+=("AWS OFI NCCL initialized TCP outside ${TCP_NET_DEVICE}: ${unexpected_tcp_devices//$'\n'/, }")
      fi
      evidence_summary="backend=NCCL plugin=AWS_OFI provider=tcp transport=tcp device=${TCP_NET_DEVICE}"
    fi
    ;;
  rdt-nixl-*)
    require_evidence \
      "Ray NIXL transport override installed: backend=${NIXL_BACKEND}" \
      "the benchmark installed the Ray override for ${NIXL_BACKEND}"
    require_evidence "Sender transport initialized: manager=NixlTensorTransport" \
      "the sender initialized Ray's NIXL transport"
    require_evidence "Receiver transport initialized: manager=NixlTensorTransport" \
      "the receiver initialized Ray's NIXL transport"
    require_evidence "Created backend:[[:space:]]*${NIXL_BACKEND}" \
      "NIXL created ${NIXL_BACKEND}"
    require_evidence "Backend ${NIXL_BACKEND} was instantiated" \
      "NIXL instantiated ${NIXL_BACKEND}"
    require_evidence "Selected backend:[[:space:]]*${NIXL_BACKEND}" \
      "NIXL selected ${NIXL_BACKEND} for the transfer"

    if (( USES_NIXL_CXI )); then
      require_evidence \
        "Created ${CXI_RAIL_COUNT} rails using provider=cxi" \
        "NIXL created exactly ${CXI_RAIL_COUNT} CXI rail(s)"
      require_evidence \
        "Registered memory on ${CXI_RAIL_COUNT} rails" \
        "NIXL registered the payload on ${CXI_RAIL_COUNT} CXI rail(s)"
      for device_name in "${SELECTED_CXI_DEVICES[@]}"; do
        require_evidence \
          "Created rail [0-9]+.*device=${device_name}.*provider=cxi" \
          "NIXL created a rail on ${device_name}"
      done
      if (( CXI_RAIL_COUNT == 4 )); then
        require_evidence \
          "Striping: submitted 4 requests" \
          "NIXL striped the payload across all four CXI rails"
      else
        reject_evidence \
          "Striping: submitted ([2-9]|[1-9][0-9]+) requests" \
          "NIXL unexpectedly striped the one-rail transfer"
      fi
      reject_evidence \
        "Selected backend:[[:space:]]*UCX|Created rail .*provider=tcp" \
        "the transfer log contains a selected UCX backend or TCP rail"
      evidence_summary="backend=LIBFABRIC provider=cxi transport=slingshot rails=${CXI_RAIL_COUNT} devices=${CXI_DEVICE_NAMES}"
    else
      require_evidence "self/memory" \
        "UCX used self/memory for NIXL's intra-process connection"
      require_evidence \
        "inter-node.*tcp/${UCX_NET_DEVICE}([^[:alnum:]_]|$)" \
        "UCX selected TCP on ${UCX_NET_DEVICE} for the inter-node connection"
      reject_evidence \
        "Selected backend:[[:space:]]*LIBFABRIC|Created rail .*provider=cxi" \
        "the transfer log contains selected LIBFABRIC/CXI evidence"
      unexpected_tcp="$({ grep -Eio -- 'tcp/[[:alnum:]_.-]+' "${RAY_BENCH_LOG}" || true; } \
        | sort -u | grep -Ev -- "^tcp/${UCX_NET_DEVICE}$" || true)"
      if [[ -n "${unexpected_tcp}" ]]; then
        evidence_status=fail
        missing_evidence+=("log contains TCP transport outside ${UCX_NET_DEVICE}: ${unexpected_tcp//$'\n'/, }")
      fi
      evidence_summary="backend=UCX local=self/memory transport=tcp device=${UCX_NET_DEVICE}"
    fi
    ;;
esac

if [[ "${evidence_status}" == "pass" ]]; then
  emit_record "EVIDENCE ${evidence_summary} status=pass"
else
  emit_record "EVIDENCE ${evidence_summary} status=fail"
  printf 'ERROR: %s\n' "${missing_evidence[@]}" >&2
fi

if (( run_status == 0 )); then
  if ! grep -q -- "DRIVER_SHUTDOWN_STATUS=clean" "${RAY_BENCH_LOG}"; then
    emit_record "SHUTDOWN_STATUS=unknown process_status=${run_status}"
    die "Ray exited without reporting clean shutdown"
  fi
  [[ "${evidence_status}" == "pass" ]] \
    || die "transfer or transport evidence validation failed"
  emit_record "SHUTDOWN_STATUS=clean"
  exit 0
fi

if (( run_status == 124 )) \
    && (( USES_NIXL_CXI )) \
    && [[ "${evidence_status}" == "pass" ]] \
    && grep -q -- "Destroying rail manager" "${RAY_BENCH_LOG}"; then
  emit_record "SHUTDOWN_STATUS=timeout phase=libfabric_cxi_teardown"
  echo "Transfer and CXI evidence passed; libfabric/CXI teardown timed out."
  exit 0
fi

if (( run_status == 124 )); then
  emit_record "SHUTDOWN_STATUS=timeout phase=transfer_or_unknown"
else
  emit_record "SHUTDOWN_STATUS=abnormal process_status=${run_status}"
fi
die "Ray benchmark failed (process status ${run_status})"
