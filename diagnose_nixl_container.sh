#!/usr/bin/env bash
# Inspect which libfabric NIXL resolves inside the benchmark container.

set -euo pipefail

die() {
  echo "ERROR: $*" >&2
  exit 2
}

if [[ "${1:-}" == "--inside-container" ]]; then
  variant="${2:?missing probe variant}"
  host="${SLURMD_NODENAME:-$(hostname)}"
  echo "PROBE host=${host} variant=${variant}"
  echo "ENV host=${host} LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"
  echo "ENV host=${host} LD_PRELOAD=${LD_PRELOAD:-unset}"

  echo "LIBRARIES host=${host} source=injected-cray"
  ls -l /opt/udiImage/modules/nccl-plugin/deps/lib/libfabric.so* 2>&1 || true
  echo "LIBRARIES host=${host} source=image-amazon-efa"
  ls -l /opt/amazon/efa/lib/libfabric.so* 2>&1 || true

  plugin="/usr/local/lib/python3.12/dist-packages/.nixl_cu13.mesonpy.libs/plugins/libplugin_LIBFABRIC.so"
  [[ -f "${plugin}" ]] || die "NIXL cu13 LIBFABRIC plugin was not found"
  echo "PLUGIN host=${host} path=${plugin}"
  if command -v readelf >/dev/null 2>&1; then
    readelf -d "${plugin}" \
      | grep -E 'NEEDED|RPATH|RUNPATH' \
      | sed "s/^/ELF host=${host} /" || true
  fi
  ldd "${plugin}" | grep -E 'fabric|cxi|cuda' \
    | sed "s/^/LDD host=${host} /" || true

  if command -v fi_info >/dev/null 2>&1; then
    set +e
    fi_info -p cxi >/dev/null 2>&1
    fi_status=$?
    set -e
    echo "FI_INFO host=${host} provider=cxi status=${fi_status}"
  else
    echo "FI_INFO host=${host} provider=cxi status=unavailable"
  fi

  set +e
  python - <<'PY'
import os

import nixl
from nixl._api import nixl_agent, nixl_agent_config


def mapped_libfabric() -> list[str]:
    paths = set()
    with open("/proc/self/maps", encoding="utf-8") as maps_file:
        for line in maps_file:
            fields = line.rstrip().split(maxsplit=5)
            if len(fields) == 6 and "/libfabric.so" in fields[5]:
                paths.add(fields[5].removesuffix(" (deleted)"))
    return sorted(paths)


host = os.environ.get("SLURMD_NODENAME", "unknown")
status = 0
try:
    selected = getattr(nixl._pkg, "__name__", "unknown")
    if selected != "nixl_cu13":
        raise RuntimeError(f"selected {selected}, expected nixl_cu13")
    agent = nixl_agent("container_probe", nixl_agent_config(backends=[]))
    agent.create_backend("LIBFABRIC", {})
except Exception as error:
    status = 1
    print(
        f"NIXL_BACKEND host={host} backend=LIBFABRIC status=fail "
        f"error={type(error).__name__}:{error}",
        flush=True,
    )
else:
    print(
        f"NIXL_BACKEND host={host} backend=LIBFABRIC status=pass",
        flush=True,
    )
finally:
    paths = mapped_libfabric()
    print(
        f"NIXL_LIBFABRIC host={host} mapped="
        f"{','.join(paths) if paths else 'none'}",
        flush=True,
    )
    # Avoid exercising NIXL/libfabric teardown in this selection probe.
    os._exit(status)
PY
  backend_status=$?
  set -e
  exit "${backend_status}"
fi

IMAGE=""
while (( $# )); do
  case "$1" in
    --image)
      (( $# >= 2 )) || die "--image requires a value"
      IMAGE="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 --image IMAGE"
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -n "${IMAGE}" ]] || die "--image is required"
[[ -n "${SLURM_JOB_ID:-}" ]] || die "run inside a Slurm allocation"
[[ -n "${SLURM_JOB_NODELIST:-}" ]] || die "SLURM_JOB_NODELIST is not set"
for command_name in scontrol srun podman-hpc; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || die "required command not found: ${command_name}"
done

mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
(( ${#nodes[@]} == 2 )) \
  || die "the diagnostic requires exactly two nodes; found ${#nodes[@]}"

workdir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cpus_per_task="${SLURM_CPUS_PER_TASK:-128}"
srun_args=(
  --nodes=2
  --ntasks=2
  --ntasks-per-node=1
  --cpus-per-task="${cpus_per_task}"
  --gpus-per-task=4
  --gpu-bind=none
)

echo "IMAGE tag=${IMAGE}"
set +e
srun "${srun_args[@]}" podman-hpc image inspect \
  --format 'IMAGE host={{.Id}} created={{.Created}}' "${IMAGE}"
set -e

run_probe() {
  local variant="$1"
  local -a podman_args=(
    run
    --rm
    --net host
    --gpu
    --nccl-cu13
    --device=/dev/cxi3
    --device=/dev/cxi_sbl
    -v "${workdir}:/workdir"
    -w /workdir
    --env SLURMD_NODENAME
    --env "CUDA_VISIBLE_DEVICES="
    --env "FI_PROVIDER=cxi"
    --env "FI_CXI_DEVICE_NAME=cxi3"
    --env "FI_LOG_LEVEL=warn"
    --env "NIXL_LOG_LEVEL=WARN"
  )
  if [[ "${variant}" == "cray-preload" ]]; then
    podman_args+=(
      --env "LD_PRELOAD=/opt/udiImage/modules/nccl-plugin/deps/lib/libfabric.so.1"
    )
  fi

  echo "RUN probe=${variant} nodes=2"
  set +e
  srun "${srun_args[@]}" \
    podman-hpc "${podman_args[@]}" "${IMAGE}" \
    bash /workdir/diagnose_nixl_container.sh --inside-container "${variant}"
  local status=$?
  set -e
  echo "RESULT probe=${variant} status=${status}"
  return "${status}"
}

default_status=0
preload_status=0
run_probe default || default_status=$?
run_probe cray-preload || preload_status=$?

echo "SUMMARY default_status=${default_status} cray_preload_status=${preload_status}"
if (( default_status != 0 && preload_status == 0 )); then
  echo "DIAGNOSIS selection=cray-preload-resolves-cxi"
elif (( default_status == 0 )); then
  echo "DIAGNOSIS selection=default-already-works"
else
  echo "DIAGNOSIS selection=preload-insufficient"
fi
