#!/usr/bin/env bash
# Validate five completed transport logs and publish the canonical artifacts.

set -euo pipefail

die() {
  echo "ERROR: $*" >&2
  exit 2
}

IMAGE=""
while (( $# )); do
  case "$1" in
    --image)
      (( $# >= 2 )) || die "--image requires a value"
      IMAGE="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 --image IMAGE OBJECT_LOG NIXL_SINGLE_RAIL_LOG NIXL_FOUR_RAIL_LOG NCCL_SWEEP_LOG NCCL_SCALING_LOG"
      exit 0
      ;;
    --*)
      die "unknown argument: $1"
      ;;
    *)
      break
      ;;
  esac
done

[[ -n "${IMAGE}" ]] || die "--image is required"
(( $# == 5 )) || die "provide exactly five transport logs"
command -v podman-hpc >/dev/null 2>&1 || die "required command not found: podman-hpc"

WORKDIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
container_logs=()
for log in "$@"; do
  [[ -f "${log}" ]] || die "log does not exist: ${log}"
  absolute_log="$(cd -- "$(dirname -- "${log}")" && pwd -P)/$(basename -- "${log}")"
  case "${absolute_log}" in
    "${WORKDIR}"/*) ;;
    *) die "log must be beneath ${WORKDIR}: ${log}" ;;
  esac
  container_logs+=("/workdir/${absolute_log#"${WORKDIR}"/}")
done

podman-hpc run --rm -v "${WORKDIR}:/workdir" -w /workdir "${IMAGE}" \
  python -u /workdir/plot_benchmark_results.py publish \
  --csv /workdir/results/benchmark-results.csv \
  --headline-svg /workdir/docs/headline-throughput.svg \
  --baseline-svg /workdir/docs/baseline-throughput.svg \
  --flows-svg /workdir/docs/single-nic-flow-scaling.svg \
  --flows-latency-svg /workdir/docs/single-nic-latency.svg \
  --nics-svg /workdir/docs/multi-nic-scaling.svg \
  --nics-latency-svg /workdir/docs/multi-nic-latency.svg \
  "${container_logs[@]}"
