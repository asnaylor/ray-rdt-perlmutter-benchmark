#!/usr/bin/env bash
# Run the last committed NIXL/CXI benchmark as a regression control.

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
[[ -n "${SCRATCH:-}" && -d "${SCRATCH}" ]] \
  || die "SCRATCH must name a shared directory"
for command_name in git tar mktemp; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || die "required command not found: ${command_name}"
done

control_commit=8272484
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
git -C "${repo}" cat-file -e "${control_commit}^{commit}" \
  || die "control commit ${control_commit} is unavailable"

control_dir="$(mktemp -d "${SCRATCH}/ray-nixl-control.XXXXXX")"
cleanup() {
  [[ -n "${control_dir:-}" && -d "${control_dir}" ]] \
    && rm -rf -- "${control_dir}"
}
trap cleanup EXIT

git -C "${repo}" archive "${control_commit}" \
  | tar -xf - -C "${control_dir}"

stamp="${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)"
log_dir="${repo}/logs/original-control-${stamp}"
mkdir -p -- "${log_dir}"
log="${log_dir}/nixl-cxi-1mib.log"

echo "RUN control=original-nixl commit=${control_commit} image=${IMAGE} log=${log}"
set +e
(
  cd -- "${control_dir}"
  IMAGE="${IMAGE}" \
  BENCH=rdt-nixl-cxi-cpu \
  BENCH_ARGS="--size-mb 1 --warmup 1 --iterations 1" \
  RAY_BENCH_LOG="${log}" \
    ./run_ray_symmetric_bench_interactive.sh
)
status=$?
set -e

echo "RESULT control=original-nixl status=${status} log=${log}"
exit "${status}"
