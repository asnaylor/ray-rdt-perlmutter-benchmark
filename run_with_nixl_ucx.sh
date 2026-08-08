#!/usr/bin/env bash

# Make dependencies on conventional UCX SONAMEs resolve to the UCX libraries
# bundled in the nixl_cu13 wheel. The NVIDIA PyTorch base image otherwise loads
# HPC-X UCX first, leaving two incompatible UCX stacks in one Python process.

set -euo pipefail

(( $# > 0 )) || {
  echo "ERROR: run_with_nixl_ucx.sh requires a command" >&2
  exit 2
}

nixl_ucx_source=/usr/local/lib/python3.12/dist-packages/nixl_cu13.libs
nixl_ucx_compat=/tmp/nixl-ucx-compat

[[ -d "${nixl_ucx_source}" ]] || {
  echo "ERROR: NIXL UCX library directory is absent: ${nixl_ucx_source}" >&2
  exit 2
}

mkdir -p -- "${nixl_ucx_compat}"

for component in ucs ucm uct ucp; do
  shopt -s nullglob
  matches=("${nixl_ucx_source}/lib${component}-"*.so.0.0.0)
  shopt -u nullglob
  (( ${#matches[@]} == 1 )) || {
    echo "ERROR: expected one bundled lib${component}, found ${#matches[@]}" >&2
    exit 2
  }
  ln -sfn -- "${matches[0]}" "${nixl_ucx_compat}/lib${component}.so.0"
done

# The compatibility directory satisfies dependencies on conventional UCX
# SONAMEs (for example libucp.so.0). The wheel directory is also required
# because those libraries depend on the wheel's hashed SONAMEs (for example
# libucm-258590e9.so.0.0.0).
export LD_LIBRARY_PATH="${nixl_ucx_compat}:${nixl_ucx_source}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export UCX_VFS_ENABLE="${UCX_VFS_ENABLE:-n}"

exec "$@"
