#!/usr/bin/env python3
"""Container-side dependency checks required before NIXL benchmarks."""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path

import nixl


def require_cuda13_package() -> None:
    selected = getattr(getattr(nixl, "_pkg", None), "__name__", "unknown")
    if selected != "nixl_cu13":
        raise RuntimeError(
            f"NIXL selected {selected!r}; expected the nixl_cu13 implementation"
        )
    print("NIXL_PACKAGE_PREFLIGHT=pass package=nixl_cu13", flush=True)


def find_libfabric_plugin() -> Path:
    implementation_dir = Path(nixl._pkg.__file__).resolve().parent
    site_packages = implementation_dir.parent
    implementation_name = nixl._pkg.__name__
    candidates = sorted(
        site_packages.glob(
            f".{implementation_name}.mesonpy.libs/plugins/"
            "libplugin_LIBFABRIC.so"
        )
    )
    candidates.extend(
        sorted(
            site_packages.glob(
                f"{implementation_name}.libs/nixl/libplugin_LIBFABRIC.so"
            )
        )
    )
    if not candidates:
        raise RuntimeError(
            "could not find libplugin_LIBFABRIC.so next to "
            f"{implementation_dir}"
        )
    return candidates[0]


def preflight_libfabric() -> None:
    require_cuda13_package()
    plugin = find_libfabric_plugin()
    mode = getattr(os, "RTLD_NOW", 2) | getattr(os, "RTLD_LOCAL", 0)
    try:
        ctypes.CDLL(str(plugin), mode=mode)
    except OSError as exc:
        raise RuntimeError(
            f"NIXL LIBFABRIC plugin dependency load failed: {plugin}: {exc}"
        ) from exc
    print(f"LIBFABRIC_PLUGIN_DLOPEN=pass path={plugin}", flush=True)


def mapped_ucx_libraries() -> list[str]:
    libraries: set[str] = set()
    with open("/proc/self/maps", encoding="utf-8") as maps_file:
        for line in maps_file:
            fields = line.split()
            if not fields or not fields[-1].startswith("/"):
                continue
            path = fields[-1]
            if path.rsplit("/", 1)[-1].startswith(
                ("libucp", "libuct", "libucs", "libucm")
            ):
                libraries.add(path)
    return sorted(libraries)


def preflight_ucx() -> None:
    require_cuda13_package()
    from nixl import nixl_agent, nixl_agent_config

    agent = nixl_agent("ucx_preflight", nixl_agent_config(backends=[]))
    parameter = "ucx_error_handling_mode"
    available_params = agent.get_plugin_params("UCX")
    if parameter not in available_params:
        raise RuntimeError(
            f"NIXL UCX plugin does not expose {parameter!r}: {available_params}"
        )
    agent.create_backend("UCX", {parameter: "none"})
    print(
        "UCX_BACKEND_PREFLIGHT=pass "
        "ucx_error_handling_mode=none local_transport=self/memory",
        flush=True,
    )

    libraries = mapped_ucx_libraries()
    for path in libraries:
        print(f"UCX_MAPPED_LIBRARY={path}", flush=True)
    roots = {path.rsplit("/", 1)[0] for path in libraries}
    print(f"UCX_MAPPED_LIBRARY_ROOTS={len(roots)}", flush=True)

    if not libraries:
        raise RuntimeError("no mapped UCX libraries were found")
    if any(path.startswith("/opt/hpcx/ucx/") for path in libraries):
        raise RuntimeError("HPC-X and NIXL UCX libraries are both mapped")
    if len(roots) != 1:
        raise RuntimeError(f"expected one mapped UCX library root; found {roots}")
    print("UCX_LIBRARY_ISOLATION=pass provider=nixl_cu13", flush=True)

    # Avoid invoking UCX process-global teardown from this isolated probe.
    os._exit(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("ucx", "libfabric"))
    return parser.parse_args()


def main() -> None:
    backend = parse_args().backend
    if backend == "ucx":
        preflight_ucx()
    else:
        preflight_libfabric()


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
