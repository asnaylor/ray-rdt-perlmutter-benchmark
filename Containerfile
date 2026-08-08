FROM nersc/pytorch:26.01.01

SHELL ["/bin/bash", "-lc"]

ARG CUPY_VERSION=14.1.1
ARG NIXL_VERSION=1.3.2

RUN python -m pip install --no-cache-dir \
      "cupy-cuda13x==${CUPY_VERSION}" \
      "nixl==${NIXL_VERSION}" \
 && python -c \
      'import importlib.metadata as m; import cupy, nixl, torch; assert nixl._pkg.__name__ == "nixl_cu13", nixl._pkg.__name__; print("Installed CuPy:", cupy.__version__); print("Installed NIXL:", m.version("nixl"), "using", nixl._pkg.__name__); print("Torch CUDA:", torch.version.cuda)'
