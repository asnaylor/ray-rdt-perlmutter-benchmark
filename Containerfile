FROM nersc/pytorch:26.01.01

SHELL ["/bin/bash", "-lc"]

ARG CUPY_VERSION=14.1.1

RUN python -m pip install --no-cache-dir \
      "cupy-cuda13x==${CUPY_VERSION}" \
 && python -c \
      'import cupy; print("Installed CuPy:", cupy.__version__)'
