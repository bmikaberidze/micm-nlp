# nvidia-smi
# nvidia-container-toolkit --version
#
# A plain Python base, not a CUDA one.
#
# A CUDA setup has four layers, and only some can live in an image. The kernel
# module and the driver userspace (libcuda.so, nvidia-smi) must match the host and
# are injected at `docker run` by the NVIDIA Container Toolkit -- no image has ever
# been able to ship them. The CUDA runtime and math libraries can come from a base
# image *or* from pip, and the toolkit (nvcc) is only needed to compile CUDA code.
#
# `pip install -e .` below resolves torch from PyPI, and the Linux torch wheels
# vendor their own runtime -- cuBLAS, cuDNN, NCCL, cuSPARSE -- as `nvidia-*` wheels
# pinned to what torch was built against. That is where roughly 4 GB of this image
# still goes: CUDA was not removed, a second unused copy of it was.
#
# The old base, pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel, supplied that second
# copy and torch never touched it. Reading /proc/self/maps inside that image after
# forcing a cuBLAS and a cuDNN call shows every CUDA library loaded from
# site-packages/nvidia/, none from /usr/local/cuda-11.7/, and torch.version.cuda
# reporting 13.0 inside a CUDA-11.7 base.
FROM python:3.12-slim

# Create working directory
ENV WORKDIR=/app
WORKDIR ${WORKDIR}

# Wheels are downloaded once and installed once; keeping pip's copy would bake
# several GB of duplicate archives into the layer below.
ENV PIP_NO_CACHE_DIR=1

# Install system packages (git needed for GitPython; strace for debugging)
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        git \
        strace \
        apt-transport-https \
        ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install package
RUN pip install --upgrade pip
# pyproject.toml declares `readme = "README.md"`, so the build backend reads it
# while generating metadata -- without this COPY, `pip install -e .` below fails
# with `OSError: Readme file does not exist: README.md`.
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install -e ".[dev]"

# Check if CUDA is available:
CMD [ "/bin/bash", "-c",  "python -c \"import torch; print(f'CUDA is available: {torch.cuda.is_available()}')\"" ]
