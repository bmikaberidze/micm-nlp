# nvidia-smi
# nvidia-container-toolkit --version
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Create working directory
ENV WORKDIR=/app
WORKDIR ${WORKDIR}

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
