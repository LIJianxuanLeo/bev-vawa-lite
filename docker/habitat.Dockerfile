# Remote-training image: CUDA + habitat-sim + bev-vawa-lite.
#
# Builds a headless EGL-capable container so habitat-sim can render depth on a
# Linux GPU node (AWS g5, Lambda Labs, RunPod, etc.). This image is NOT used
# on the local Apple M4 — local training uses the MuJoCo track.
#
# Build (from repo root):
#   docker build -f docker/habitat.Dockerfile -t bev-vawa-lite:habitat .
#
# Run (with a CUDA GPU visible):
#   docker run --gpus all --rm -it \
#       -v "$PWD":/workspace \
#       -v "$PWD/data":/workspace/data \
#       bev-vawa-lite:habitat \
#       bash docker/train_remote.sh --tiny
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH

# --- system deps (including EGL + OpenGL headers habitat-sim needs) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates \
    build-essential cmake ninja-build pkg-config \
    libglvnd-dev libgl1 libglx0 libegl1 libgles2 libopengl0 \
    libglfw3-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
    libjpeg-turbo8-dev libpng-dev libtiff-dev \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# --- miniforge (conda-forge defaults, smaller than full anaconda) ---
RUN curl -L -o /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p $CONDA_DIR \
    && rm /tmp/miniforge.sh \
    && conda clean -afy

# --- habitat-sim (headless EGL build, from the aihabitat channel) ---
# Pin to a known-good version; upgrade intentionally when the project bumps.
RUN conda install -y -n base -c conda-forge -c aihabitat \
        python=3.10 \
        habitat-sim=0.3.2 \
        withbullet=1 \
        headless=1 \
    && conda clean -afy

# --- project dependencies (mirror pyproject.toml [project]) ---
RUN pip install --upgrade pip \
    && pip install \
        "torch>=2.4" \
        "numpy>=1.26" \
        "pyyaml>=6.0" \
        "tqdm>=4.65" \
        "imageio>=2.30" \
        "imageio-ffmpeg>=0.4" \
        "matplotlib>=3.7" \
        "pytest>=7"

WORKDIR /workspace
COPY . /workspace
RUN pip install -e . --no-deps

# Non-interactive healthcheck: the image can import both torch (with CUDA)
# and habitat-sim, and can import our package.
RUN python -c "import torch, habitat_sim, bev_vawa; print('torch.cuda =', torch.cuda.is_available()); print('habitat-sim =', habitat_sim.__version__)"

CMD ["bash"]
