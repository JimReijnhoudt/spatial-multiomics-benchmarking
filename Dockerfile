# =============================================================================
# Spatial Multi-Omics Docker Image
# Contains two conda environments:
#   - spatial: SpatialMETA + Scanpy + Squidpy (Python 3.9)
#   - miso:    MISO (Python 3.7)
#
# Base image uses CUDA 12.1 if gpu is available:
#   - spatial: torch==2.2.2+cu121
#   - miso:    torch==1.13.1 (cu11, runs on cu12 host)
#
# Build:
#   docker build -t spatial-multiomics .
#
# Run (with GPU):
#   docker run --gpus all -v $(pwd):/workspace spatial-multiomics
#
# Run (CPU only):
#   docker run -v $(pwd):/workspace spatial-multiomics
#
# =============================================================================

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    git-lfs \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && git lfs install \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Miniconda
# ---------------------------------------------------------------------------
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p $CONDA_DIR && \
    rm /tmp/miniforge.sh && \
    conda update -n base -c conda-forge conda -y && \
    conda clean -afy

# ---------------------------------------------------------------------------
# Matplotlib config: headless server setup (no display server)
# ---------------------------------------------------------------------------
RUN mkdir -p /root/.config/matplotlib && \
    echo "backend: Agg" >> /root/.config/matplotlib/matplotlibrc && \
    echo "font.family: DejaVu Sans" >> /root/.config/matplotlib/matplotlibrc

# ---------------------------------------------------------------------------
# Environment 1: spatial (Python 3.9)
# SpatialMETA + Scanpy + Squidpy
# ---------------------------------------------------------------------------
RUN conda create -n spatial python=3.9 -y

# Clone SpatialMETA
RUN git clone https://github.com/WanluLiuLab/SpatialMETA.git /opt/SpatialMETA

# Install in confirmed working order:
# 1. Pin critical packages first to avoid resolver conflicts
# 2. Install from requirements.txt
# 3. Re-pin after requirements.txt may override
# 4. Install SpatialMETA
RUN conda run -n spatial pip install \
    "numpy==1.24.4" \
    "pandas==1.5.3" \
    "matplotlib==3.5.3" \
    "matplotlib-inline==0.1.6" \
    "xarray==2024.2.0" \
    "dask==2021.10.0" \
    "scanpy==1.9.1" \
    "squidpy==1.2.2"

RUN conda run -n spatial pip install -r /opt/SpatialMETA/docs/requirements.txt

# Re-pin after requirements.txt — these are the confirmed working versions
RUN conda run -n spatial pip install \
    "numpy==1.24.4" \
    "pandas==1.5.3" \
    "matplotlib==3.5.3" \
    "matplotlib-inline==0.1.6" \
    "xarray==2024.2.0" \
    "dask==2021.10.0" \
    "scanpy==1.9.1" \
    "squidpy==1.2.2"

RUN cd /opt/SpatialMETA && conda run -n spatial python setup.py install

# Fix missing zonodo url data directory (bug in SpatialMETA installation)
RUN mkdir -p /opt/conda/envs/spatial/lib/python3.9/site-packages/spatialmeta/data/datasets && \
    cp /opt/SpatialMETA/spatialmeta/data/zenodo_url.txt \
       /opt/conda/envs/spatial/lib/python3.9/site-packages/spatialmeta/data/zenodo_url.txt && \
    echo "SpatialMETA data dirs fixed"

# Register Jupyter kernel
RUN conda run -n spatial pip install ipykernel && \
    conda run -n spatial python -m ipykernel install \
    --name=spatial \
    --display-name "Python (spatial)" \
    --prefix /opt/conda/envs/spatial

# ---------------------------------------------------------------------------
# Environment 2: miso (Python 3.7)
# MISO spatial multi-omics integration
# ---------------------------------------------------------------------------
RUN conda create -n miso python=3.7 -y

# Install confirmed working MISO dependencies
RUN conda run -n miso pip install \
    "numpy==1.21.6" \
    "pandas==1.3.5" \
    "matplotlib==3.5.3" \
    "matplotlib-inline==0.1.6" \
    "torch==1.13.1" \
    "torchvision==0.14.1" \
    "scanpy==1.9.1" \
    "anndata==0.8.0" \
    "scikit-image==0.19.3" \
    "scikit-learn==1.0.2" \
    "scipy==1.7.3" \
    "opencv-python==4.7.0.72" \
    "einops==0.6.0" \
    "tqdm==4.64.1" \
    "numba==0.56.4" \
    "llvmlite==0.39.1" \
    "umap-learn==0.5.7" \
    "seaborn==0.12.2" \
    "natsort==8.4.0" \
    "pynndescent==0.6.0" \
    "session-info==1.0.1" \
    "statsmodels==0.13.5" \
    "ipywidgets==8.1.8" \
    "importlib-metadata==6.7.0" \
    "gdown==4.7.3"

# Python 3.7 importlib.metadata shim
RUN conda run -n miso pip install "importlib_metadata"

# Clone and install MISO
# NOTE: git lfs pull may fail if GitHub LFS budget is exceeded.
# In that case, copy .pth files manually — see README.
RUN git clone https://github.com/kpcoleman/miso.git /opt/miso && \
    cd /opt/miso && \
    git lfs pull || echo "WARNING: git lfs pull failed — copy .pth weights manually" && \
    conda run -n miso pip install -e /opt/miso

# Create checkpoints directory (in case lfs pull failed)
RUN mkdir -p /opt/miso/miso/checkpoints

# Register Jupyter kernel
RUN conda run -n miso pip install ipykernel && \
    conda run -n miso python -m ipykernel install \
    --name=miso \
    --display-name "miso (Python 3.7)" \
    --prefix /opt/conda/envs/miso

# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------
RUN mkdir -p /workspace
WORKDIR /workspace

# Copy smoke test notebooks if present in build context
# COPY env_smoke_test.ipynb /workspace/ 2>/dev/null || true
# COPY miso_smoke_test.ipynb /workspace/ 2>/dev/null || true

# ---------------------------------------------------------------------------
# Entrypoint: interactive shell
# Notebooks are opened and run directly in VS Code using the
# registered kernels (spatial / miso)
# ---------------------------------------------------------------------------
CMD ["/bin/bash"]