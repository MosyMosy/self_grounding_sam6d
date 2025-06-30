#!/bin/bash

set -e  # Exit on error

ENV_NAME="SG-SAM2"
PYTHON_VERSION="3.9"
CUDA_VERSION="11.8"
CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"
PIP_INDEX_URL="https://download.pytorch.org/whl/cu118"

echo "📦 Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
conda create -y -n ${ENV_NAME} python=${PYTHON_VERSION}

echo "🔁 Activating environment and setting up CUDA ${CUDA_VERSION} paths..."
# Use conda run for all commands inside the new env
CONDA_RUN="conda run -n ${ENV_NAME}"

# CUDA env vars to use inside conda run
CUDA_EXPORTS="export CUDA_HOME=${CUDA_HOME} && \
              export PATH=\$CUDA_HOME/bin:\$PATH && \
              export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"

echo "🔥 Installing PyTorch 2.7.1 with CUDA 11.8..."
${CONDA_RUN} bash -c "${CUDA_EXPORTS} && pip install torch==2.7.1 torchvision torchaudio --index-url ${PIP_INDEX_URL}"

echo "📦 Installing PyTorch3D and IOPATH..."
${CONDA_RUN} bash -c "${CUDA_EXPORTS} && pip install -U 'git+https://github.com/facebookresearch/iopath'"
${CONDA_RUN} bash -c "${CUDA_EXPORTS} && pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'"

echo "📦 Installing additional Python packages..."
${CONDA_RUN} bash -c "${CUDA_EXPORTS} && pip install \
    fvcore \
    xformers --index-url ${PIP_INDEX_URL} \
    torchmetrics==0.10.3 \
    opencv-python \
    omegaconf \
    ruamel.yaml \
    hydra-colorlog \
    hydra-core \
    gdown \
    pandas \
    imageio \
    pycocotools \
    distinctipy \
    git+https://github.com/facebookresearch/segment-anything.git \
    ultralytics==8.0.135 \
    numpy==1.26.4 \
    scipy \
    wheel \
    trimesh \
    ptflops"

echo "🛠️ Installing specific pip and pytorch-lightning versions..."
${CONDA_RUN} bash -c "${CUDA_EXPORTS} && pip install pip==23.3.2 && pip install pytorch-lightning==1.8.1 && pip install --upgrade pip"

conda activate ${ENV_NAME}

echo "✅ Environment '${ENV_NAME}' setup complete. You can activate it now:"
echo ""
echo "    conda activate ${ENV_NAME}"
echo ""
