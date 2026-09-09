#!/usr/bin/env bash

set -Eeuo pipefail

# FOCUS uses LMDeploy's PyTorch backend and CUDA/Triton kernels.  Run this
# script from the repository root after cloning the repository.
#
# Optional overrides:
#   ENV_NAME=focus PYTHON_VERSION=3.13 bash setup_focus_env.sh
#   INSTALL_TEST_DEPS=1 bash setup_focus_env.sh

ENV_NAME=${ENV_NAME:-focus}
PYTHON_VERSION=${PYTHON_VERSION:-3.13}
INSTALL_TEST_DEPS=${INSTALL_TEST_DEPS:-0}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

if [[ ! -f setup.py || ! -f requirements/runtime_cuda.txt ]]; then
    echo "Error: this script must be located in the FOCUS repository root." >&2
    exit 1
fi

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Error: FOCUS's CUDA/Triton execution path requires a Linux server." >&2
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda was not found. Install Miniconda/Anaconda first." >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Warning: nvidia-smi was not found. Installation can continue, but" >&2
    echo "         running FOCUS requires an NVIDIA GPU and a working driver." >&2
else
    echo "Detected NVIDIA driver/GPU:"
    nvidia-smi --query-gpu=name,driver_version,memory.total \
        --format=csv,noheader || true
fi

# Load conda activation support in this non-interactive shell.
CONDA_BASE=$(conda info --base)
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk -v name="${ENV_NAME}" '$1 == name { found=1 } END { exit !found }'; then
    echo "Using existing conda environment: ${ENV_NAME}"
else
    echo "Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
    conda create --name "${ENV_NAME}" "python=${PYTHON_VERSION}" pip -y
fi

conda activate "${ENV_NAME}"

echo "Installing FOCUS runtime dependencies..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements/runtime_cuda.txt

if [[ "${INSTALL_TEST_DEPS}" == "1" ]]; then
    echo "Installing test dependencies..."
    python -m pip install -r requirements/test.txt
fi

# FOCUS uses the extensible PyTorch engine.  Disabling TurboMind avoids an
# unnecessary local C++/CUDA build and follows this repository's README.
echo "Installing this repository in editable mode..."
DISABLE_TURBOMIND=1 python -m pip install -e .

echo "Verifying the installation..."
python - <<'PY'
import platform

import lmdeploy
import torch
import transformers
import triton

print(f"Python:       {platform.python_version()}")
print(f"LMDeploy:     {lmdeploy.__version__}")
print(f"PyTorch:      {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"Triton:       {triton.__version__}")
print(f"CUDA build:   {torch.version.cuda}")
print(f"CUDA usable:  {torch.cuda.is_available()}")
print(f"Visible GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        gib = props.total_memory / 1024**3
        print(f"  GPU {index}: {props.name} ({gib:.1f} GiB)")
else:
    print("WARNING: PyTorch cannot currently access CUDA. Check the NVIDIA driver,")
    print("         scheduler GPU allocation, and CUDA_VISIBLE_DEVICES before running.")
PY

echo
echo "FOCUS environment setup completed."
echo "Activate it in a new shell with: conda activate ${ENV_NAME}"
echo "Run commands from repository root: ${SCRIPT_DIR}"
echo "For the two-GPU LLaDA2 MoE experiment, verify that GPUs 0 and 1 are visible."
