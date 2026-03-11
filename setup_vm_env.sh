#!/bin/bash
# VM environment setup for RLix (assumes CUDA drivers already installed).
# Creates conda env "rlix" with Python 3.10 + CUDA toolkit 12.4,
# then installs all Python dependencies.
# Use -eo but not -u because conda's shell init references unset variables like PS1
set -eo pipefail

CONDA_ENV_NAME="rlix"
PYTHON_VERSION="3.10"
CUDA_TOOLKIT_VERSION="12.4"

touch ~/.no_auto_tmux

# --- Git config + SSH ---
if ! grep -q 'GIT_SSH_COMMAND' ~/.bashrc 2>/dev/null; then
  echo 'export GIT_SSH_COMMAND="ssh -i /workspace/.ssh/id_ed25519 -o IdentitiesOnly=yes"' >> ~/.bashrc
fi
export GIT_SSH_COMMAND="ssh -i /workspace/.ssh/id_ed25519 -o IdentitiesOnly=yes"

git config --global user.name "Tao Luo"
git config --global user.email "taoluo321@outlook.com"

# --- Install Miniconda if conda is not available ---
if ! command -v conda &> /dev/null; then
  if [[ -d "$HOME/miniconda3" ]]; then
    # Miniconda dir exists but conda not on PATH, just init it
    echo "Found existing Miniconda install, initializing..."
  else
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
  fi
  # Initialize conda for future bash sessions
  "$HOME/miniconda3/bin/conda" init bash
fi
# Source conda into current shell
# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# Accept conda default channel TOS (required for non-interactive use)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# --- Create conda env with Python 3.10 and CUDA toolkit 12.4 ---
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
  echo "Conda env '${CONDA_ENV_NAME}' already exists, skipping creation."
else
  echo "Creating conda env '${CONDA_ENV_NAME}' with Python ${PYTHON_VERSION} and CUDA toolkit ${CUDA_TOOLKIT_VERSION}..."
  # Use versioned nvidia channel for reproducible CUDA installs, include cudnn
  conda create -n "${CONDA_ENV_NAME}" python="${PYTHON_VERSION}" \
    "cuda=${CUDA_TOOLKIT_VERSION}" "cudnn>=9.1.0" \
    -c "nvidia/label/cuda-${CUDA_TOOLKIT_VERSION}.0" -c nvidia -y
fi

# Activate the conda env
conda activate "${CONDA_ENV_NAME}"

# Set CUDA_HOME so build tools (e.g. transformer-engine, apex) find the toolkit
conda env config vars set CUDA_HOME="$CONDA_PREFIX"
# Re-activate to pick up the new env var
conda activate "${CONDA_ENV_NAME}"

echo "Active env: $(conda info --envs | grep '*')"
echo "Python: $(python --version)"
echo "nvcc: $(nvcc --version | tail -1)"
echo "CUDA_HOME: ${CUDA_HOME}"

# --- Install uv if not available ---
if ! command -v uv &> /dev/null; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# --- Install Python dependencies ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/external/ROLL_rlix"

uv pip install -r requirements_torch260_vllm.txt
uv pip install --no-build-isolation "transformer-engine[pytorch]==2.2.0"

# Install ROLL and RLix in editable mode so 'roll' and 'rlix' are importable
uv pip install -e "${SCRIPT_DIR}/external/ROLL_rlix"
uv pip install -e "${SCRIPT_DIR}"

# --- System packages for tracing ---
# Wait for any running apt/dpkg locks (e.g. unattended-upgrades) before installing
while fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do echo "Waiting for dpkg lock..."; sleep 5; done
apt-get update && apt-get install -y protobuf-compiler libprotobuf-dev nvtop
uv pip install "protobuf<3.21.0" "tg4perfetto>=0.0.6"

# --- Dev tools ---
curl -fsSL https://claude.ai/install.sh | bash
if ! grep -q 'HOME/.local/bin' ~/.bashrc 2>/dev/null; then
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi
# Install iflow first (it sets up node/npm via nvm)
bash -c "$(curl -fsSL https://gist.githubusercontent.com/taoluo/d5ada7e9210c34e4108988bf1b34681d/raw/9e155e480db1d1efea37975b8f47b2b865a27cc0/iflow_cli_install.sh)"
# Source nvm so npm is available for codex install
export NVM_DIR="$HOME/.nvm"
# shellcheck disable=SC1091
source "$NVM_DIR/nvm.sh"
npm i -g @openai/codex

echo "Done! Run 'conda activate ${CONDA_ENV_NAME}' to use the environment."
