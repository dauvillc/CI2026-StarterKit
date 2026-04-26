#!/bin/bash

# Brev setup script for the CI 2026 Hackathon Starter Kit.
# Runs automatically on first boot before participants access the instance.
# Requires NVIDIA driver >= 570 (ships with Brev CUDA 12.8 base images).
set -euo pipefail

# ── 0. System dependencies ────────────────────────────────────────────────────
echo "[setup] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq unzip

REPO_DIR="/home/ubuntu/workspace"
DATA_DIR="/home/ubuntu/workspace/data"
ENV_NAME="ci26_starter_kit"

# ── 1. Conda ──────────────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null && [ ! -x "$HOME/miniconda3/bin/conda" ]; then
    echo "[setup] Installing Miniconda..."
    curl -fsSL \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
    "$HOME/miniconda3/bin/conda" init bash
fi
CONDA_BIN=$(command -v conda 2>/dev/null || echo "$HOME/miniconda3/bin/conda")
eval "$("$CONDA_BIN" shell.bash hook)"
# Pin to conda-forge only
cat > "$HOME/.condarc" << 'EOF'
channels:
  - conda-forge
channel_priority: strict
EOF
# Accept ToS non-interactively for conda-forge (required since conda 25.x)
mkdir -p "$HOME/.cache/conda-anaconda-tos"
sudo chown -R "$USER":"$USER" "$HOME/.cache/conda-anaconda-tos" 2>/dev/null || true
for _ch in \
    https://conda.anaconda.org/conda-forge \
    https://repo.anaconda.com/pkgs/main \
    https://repo.anaconda.com/pkgs/r; do
    "$CONDA_BIN" tos accept --override-channels --channel "$_ch" 2>/dev/null || true
done

# ── 2. Conda environment ──────────────────────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[setup] Conda env '${ENV_NAME}' already exists, skipping creation."
else
    echo "[setup] Creating conda env '${ENV_NAME}'..."
    conda env create -f "$REPO_DIR/environment.yml"
fi
conda activate "$ENV_NAME"
ENV_PYTHON=$("$CONDA_BIN" run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')

# ── 3. PyTorch with CUDA 12.8 ─────────────────────────────────────────────────
echo "[setup] Installing PyTorch (CUDA 12.8)..."
conda run -n "$ENV_NAME" uv pip install --python "$ENV_PYTHON" \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# ── 4. Remaining requirements (torch already installed above) ─────────────────
echo "[setup] Installing requirements..."
_req_tmp=$(mktemp)
grep -vE "^torch(vision|audio)?($|[>=<! ])" "$REPO_DIR/requirements.txt" > "$_req_tmp"
conda run -n "$ENV_NAME" uv pip install --python "$ENV_PYTHON" -r "$_req_tmp"
rm -f "$_req_tmp"

# ── 5. Install starter kit in editable mode ───────────────────────────────────
echo "[setup] Installing starter kit package..."
conda run -n "$ENV_NAME" pip install -e "$REPO_DIR/"

# ── 6. Training data from HuggingFace ─────────────────────────────────────────
echo "[setup] Downloading training data..."
mkdir -p "$DATA_DIR"
conda run -n "$ENV_NAME" hf download tobifinn/CI2026Hackathon \
    --repo-type dataset \
    --local-dir "$DATA_DIR/train_data"
find "$DATA_DIR/train_data" -name "*.zip" -exec unzip -o {} -d "$DATA_DIR/train_data" \; \
    -exec rm {} \;

echo "[setup] Done. Activate the environment with: conda activate ${ENV_NAME}"
