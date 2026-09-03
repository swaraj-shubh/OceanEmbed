#!/usr/bin/env bash
# Bootstrap an EC2 GPU instance for OceanEmbed training.
#   BUCKET=my-bucket bash deploy/setup.sh
# Assumes the Deep Learning AMI (PyTorch), which already has CUDA + torch.
set -euo pipefail

BUCKET="${BUCKET:?set BUCKET to your S3 bucket name}"
PREFIX="${PREFIX:-oceanembed}"
REPO="${REPO:-https://github.com/swaraj-shubh/OceanEmbed.git}"

cd "$HOME"
[ -d OceanEmbed ] || git clone "$REPO"
cd OceanEmbed

# The DLAMI ships torch; only the data stack is missing. No torch here on purpose --
# reinstalling it from PyPI would replace the CUDA build with a CPU one.
pip install -q xarray zarr dask netCDF4 pyyaml pandas scipy

# The store is the only thing the model needs: 3.1 GB, versus 45 GB of raw downloads.
mkdir -p data/processed checkpoints results
aws s3 sync "s3://$BUCKET/$PREFIX/processed/nio_daily.zarr" data/processed/nio_daily.zarr
aws s3 cp  "s3://$BUCKET/$PREFIX/processed/norm_stats.json" data/processed/norm_stats.json

python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU ONLY")
PY
echo "ready:  python src/train.py configs/m2_unet.yaml"

# The sync loop is not optional. Day 2 lost 18 checkpoints because it was a separate
# manual command in a second shell that nobody ran. Starting it here means the only way
# to train without a sync is to not use this script.
pkill -f sync_checkpoints.sh || true
BUCKET="$BUCKET" PREFIX="$PREFIX" nohup bash deploy/sync_checkpoints.sh > sync.log 2>&1 &
echo "checkpoint sync -> s3://$BUCKET/$PREFIX/checkpoints (pid $!)"
