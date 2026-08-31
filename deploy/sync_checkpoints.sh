#!/usr/bin/env bash
# Mirror checkpoints and results to S3 every 5 minutes.
# Spot instances lose the root volume on interruption, so a checkpoint that only
# exists on the instance is not a checkpoint. Run this alongside training:
#   BUCKET=my-bucket nohup bash deploy/sync_checkpoints.sh &
set -euo pipefail
BUCKET="${BUCKET:?set BUCKET}"; PREFIX="${PREFIX:-oceanembed}"
while true; do
    aws s3 sync checkpoints "s3://$BUCKET/$PREFIX/checkpoints" --quiet
    aws s3 sync results     "s3://$BUCKET/$PREFIX/results"     --quiet
    sleep 300
done
