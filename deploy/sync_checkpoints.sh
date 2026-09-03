#!/usr/bin/env bash
# Mirror checkpoints, results AND code to S3 every 5 minutes.
# Spot instances lose the root volume on interruption, so a checkpoint that only
# exists on the instance is not a checkpoint. Run this alongside training:
#   BUCKET=my-bucket nohup bash deploy/sync_checkpoints.sh &
#
# src/ and configs/ are synced too, not just data: docs/10 sec.9 found a whole day of
# work (bias_correct.py, ablation.py, audit_leakage.py, 7 configs) that existed only on
# a running instance, in a directory that was never a git clone -- the exact bug docs/08
# already recorded once. Losing the instance would have lost the code, not just the run.
set -euo pipefail
BUCKET="${BUCKET:?set BUCKET}"; PREFIX="${PREFIX:-oceanembed}"
while true; do
    aws s3 sync checkpoints "s3://$BUCKET/$PREFIX/checkpoints" --quiet
    aws s3 sync results     "s3://$BUCKET/$PREFIX/results"     --quiet
    aws s3 sync src         "s3://$BUCKET/$PREFIX/src"         --quiet --exclude "*__pycache__*"
    aws s3 sync configs     "s3://$BUCKET/$PREFIX/configs"     --quiet
    sleep 300
done
