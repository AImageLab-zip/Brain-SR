#!/bin/bash

cd /homes/gcasari/bigbrain/utils/

set -euo pipefail

# Hardcoded lists (one-to-one mapping)
EXPS=(
  "/homes/gcasari/bigbrain/work_data/out_report/fft_nol2_small_folds/"
  "/homes/gcasari/bigbrain/work_data/out_report/fft_nol2_folds/"
  "/homes/gcasari/bigbrain/work_data/out_report/fft_nol2_big_folds/"
  "/homes/gcasari/bigbrain/work_data/out_report/fft_bridge16_folds/"
  "/homes/gcasari/bigbrain/work_data/out_report/lpip2_2e5_folds/"
  "/homes/gcasari/bigbrain/work_data/out_report/mid_disc_folds/"
)

OUTS=(
  "/homes/gcasari/bigbrain/work_data/out_report/fft_nol2_small_folds/fold_metrics.csv"
  "/homes/gcasari/bigbrain/work_data/out_report/fft_nol2_folds/fold_metrics.csv"
  "/homes/gcasari/bigbrain/work_data/out_report/fft_nol2_big_folds/fold_metrics.csv"
  "/homes/gcasari/bigbrain/work_data/out_report/fft_bridge16_folds/fold_metrics.csv"
  "/homes/gcasari/bigbrain/work_data/out_report/lpip2_2e5_folds/fold_metrics.csv"
  "/homes/gcasari/bigbrain/work_data/out_report/mid_disc_folds/fold_metrics.csv"
)

if [ "${#EXPS[@]}" -ne "${#OUTS[@]}" ]; then
  echo "Error: CKPTS and OUTS arrays must have the same length" >&2
  exit 1
fi

# Execute commands
for i in "${!EXPS[@]}"; do
  exp="${EXPS[i]}"
  out="${OUTS[i]}"

  echo "Running with experimnt: $exp"
  echo " -> output: $out"

  MESSAGE="Starting evaluation for : ${exp}"
  curl -s -X POST https://api.telegram.org/bot7334130243:AAEq4ZeSbeMgn2MotMUSnBVmDeF7qD5acr4/sendMessage -d chat_id=20403805 -d text="$MESSAGE"

  python3 compute_metric_fold.py --sr "$exp" --output_csv "$out"

done