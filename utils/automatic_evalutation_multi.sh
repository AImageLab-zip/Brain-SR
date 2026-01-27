#!/bin/bash

cd /homes/gcasari/bigbrain/utils/

set -euo pipefail

# Hardcoded lists (one-to-one mapping)
EXPS=(
  #"/homes/gcasari/bigbrain/work_data/out_report/l2only_bridge_folds/"
  "/homes/gcasari/bigbrain/work_data/out_report/fft_nol2_small_folds/"
  "/homes/gcasari/bigbrain/work_data/out_report/swin_base_folds/"
  "/homes/gcasari/bigbrain/work_data/out_report/swin_fft_folds/"
)

OUTS=(
  #"/homes/gcasari/bigbrain/work_data/out_report/l2only_bridge_folds/fold_metrics.csv"
  "/homes/gcasari/bigbrain/work_data/out_report/fft_nol2_small_folds/fold_metric_recons.csv"
  "/homes/gcasari/bigbrain/work_data/out_report/swin_base_folds/fold_metrics.csv"
  "/homes/gcasari/bigbrain/work_data/out_report/swin_fft_folds/fold_metrics.csv"
)

PARAMS=(
  "--gt_folder /homes/gcasari/bigbrain/work_data/out_report/fold_recons/ --recons"
  "--swinir"
  "--swinir"
)

if [ "${#EXPS[@]}" -ne "${#OUTS[@]}" ]; then
  echo "Error: CKPTS and OUTS arrays must have the same length" >&2
  exit 1
fi

# Execute commands
for i in "${!EXPS[@]}"; do
  exp="${EXPS[i]}"
  out="${OUTS[i]}"
  params="${PARAMS[i]}"

  echo "Running with experimnt: $exp"
  echo " -> output: $out"

  MESSAGE="Starting evaluation for : ${exp}"
  curl -s -X POST https://api.telegram.org/bot7334130243:AAEq4ZeSbeMgn2MotMUSnBVmDeF7qD5acr4/sendMessage -d chat_id=20403805 -d text="$MESSAGE"

  python3 compute_metric_fold_opts.py --sr "$exp" --output_csv "$out" $params

done