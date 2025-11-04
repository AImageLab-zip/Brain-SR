#!/bin/bash

cd /homes/gcasari/bigbrain/InvSR/

set -euo pipefail

echo "$(pwd)"

# Configuration
CFG_PATH="/homes/gcasari/bigbrain/InvSR/configs/sample-sd-turbo.yaml"
BS=16

# Hardcoded lists (one-to-one mapping)
CKPTS=(
  "/homes/gcasari/bigbrain/work_data/logs/fft_nol2_adj_small/ckpts/model_99000.pth"
  "/homes/gcasari/bigbrain/work_data/logs/fft_nol2_adj/ckpts/model_99000.pth"
  "/homes/gcasari/bigbrain/work_data/logs/fft_nol2_adj_big1/ckpts/model_99000.pth"
  "/homes/gcasari/bigbrain/work_data/logs/fft_bridge_16/ckpts/model_99000.pth"
  "/homes/gcasari/bigbrain/work_data/logs/lpip2_2e5/ckpts/model_44000.pth"
  "/homes/gcasari/bigbrain/work_data/logs/mid_disc/ckpts/model_50000.pth" 
)

OUTS=(
  "/homes/gcasari/bigbrain/work_data/out_report/fft_nol2_small_folds/"
  "/homes/gcasari/bigbrain/work_data/out_report/fft_nol2_folds/"
  "/homes/gcasari/bigbrain/work_data/out_report/fft_nol2_big_folds/"
  "/homes/gcasari/bigbrain/work_data/out_report/fft_bridge16_folds/"
  "/homes/gcasari/bigbrain/work_data/out_report/lpip2_2e5_folds/"
  "/homes/gcasari/bigbrain/work_data/out_report/mid_disc_folds/"
)

if [ "${#CKPTS[@]}" -ne "${#OUTS[@]}" ]; then
  echo "Error: CKPTS and OUTS arrays must have the same length" >&2
  exit 1
fi

# Execute commands
for i in "${!CKPTS[@]}"; do
  ckpt="${CKPTS[i]}"
  out="${OUTS[i]}"

  echo "Running with checkpoint: $ckpt"
  echo " -> output: $out"

  MESSAGE="Starting inference for checkpoint: ${ckpt} — output: ${out}"
  curl -s -X POST https://api.telegram.org/bot7334130243:AAEq4ZeSbeMgn2MotMUSnBVmDeF7qD5acr4/sendMessage -d chat_id=20403805 -d text="$MESSAGE"

  python3 inference_invsr_fold.py --started_ckpt_path "$ckpt" --cfg_path "$CFG_PATH" -o "$out" --bs "$BS"

done