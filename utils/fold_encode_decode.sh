#!/bin/bash

cd /homes/gcasari/bigbrain/InvSR/

set -euo pipefail

# Hardcoded lists (one-to-one mapping)
IN_FOLDERS=(
  "/homes/gcasari/kair/work_data/crops_datasets/test_folds/fold_0/high"
  "/homes/gcasari/kair/work_data/crops_datasets/test_folds/fold_1/high"
  "/homes/gcasari/kair/work_data/crops_datasets/test_folds/fold_2/high"
  "/homes/gcasari/kair/work_data/crops_datasets/test_folds/fold_3/high"
  "/homes/gcasari/kair/work_data/crops_datasets/test_folds/fold_4/high"
)

OUT_FOLDERS=(
  "/homes/gcasari/bigbrain/work_data/out_report/fold_recons/fold_0"
  "/homes/gcasari/bigbrain/work_data/out_report/fold_recons/fold_1"
  "/homes/gcasari/bigbrain/work_data/out_report/fold_recons/fold_2"
  "/homes/gcasari/bigbrain/work_data/out_report/fold_recons/fold_3"
  "/homes/gcasari/bigbrain/work_data/out_report/fold_recons/fold_4"
)

if [ "${#IN_FOLDERS[@]}" -ne "${#OUT_FOLDERS[@]}" ]; then
  echo "Error: IN_FOLDERS and OUT_FOLDERS arrays must have the same length" >&2
  exit 1
fi

# Execute commands
for i in "${!IN_FOLDERS[@]}"; do
  in_folder="${IN_FOLDERS[i]}"
  out_folder="${OUT_FOLDERS[i]}"

  echo "Running encode_decode for input: $in_folder"
  echo " -> output: $out_folder"

  MESSAGE="Starting encode_decode for fold $i: ${in_folder}"
  curl -s -X POST https://api.telegram.org/bot7334130243:AAEq4ZeSbeMgn2MotMUSnBVmDeF7qD5acr4/sendMessage -d chat_id=20403805 -d text="$MESSAGE"

  python3 encode_decode.py \
    --started_ckpt_path "/homes/gcasari/bigbrain/work_data/logs/fft_nol2_adj/ckpts/model_99000.pth" \
    --cfg_path "/homes/gcasari/bigbrain/InvSR/configs/sample-sd-turbo.yaml" \
    -i "$in_folder" \
    -o "$out_folder"

done