#!/bin/bash
#SBATCH --job-name=bigbrain_custom
#SBATCH --output=/homes/gcasari/io/output_%x.txt
#SBATCH --error=/homes/gcasari/io/error_%x.txt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=all_usr_prod
#SBATCH --account=bolelli_synthetic
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-gpu=15GB
#SBATCH --constraint="gpu_RTX5000_16G|gpu_RTX6000_24G|gpu_RTXA5000_24G|gpu_A40_48G|gpu_L40S_48G"

cd /homes/gcasari/bigbrain/InvSR/

base="
--cfg_path /homes/gcasari/bigbrain/InvSR/configs/custom-test-sr.yaml
"

python3 main_custom.py ${base} > /homes/gcasari/io/output_real_custom.txt 2> /homes/gcasari/io/error_real_custom.txt
