#!/bin/bash
#SBATCH --job-name=bigbrain_eval
#SBATCH --output=/homes/gcasari/io/output_%x.txt
#SBATCH --error=/homes/gcasari/io/error_%x.txt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=all_usr_prod
#SBATCH --account=bolelli_synthetic
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTXA5000_24G|gpu_A40_48G|gpu_L40S_48G"

cd /homes/gcasari/bigbrain/utils

bash automatic_evalutation.sh