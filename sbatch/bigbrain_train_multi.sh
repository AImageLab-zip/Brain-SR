#!/bin/bash
#SBATCH --job-name=bigbrain_train_3gpu
#SBATCH --output=/homes/gcasari/io/output_%x.txt
#SBATCH --error=/homes/gcasari/io/error_%x.txt
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --partition=all_usr_prod
#SBATCH --account=bolelli_synthetic
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-gpu=18GB
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTXA5000_24G|gpu_A40_48G|gpu_L40S_48G"


cd /homes/gcasari/bigbrain/InvSR/

CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 --nnodes=1 main.py --run_name lpips_2e5 --cfg_path /homes/gcasari/bigbrain/InvSR/configs/lpips_2e5.yaml --resume