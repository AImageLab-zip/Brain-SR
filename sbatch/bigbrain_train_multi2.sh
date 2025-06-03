#!/bin/bash
#SBATCH --job-name=bigbrain_train_2gpu
#SBATCH --output=/homes/gcasari/io/output_%x.txt
#SBATCH --error=/homes/gcasari/io/error_%x.txt
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --partition=all_usr_prod
#SBATCH --account=bolelli_synthetic
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

#SBATCH --constraint="gpu_RTX5000_16G|gpu_RTX6000_24G|gpu_RTXA5000_24G|gpu_A40_48G|gpu_L40S_48G"


cd /homes/gcasari/bigbrain/InvSR/

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 --nnodes=1 main.py --save_dir ../work_data/logs/ --cfg_path /homes/gcasari/bigbrain/InvSR/configs/bigbrain-train-3-gpu.yaml