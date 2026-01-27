#!/bin/bash
#SBATCH --job-name=bigbrain_nol2
#SBATCH --output=/homes/gcasari/io/output_%x.txt
#SBATCH --error=/homes/gcasari/io/error_%x.txt
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=bolelli_synthetic
#SBATCH --cpus-per-task=21
#SBATCH --time=24:00:00
#SBATCH --mem-per-gpu=25GB
#SBATCH --constraint="gpu_A40_48G|gpu_L40S_48G"


cd /homes/gcasari/bigbrain/InvSR/

CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 --nnodes=1 main.py --run_name fft_bridge_nol2 --cfg_path /homes/gcasari/bigbrain/InvSR/configs/fft_bridge_nol2.yaml --resume
