#!/bin/bash
#SBATCH --job-name=bigbrain_train_bs
#SBATCH --output=/homes/gcasari/io/output_%x.txt
#SBATCH --error=/homes/gcasari/io/error_%x.txt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=bolelli_synthetic
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-gpu=40GB

cd /homes/gcasari/bigbrain/InvSR/

base="
--save_dir ../work_data/logs/
--cfg_path /homes/gcasari/bigbrain/InvSR/configs/bigbrain-train-adj-bs.yaml
"

python3 main.py ${base}
