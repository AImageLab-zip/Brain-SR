#!/bin/bash
#SBATCH --job-name=bb_crop_gen
#SBATCH --output=/homes/gcasari/io/output_%x.txt
#SBATCH --error=/homes/gcasari/io/error_%x.txt
#SBATCH --nodes=1
#SBATCH --partition=all_usr_prod
#SBATCH --account=bolelli_synthetic
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

cd /homes/gcasari/bigbrain/utils

python3 CropGenerator.py
