import os
import random
import shutil
from tqdm import tqdm

high_dir = "/homes/gcasari/bigbrain/work_data/crops_datasets/test/high"
low_dir = "/homes/gcasari/bigbrain/work_data/crops_datasets/test/low"


out_dir = "/homes/gcasari/bigbrain/work_data/crops_datasets/test_folds"
n = 5000  # number of pairs to copy
fold_numbers = 5

common = sorted(set(os.listdir(high_dir)) & set(os.listdir(low_dir)))
sampled = random.sample(common, n*fold_numbers)

for i in range(0, fold_numbers):
    print("processing fold", i)

    fold_dir = os.path.join(out_dir, f"fold_{i}")
    fold_high_dir = os.path.join(fold_dir, "high")
    fold_low_dir = os.path.join(fold_dir, "low")

    # Create output subfolders
    os.makedirs(fold_dir, exist_ok=True)
    os.makedirs(fold_high_dir, exist_ok=True)
    os.makedirs(fold_low_dir, exist_ok=True)

    fold_data = sampled[i*n:(i+1)*n]

    # Copy selected files
    for name in tqdm(fold_data):
        shutil.copy2(os.path.join(high_dir, name), os.path.join(fold_high_dir, name))
        shutil.copy2(os.path.join(low_dir, name),  os.path.join(fold_low_dir, name))

    print(f"Copied {len(fold_data)} pairs to '{fold_dir}'")