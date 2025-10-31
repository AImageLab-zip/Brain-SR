import os
import random
import shutil

high_dir = "/homes/gcasari/bigbrain/work_data/crops_datasets/val/high"
low_dir = "/homes/gcasari/bigbrain/work_data/crops_datasets/val/low"
out_dir = "/homes/gcasari/bigbrain/work_data/crops_datasets/val1000"
n = 1000  # number of pairs to copy

# Create output subfolders
os.makedirs(os.path.join(out_dir, "high"), exist_ok=True)
os.makedirs(os.path.join(out_dir, "low"), exist_ok=True)

# Get list of matching filenames
common = sorted(set(os.listdir(high_dir)) & set(os.listdir(low_dir)))
sampled = random.sample(common, min(n, len(common)))

# Copy selected files
for name in sampled:
    shutil.copy2(os.path.join(high_dir, name), os.path.join(out_dir, "high", name))
    shutil.copy2(os.path.join(low_dir, name),  os.path.join(out_dir, "low", name))

print(f"Copied {len(sampled)} pairs to '{out_dir}'")