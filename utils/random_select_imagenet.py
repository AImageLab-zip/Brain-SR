import os
import random
from PIL import Image

# Source (unzipped Kaggle dataset)
src_dir = "/homes/gcasari/bigbrain/work_data/imagenet/130k"
# Destination folder
dst_dir = "/homes/gcasari/bigbrain/work_data/imagenet/net1k"
os.makedirs(dst_dir, exist_ok=True)

# Collect all image paths (recursively)
all_imgs = []
for root, _, files in os.walk(src_dir):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            all_imgs.append(os.path.join(root, f))

print(f"Total images found: {len(all_imgs)}")

# Randomly pick 1000
sample = random.sample(all_imgs, 1000)

# Convert + save as PNG
for i, img_path in enumerate(sample, 1):
    try:
        img = Image.open(img_path).convert("RGB")
        out_path = os.path.join(dst_dir, f"img_{i:04d}.png")
        img.save(out_path, "PNG")
    except Exception as e:
        print(f"Skipped {img_path} due to error: {e}")

print(f"Saved {len(sample)} images as PNG in '{dst_dir}'")