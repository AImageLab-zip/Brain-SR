import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def split_lr_hr_dataset(lr_dir, hr_dir, out_dir, test_ratio=0.1):
    
    lr_dir = Path(lr_dir)
    hr_dir = Path(hr_dir)
    out_dir = Path(out_dir)

    # Ensure output directories exist
    for split in ['train', 'test']:
        for res in ['lr', 'hr']:
            (out_dir / split / res).mkdir(parents=True, exist_ok=True)

    print("sorting")
    # Get matching filenames
    lr_files = os.listdir(lr_dir)
    hr_files = os.listdir(hr_dir)

    print("matching")
    filenames = [f for f in lr_files if f.endswith("png") and f in hr_files]
    print("shuffling")
    random.shuffle(filenames)
    
    print("splitting")

    split_index = int(len(filenames) * (1 - test_ratio))
    train_files = filenames[:split_index]
    test_files = filenames[split_index:]

    def copy_files(file_list, split):
        for fname in tqdm(file_list):
            shutil.copy(lr_dir / fname, out_dir / split / 'lr' / fname)
            shutil.copy(hr_dir / fname, out_dir / split / 'hr' / fname)

    copy_files(train_files, 'train')
    copy_files(test_files, 'test')

    print(f"Split complete: {len(train_files)} train and {len(test_files)} test samples.")

# Example usage:
split_lr_hr_dataset("/homes/gcasari/bigbrain/work_data/crops/random10/low", 
                    "/homes/gcasari/bigbrain/work_data/crops/random10/high", 
                    "/homes/gcasari/bigbrain/work_data/crops_datasets/random10", 
                    test_ratio=0.1)