import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import random

import sys
import cv2

def bgr2rgb(im): return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def imread(path, chn='rgb', dtype='float32', force_gray2rgb=True, force_rgba2rgb=False):
    '''
    Read image.
    chn: 'rgb', 'bgr' or 'gray'
    out:
        im: h x w x c, numpy tensor
    '''
    try:
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)  # BGR, uint8
    except:
        print(str(path))

    if im is None:
        print(str(path))

    if chn.lower() == 'gray':
        assert im.ndim == 2, f"{str(path)} can't be successfuly read!"
    else:
        if im.ndim == 2:
            if force_gray2rgb:
                im = np.stack([im, im, im], axis=2)
            else:
                raise ValueError(f"{str(path)} has {im.ndim} channels!")
        elif im.ndim == 4:
            if force_rgba2rgb:
                im = im[:, :, :3]
            else:
                raise ValueError(f"{str(path)} has {im.ndim} channels!")
        else:
            if chn.lower() == 'rgb':
                im = bgr2rgb(im)
            elif chn.lower() == 'bgr':
                pass

    if dtype == 'float32':
        im = im.astype(np.float32) / 255.
    elif dtype ==  'float64':
        im = im.astype(np.float64) / 255.
    elif dtype == 'uint8':
        pass
    else:
        sys.exit('Please input corrected dtype: float32, float64 or uint8!')

    return im


def compute_global_mean_std(image_dir: str) -> tuple[float, float]:
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    total_pixel_count = 0

    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    sample_size = int(len(image_files)*0.05)
    image_files = random.sample(image_files, sample_size)
    print(f"Computing params over {sample_size} images")

    for fname in tqdm(image_files, desc=f"Processing {image_dir}"):
        img = imread(os.path.join(image_dir, fname), chn='rgb', dtype='float32')
        img_np = img.astype(np.float64)

        pixel_sum += img_np.sum()
        pixel_sq_sum += (img_np ** 2).sum()
        total_pixel_count += img_np.size

    mean = pixel_sum / total_pixel_count
    std = np.sqrt((pixel_sq_sum / total_pixel_count) - mean**2)

    return mean, std

lr_dir = "/homes/gcasari/bigbrain/work_data/crops/random10/low"
hr_dir = "/homes/gcasari/bigbrain/work_data/crops/random10/high"

mean_lr, std_lr = compute_global_mean_std(lr_dir)
mean_hr, std_hr = compute_global_mean_std(hr_dir)

print(f"LR mean: {mean_lr:.4f}, std: {std_lr:.4f}")
print(f"HR mean: {mean_hr:.4f}, std: {std_hr:.4f}")