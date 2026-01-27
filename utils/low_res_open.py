import os
import nibabel as nib
import numpy as np
from PIL import Image

from open_crop import open_low_res

#low_res_path = '/work/bolelli_synthetic/example_data/original/pm2956o.mnc'
#low_res_img, low_res_affine = open_low_res(low_res_path)

#min_x, max_x = 0, 4000
#min_z, max_z = 1000, 3000
#
#low_res_crop = low_res_img[min_z:max_z, 0, min_x:max_x]
#
#plt.imshow(low_res_crop, origin="lower")
#plt.show()
#
#print()


def save_slices_as_png(volume: np.ndarray, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Normalize to 0–255
    vol_norm = (volume - volume.min()) / (volume.max() - volume.min())
    vol_uint8 = (vol_norm * 255).astype(np.uint8)
    
    slice_img = vol_uint8[:, 0, :]

    slice_img = np.flipud(slice_img)

    im = Image.fromarray(slice_img)
    im.save(os.path.join(out_dir, f"{os.path.basename(low_res_path).split('.')[0]}.png"))

low_res_path = '/homes/gcasari/bigbrain/work_data/BigBrain/low_res_coronal_minc/pm3196o.mnc'
low_res_img, low_res_affine = open_low_res(low_res_path)

save_slices_as_png(low_res_img, out_dir="/homes/gcasari/bigbrain/work_data/example_data/original_3196")
