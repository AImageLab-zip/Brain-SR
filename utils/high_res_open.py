import tifffile as tiff
import matplotlib.pyplot as plt
import numpy as np

high_res_path = "/work/bolelli_synthetic/example_data/high-res/aligned/B20_2956.tif"

with tiff.TiffFile(high_res_path) as tif:
    high_res_image_page = tif.pages[0]


min_x, max_x = 70000, 71000
min_z, max_z = 60000, 61000


hig_res_image = high_res_image_page.asarray()#out="memmap")
hig_res_image = np.flip(hig_res_image, axis=0)

hig_res_crop = hig_res_image[min_z:max_z, min_x:max_x]

hig_res_crop_downsampled = hig_res_crop[::16, ::16]
plt.imshow(hig_res_crop_downsampled, origin="lower")
plt.show()

