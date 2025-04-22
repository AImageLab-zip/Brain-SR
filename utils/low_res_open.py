import numpy as np
import matplotlib.pyplot as plt

from open_crop import open_low_res

low_res_path = '/work/bolelli_synthetic/example_data/original/pm2956o.mnc'
low_res_img, low_res_affine = open_low_res(low_res_path)

min_x, max_x = 0, 4000
min_z, max_z = 1000, 3000

low_res_crop = low_res_img[min_z:max_z, 0, min_x:max_x]

plt.imshow(low_res_crop, origin="lower")
plt.show()

print()



