import sys
import cv2
import numpy as np
from skimage import img_as_ubyte, img_as_float32


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

    print("Finished!")
    return im

if __name__ == "__main__":
    imread("/homes/gcasari/bigbrain/crops/2956/low_2956_2.png")