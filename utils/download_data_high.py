import wget
from urllib.error import HTTPError
import os

base_url = "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000070_BigBrain-selected_1um_scans_pub/v1.0/aligned/"

save_path = "/work/bolelli_synthetic/BigBrain/high_res_aligned/"

affine_files = os.listdir(save_path)
affine_files = [f for f in affine_files if f.endswith(".json")]
affine_ids = [filename.split("_")[1] for filename in affine_files]
affine_ids.sort()

for i in affine_ids:
    filename = f"B20_{i}.tif"
    url = base_url + filename

    outpath = save_path + filename

    if os.path.exists(outpath):
        print("Already downloaded:", filename)
        continue

    try:
        print("DOWNLOADING:", filename)
        wget.download(url, outpath)
        print("Downloaded:", filename)
    except HTTPError as e:
        print(e)
        continue