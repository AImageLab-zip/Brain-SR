import wget
from urllib.error import HTTPError
import os

base_url = "https://ftp.bigbrainproject.org/bigbrain-ftp/BigBrainRelease.2015/2D_Final_Sections/Coronal/Minc/"

save_path = "/homes/gcasari/bigbrain/work_data/BigBrain/low_res_coronal_minc/"

hr_path  = "/homes/gcasari/bigbrain/work_data/BigBrain/high_res_aligned/"
affine_files = os.listdir(hr_path)
affine_files = [f for f in affine_files if f.endswith(".json")]
affine_ids = [filename.split("_")[1] for filename in affine_files]
affine_ids.sort()

for i in affine_ids:
    filename = f"pm{i}o.mnc"
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