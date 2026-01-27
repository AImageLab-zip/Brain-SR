import pyvips

src = "/homes/gcasari/bigbrain/work_data/example_data/high-res/aligned/B20_2956.tif"
x0, y0 = 50000, 30000
w, h = 2000, 2000
dst = "/homes/gcasari/bigbrain/work_data/example_data/crops/B20_2956_crop_2k.tif"

im = pyvips.Image.new_from_file(src, access="sequential")
crop = im.crop(x0, y0, w, h)
crop.write_to_file(f"{dst}[tile,pyramid,bigtiff,compression=zstd]")
print("Saved pyramidal crop ready for QuPath:", dst)