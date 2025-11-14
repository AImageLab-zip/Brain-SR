import os
import sys
import argparse
import torch
import pyiqa
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from cleanfid import fid

# Add parent directory to path to import from utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.util_fft import generate_fft_loss_func

ftt_loss = generate_fft_loss_func([{"patch_size": 64, "patch_stride": 32}, 
                                   {"patch_size": 32, "patch_stride": 16}])

def load_image(path):
    img = Image.open(path).convert("RGB")
    return T.ToTensor()(img)  # [C, H, W] in [0, 1]

def compute_l1(img1, img2):
    return F.l1_loss(img1, img2, reduction='mean').item()

def compute_l2(img1, img2):
    return F.mse_loss(img1, img2, reduction='mean').item()

def compute_ftt(img1, img2):
    return ftt_loss(img1, img2).item()

@torch.no_grad()
def main(f1, f2, output_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Init perceptual metrics
    lpips = pyiqa.create_metric('lpips-vgg', device=device, as_loss=False)
    psnr = pyiqa.create_metric('psnr', device=device, color_space='ycbcr', test_y_channel=True)
    ssim = pyiqa.create_metric('ssim', device=device, color_space='ycbcr', test_y_channel=True)
    ms_ssim = pyiqa.create_metric('ms_ssim', device=device, color_space='ycbcr', test_y_channel=True)

    records = []

    # List of filenames shared by both folders
    files = sorted(f for f in os.listdir(f1)
                   if os.path.isfile(os.path.join(f2, f)))
    
    print("Computing FID...")
    fid_score = fid.compute_fid(f2, f1, mode="clean", num_workers=4)

    for fname in tqdm(files, desc="Evaluating images"):
        path_ref = os.path.join(f1, fname, "low") # reference
        path_gen = os.path.join(f2, fname) # generated

        img_ref = load_image(path_ref).unsqueeze(0).to(device)  # [1, C, H, W]
        img_gen = load_image(path_gen).unsqueeze(0).to(device)

        record = {
            'filename': fname,
            'L1': compute_l1(img_ref, img_gen),
            'L2': compute_l2(img_ref, img_gen),
            'PSNR': psnr(img_gen, img_ref).item(),
            'SSIM': ssim(img_gen, img_ref).item(),
            'LPIPS-VGG': lpips(img_gen, img_ref).item(),
            'FTT': compute_ftt(img_ref, img_gen),
            "MS-SSIM": ms_ssim(img_gen, img_ref).item(),
        }
        
        records.append(record)

    # Add FID scores to all records
    for record in records:
        record['FID'] = fid_score

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)

    print("\n=== Aggregated Results ===")
    print(df.drop(columns=["filename"]).mean(numeric_only=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--f1", type=str, required=True, help="Path to reference (ground truth) images")
    parser.add_argument("--f2", type=str, required=True, help="Path to generated images")
    parser.add_argument("--output_csv", type=str, default="metrics_results.csv", help="Output CSV file for per-image metrics")
    args = parser.parse_args()

    main(args.f1, args.f2, args.output_csv)