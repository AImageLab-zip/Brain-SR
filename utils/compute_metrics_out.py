import os
import argparse
import torch
import pyiqa
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

from util_fft import generate_fft_loss_func

ftt_loss = generate_fft_loss_func([{"patch_size": 8, "patch_stride": 4}, {"patch_size": 4, "patch_stride": 2}])

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

    records = []

    # List of filenames shared by both folders
    files = sorted(f for f in os.listdir(f1)
                   if os.path.isfile(os.path.join(f2, f)))

    for fname in tqdm(files, desc="Evaluating images"):
        path_ref = os.path.join(f1, fname)
        path_gen = os.path.join(f2, fname)

        img_ref = load_image(path_ref).unsqueeze(0).to(device)  # [1, C, H, W]
        img_gen = load_image(path_gen).unsqueeze(0).to(device)

        record = {
            'filename': fname,
            'L1': compute_l1(img_ref, img_gen),
            'L2': compute_l2(img_ref, img_gen),
            'PSNR': psnr(img_gen, img_ref).item(),
            'SSIM': ssim(img_gen, img_ref).item(),
            'LPIPS-VGG': lpips(img_gen, img_ref).item(),
            'FTT': compute_ftt(img_ref, img_gen)
        }

        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)

    print("\n=== Aggregated Results ===")
    print(df.drop(columns=["filename"]).mean())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--f1", type=str, required=True, help="Path to reference (ground truth) images")
    parser.add_argument("--f2", type=str, required=True, help="Path to generated images")
    parser.add_argument("--output_csv", type=str, default="metrics_results.csv", help="Output CSV file for per-image metrics")
    args = parser.parse_args()

    main(args.f1, args.f2, args.output_csv)