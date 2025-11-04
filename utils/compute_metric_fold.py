import os
import argparse
import torch
import pyiqa
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from cleanfid import fid

from util_fft import generate_fft_loss_func

ftt_loss = generate_fft_loss_func([{"patch_size": 64, "patch_stride": 32}, {"patch_size": 32, "patch_stride": 16}])

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
def main(f1, output_csv):

    test_fold_folder = "/homes/gcasari/bigbrain/work_data/crops_datasets/test_folds"
    folds = os.listdir(test_fold_folder)

    for f in folds:
        fold_path = os.path.join(f1, f)
        if not os.path.isdir(fold_path):
            raise FileNotFoundError(f"{fold_path} do not exists!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init perceptual metrics
    lpips = pyiqa.create_metric('lpips-vgg', device=device, as_loss=False)
    psnr = pyiqa.create_metric('psnr', device=device, color_space='ycbcr', test_y_channel=True)
    ssim = pyiqa.create_metric('ssim', device=device, color_space='ycbcr', test_y_channel=True)
    ms_ssim = pyiqa.create_metric('ms_ssim', device=device, color_space='ycbcr', test_y_channel=True)

    # Store results for each fold
    fold_results = {}

    for fold in folds:

        print(f"\n=== Processing Fold {fold} ===")

        true_dir = os.path.join(test_fold_folder, fold, "high")
        gen_dir = os.path.join(f1, fold)

        records = []

        # List of filenames shared by both folders
        files = sorted(f for f in os.listdir(true_dir)
                    if os.path.isfile(os.path.join(gen_dir, f)))
        
        files = files[:20]
        
        print("Computing FID...")
        fid_score = 50#fid.compute_fid(true_dir, gen_dir, mode="clean", num_workers=4)

        for fname in tqdm(files, desc="Evaluating images"):
            path_ref = os.path.join(true_dir, fname) # reference
            path_gen = os.path.join(gen_dir, fname) # generated

            img_ref = load_image(path_ref).unsqueeze(0).to(device)  # [1, C, H, W]
            img_gen = load_image(path_gen).unsqueeze(0).to(device)

            record = {
                'filename': fname,
                "fold": fold,
                'L1': compute_l1(img_ref, img_gen),
                'L2': compute_l2(img_ref, img_gen),
                'PSNR': psnr(img_gen, img_ref).item(),
                'SSIM': ssim(img_gen, img_ref).item(),
                'LPIPS-VGG': lpips(img_gen, img_ref).item(),
                'FTT': compute_ftt(img_ref, img_gen),
                "MS-SSIM": ms_ssim(img_gen, img_ref).item(),
                "FID": fid_score,
            }
            
            records.append(record)

        df_fold = pd.DataFrame(records)
        fold_results[fold] = df_fold.drop(columns=["filename", "fold"]).mean(numeric_only=True)

        print(f"\n=== Results for Fold {fold} ===")
        print(fold_results[fold])
    

    df_fold_means = pd.DataFrame(fold_results).T  # fold_results: {fold: Series(mean metrics)}

    df_fold_means.to_csv(output_csv, index=False)

    # Compute mean of per-fold means and std of per-fold means (std over folds)
    mean_results = df_fold_means.mean(axis=0, numeric_only=True)
    std_results = df_fold_means.std(axis=0, ddof=0, numeric_only=True) 

    print("\n=== Final Results Across All Folds (computed over per-fold means) ===")
    print("\nMean:")
    print(mean_results)
    print("\nStandard Deviation (over fold means):")
    print(std_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=str, required=True, help="Path to generated")
    parser.add_argument("--output_csv", type=str, default="metrics_results.csv", help="Output CSV file for per-image metrics")
    args = parser.parse_args()

    main(args.sr, args.output_csv)