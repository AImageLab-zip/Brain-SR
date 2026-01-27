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
import math

from util_fft import generate_fft_loss_func

ftt_loss = generate_fft_loss_func([{"patch_size": 64, "patch_stride": 32}, {"patch_size": 32, "patch_stride": 16}])

def load_image(path):
    # Returns torch.float32 tensor in [0,1], shape [C,H,W]
    img = Image.open(path).convert("RGB")
    return T.ToTensor()(img)

def l1_per_image(x, y):
    # x,y: [B,C,H,W] -> [B]
    return F.l1_loss(x, y, reduction='none').mean(dim=(1,2,3))

def l2_per_image(x, y):
    # x,y: [B,C,H,W] -> [B]
    return F.mse_loss(x, y, reduction='none').mean(dim=(1,2,3))

def compute_ftt(x, y):
    # x,y: [B,C,H,W] -> [B]
    return ftt_loss(x, y)

def auto_batch_size(base=1):
    if not torch.cuda.is_available():
        return base

    props = torch.cuda.get_device_properties(0)
    total_mem_gb = props.total_memory / (1024 ** 3)

    print(total_mem_gb)

    # Simple heuristic mapping (tune to your pipeline)
    if total_mem_gb < 8:
        bs = base
    elif total_mem_gb < 16:
        bs = 16
    elif total_mem_gb < 24:
        bs = 32
    else:
        bs = 32

    print("SELECTED BS", bs)
    return bs

@torch.no_grad()
def main(gt_dir, sr_dir, output_csv, limit, swinir, recons):
    torch.backends.cudnn.benchmark = True

    batch_size = auto_batch_size()

    folds = sorted([f for f in os.listdir(gt_dir) if os.path.isdir(os.path.join(gt_dir, f))])
    for f in folds:
        fold_path = os.path.join(sr_dir, f)
        if not os.path.isdir(fold_path):
            raise FileNotFoundError(f"{fold_path} does not exist!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init metrics (batchable)
    lpips = pyiqa.create_metric('lpips-vgg', device=device, as_loss=False)
    psnr = pyiqa.create_metric('psnr', device=device, color_space='ycbcr', test_y_channel=True)
    ssim = pyiqa.create_metric('ssim', device=device, color_space='ycbcr', test_y_channel=True)
    ms_ssim = pyiqa.create_metric('ms_ssim', device=device, color_space='ycbcr', test_y_channel=True)

    fold_results = {}

    for fold in folds:
        print(f"\n=== Processing Fold {fold} ===")
        if recons:
            true_dir = os.path.join(gt_dir, fold)
        else:
            true_dir = os.path.join(gt_dir, fold, "high")
        gen_dir = os.path.join(sr_dir, fold)

        # shared filenames
        if not swinir:
            files = sorted(f for f in os.listdir(true_dir) if os.path.isfile(os.path.join(gen_dir, f)))
        else:
            # SwinIR outputs have different names
            files = sorted(f for f in os.listdir(true_dir) if os.path.isfile(os.path.join(gen_dir, f[:-4] + '_SwinIR.png')))
        
        if limit is not None and limit > 0:
            files = files[:limit]

        # FID once per fold 
        if limit is not None and limit>0:
            # if its a test on less images, do not compute fid
            fid_score = 50
        else:
            fid_score = fid.compute_fid(true_dir, gen_dir, mode="clean", num_workers=8)

        records = []
        n = len(files)
        n_batches = math.ceil(n / batch_size)

        for bi in tqdm(range(n_batches), desc="Evaluating (batched)"):
            batch_files = files[bi*batch_size : (bi+1)*batch_size]
            # Load to host
            refs_cpu = [load_image(os.path.join(true_dir, f)) for f in batch_files]
            if not swinir:
                gens_cpu = [load_image(os.path.join(gen_dir, f)) for f in batch_files]
            else:
                # SwinIR outputs have different names
                gens_cpu = [load_image(os.path.join(gen_dir,  f[:-4] + '_SwinIR.png')) for f in batch_files]

            # Stack and move once to GPU
            ref = torch.stack(refs_cpu, dim=0).to(device, non_blocking=True)
            gen = torch.stack(gens_cpu, dim=0).to(device, non_blocking=True)

            # Per-image metrics (batched)
            L1 = l1_per_image(ref, gen)                   # [B]
            L2 = l2_per_image(ref, gen)                   # [B]
            FTT = compute_ftt(ref, gen)                   # [B]

            # PyIQA metrics (they return [B] tensors)
            PSNR = psnr(gen, ref)                         # [B]
            SSIM = ssim(gen, ref)                         # [B]
            MS_SSIM = ms_ssim(gen, ref)                   # [B]
            LPIPS = lpips(gen, ref).squeeze()             # [B]

            # Move to CPU once
            L1 = L1.detach().cpu().tolist()
            L2 = L2.detach().cpu().tolist()
            FTT = FTT.detach().cpu().tolist()
            PSNR = PSNR.detach().cpu().tolist()
            SSIM = SSIM.detach().cpu().tolist()
            MS_SSIM = MS_SSIM.detach().cpu().tolist()
            LPIPS = LPIPS.detach().cpu().tolist()

            for i, fname in enumerate(batch_files):
                records.append({
                    'filename': fname,
                    'fold': fold,
                    'L1': L1[i],
                    'L2': L2[i],
                    'PSNR': PSNR[i],
                    'SSIM': SSIM[i],
                    'LPIPS-VGG': LPIPS[i],
                    'FTT': FTT[i],
                    'MS-SSIM': MS_SSIM[i],
                    'FID': float(fid_score),
                })

        df_fold = pd.DataFrame(records)
        fold_results[fold] = df_fold.drop(columns=["filename", "fold"]).mean(numeric_only=True)

        print(f"\n=== Results for Fold {fold} ===")
        print(fold_results[fold])

    df_fold_means = pd.DataFrame(fold_results).T

    # Compute mean and std across folds
    mean_results = df_fold_means.mean(axis=0, numeric_only=True)
    std_results  = df_fold_means.std(axis=0, ddof=0, numeric_only=True)

    # Add empty separator row (all NaN)
    empty_row = pd.Series({col: None for col in df_fold_means.columns}, name="")

    # Add mean and std rows with proper labels
    mean_row = pd.Series(mean_results, name="Mean")
    std_row  = pd.Series(std_results, name="Std")

    # Concatenate them
    df_fold_means = pd.concat([df_fold_means, empty_row.to_frame().T, mean_row.to_frame().T, std_row.to_frame().T])

    # Save to CSV
    df_fold_means.to_csv(output_csv, index=True)  # keep index so 'Mean' and 'Std' are visible

    print("\n=== Final Results Across All Folds (computed over per-fold means) ===")
    print("\nMean:")
    print(mean_results)
    print("\nStandard Deviation (over fold means):")
    print(std_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=str, required=True, help="Path to generated")
    parser.add_argument("--output_csv", type=str, default="metrics_results.csv",
                        help="Output CSV file for per-fold means")
    parser.add_argument("--limit", type=int, default=0, help="Limit images per fold (0=all)")
    parser.add_argument("--swinir", action="store_true", help="Use if evaluating SwinIR outputs")
    parser.add_argument("--gt_folder", type=str, default="/homes/gcasari/bigbrain/work_data/crops_datasets/test_folds")
    parser.add_argument("--recons", action="store_true", help="Use if evaluating reconstructions")

    args = parser.parse_args()
    # Use keyword args to avoid ordering mistakes and improve readability
    main(
        gt_dir=args.gt_folder,
        sr_dir=args.sr,
        output_csv=args.output_csv,
        limit=(args.limit if args.limit > 0 else None),
        swinir=args.swinir,
        recons=args.recons
    )