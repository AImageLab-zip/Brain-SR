#!/usr/bin/env python3
import argparse
import os
import math
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

def imread_float(path):
    """Read image with OpenCV (BGR) and return float32 in [0,1], preserving channels."""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.dtype != np.uint8:
        # Convert non-8bit (e.g., 16-bit) to float in [0,1]
        img = img.astype(np.float32)
        img = (img - img.min()) / max(1e-12, (img.max() - img.min()))
    else:
        img = img.astype(np.float32) / 255.0
    # Ensure 3-channel for color, or 1-channel for grayscale
    if img.ndim == 2:
        return img[..., None]  # HxW -> HxWx1
    if img.shape[2] == 4:  # drop alpha if present
        img = img[:, :, :3]
    return img

def to_luminance_y(img):
    """Convert BGR [0,1] to Y (BT.601) in [0,1], returns HxWx1."""
    if img.shape[2] == 1:
        return img
    # OpenCV uses BGR; convert to YCrCb then take Y
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0:1]  # already [0,1] if input is [0,1]
    return y

def ensure_same_size(a, b, resize_to_a=False):
    if a.shape[:2] == b.shape[:2]:
        return a, b
    if not resize_to_a:
        raise ValueError(f"Image sizes differ: real={a.shape[:2]}, gen={b.shape[:2]}. "
                         f"Use --resize-to-real to resize the generated image to the real image size.")
    b_rs = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_CUBIC)
    return a, b_rs

def compute_psnr(gt, pred, eps=1e-12):
    mse = np.mean((gt - pred) ** 2)
    if mse <= 0:
        return float('inf')
    return 20.0 * math.log10(1.0 / math.sqrt(mse + eps))

def compute_ssim_map(gt, pred, channel_axis=None):
    """
    Returns SSIM score and the SSIM map per-pixel.
    gt, pred in [0,1]; channel_axis=None for 2D, or -1 for HxWxC.
    """
    score, smap = ssim(gt.squeeze() if gt.shape[2] == 1 else gt,
                       pred.squeeze() if pred.shape[2] == 1 else pred,
                       data_range=1.0,
                       channel_axis=channel_axis,
                       full=True,
                       gaussian_weights=True,
                       sigma=1.5,
                       use_sample_covariance=False,
                       K1=0.01, K2=0.03)
    return score, smap

def norm_to_uint8(x):
    """Normalize a non-negative array to [0,255] uint8 for visualization."""
    x = np.maximum(x, 0.0).astype(np.float32)
    m = x.max()
    if m <= 1e-12:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x / m) * 255.0
    return np.clip(y, 0, 255).astype(np.uint8)

def apply_colormap(gray_uint8):
    """Apply Jet colormap (OpenCV) to a gray uint8 image; returns BGR uint8."""
    return cv2.applyColorMap(gray_uint8, cv2.COLORMAP_JET)

def overlay_heatmap(base_bgr01, heatmap_bgr_u8, alpha=0.6):
    """Overlay heatmap over base image. base in [0,1] BGR, heatmap uint8 BGR."""
    base_u8 = np.clip(base_bgr01 * 255.0, 0, 255).astype(np.uint8)
    overlay = cv2.addWeighted(heatmap_bgr_u8, alpha, base_u8, 1 - alpha, 0)
    return overlay

def stack_panel(images, labels=None, pad=8):
    """Horizontally stack a list of BGR uint8 images with labels rendered at top-left."""
    h = max(im.shape[0] for im in images)
    resized = [cv2.copyMakeBorder(cv2.resize(im, (round(im.shape[1]*h/im.shape[0]), h)),
                                  top=pad, bottom=pad, left=pad, right=pad,
                                  borderType=cv2.BORDER_CONSTANT, value=(32,32,32))
               for im in images]
    panel = np.hstack(resized)
    if labels:
        # Put labels near the top-left of each tile
        x = 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        for im, lab in zip(resized, labels):
            cv2.putText(panel, lab, (x + 12, 28), font, 0.7, (255,255,255), 2, cv2.LINE_AA)
            x += im.shape[1]
    return panel

def main():
    parser = argparse.ArgumentParser(description="Create difference visualizations between real and generated images.")
    parser.add_argument("real", type=str, help="Path to the real (GT) image")
    parser.add_argument("gen", type=str, help="Path to the generated image")
    parser.add_argument("out_prefix", type=str, help="Output prefix (directory/prefix without extension)")
    parser.add_argument("--mode", choices=["rgb", "y"], default="y",
                        help="Compare on RGB (all channels) or luminance Y (default: y)")
    parser.add_argument("--resize-to-real", action="store_true",
                        help="Resize generated image to match real image size")
    parser.add_argument("--alpha", type=float, default=0.6, help="Overlay alpha for heatmap (default: 0.6)")
    parser.add_argument("--mask-percentile", type=float, default=95.0,
                        help="Percentile of error for binary mask threshold; set <=0 to disable (default: 95)")
    parser.add_argument("--save-panel", action="store_true", help="Save a side-by-side summary panel")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)

    base_filename = os.path.basename(os.path.dirname(args.gen)) + "_" + os.path.splitext(os.path.basename(args.gen))[0]

    print(args.out_prefix)
    print(base_filename)

    # Load
    real = imread_float(args.real)   # BGR in [0,1]
    gen  = imread_float(args.gen)    # BGR in [0,1]
    real, gen = ensure_same_size(real, gen, resize_to_a=args.resize_to_real)

    # Prepare comparison tensors
    if args.mode == "y":
        A = to_luminance_y(real)  # HxWx1
        B = to_luminance_y(gen)
        channel_axis = None
        vis_base = real  # for overlay we still use color
    else:
        A = real  # HxWx3 BGR
        B = gen
        channel_axis = -1
        vis_base = real

    # Compute per-pixel absolute error (L1)
    abs_err = np.abs(A - B)  # HxWxC
    if abs_err.shape[2] > 1:
        # reduce to scalar error (mean across channels)
        abs_err_scalar = abs_err.mean(axis=2)
    else:
        abs_err_scalar = abs_err[:, :, 0]

    # Metrics
    mae = float(abs_err_scalar.mean())
    rmse = float(np.sqrt(((A - B) ** 2).mean()))
    psnr_val = compute_psnr(A, B)
    ssim_score, ssim_map = compute_ssim_map(A, B, channel_axis=channel_axis)
    dssim = 1.0 - ssim_map  # per-pixel dissimilarity in [0,2], typically ~[0,1]

    # Heatmaps
    abs_err_u8 = norm_to_uint8(abs_err_scalar)          # scale to [0,255]
    err_heat = apply_colormap(abs_err_u8)               # BGR uint8
    dssim_u8 = norm_to_uint8(dssim)                     # [0,255]
    dssim_heat = apply_colormap(dssim_u8)               # BGR uint8

    # Overlays
    overlay = overlay_heatmap(vis_base, err_heat, alpha=args.alpha)

    # Binary mask from percentile
    mask_path = None
    if args.mask_percentile > 0:
        thr = np.percentile(abs_err_scalar, args.mask_percentile)
        mask = (abs_err_scalar >= thr).astype(np.uint8) * 255
        # small morphology to clean speckles
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_path = f"{args.out_prefix}{base_filename}_mask.png"
        cv2.imwrite(mask_path, mask_bgr)

    # Save outputs
    out_heat = f"{args.out_prefix}{base_filename}_heatmap.png"
    out_overlay = f"{args.out_prefix}{base_filename}_overlay.png"
    out_ssim = f"{args.out_prefix}{base_filename}_ssim.png"
    cv2.imwrite(out_heat, err_heat)
    cv2.imwrite(out_overlay, overlay)
    cv2.imwrite(out_ssim, dssim_heat)

    # Summary panel
    if args.save_panel:
        real_u8 = np.clip(real * 255.0, 0, 255).astype(np.uint8)
        gen_u8  = np.clip(gen  * 255.0, 0, 255).astype(np.uint8)
        tiles = [real_u8, gen_u8, err_heat, dssim_heat, overlay]
        labels = ["Real", "Generated", "Abs Error Heatmap", "1-SSIM Heatmap", "Overlay"]
        panel = stack_panel(tiles, labels=labels)
        out_panel = f"{args.out_prefix}{base_filename}_panel.png"
        cv2.imwrite(out_panel, panel)
    else:
        out_panel = None

    # Print metrics
    print("=== Metrics (mode: {} on {}x{}x{}) ===".format(
        args.mode, A.shape[1], A.shape[0], A.shape[2]))
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"PSNR: {psnr_val:.3f} dB")
    print(f"SSIM: {ssim_score:.6f}")
    print("\nSaved:")
    print(f"  Error heatmap:   {out_heat}")
    print(f"  Overlay:         {out_overlay}")
    print(f"  1-SSIM heatmap:  {out_ssim}")
    if mask_path:
        print(f"  Diff mask:       {mask_path}")
    if out_panel:
        print(f"  Summary panel:   {out_panel}")

if __name__ == "__main__":
    main()