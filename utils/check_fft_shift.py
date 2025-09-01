#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import argparse
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import csv


# -------------------------------
# I/O
# -------------------------------

def load_png_as_tensor(path: str) -> torch.Tensor:
    """
    Load a PNG and return a tensor of shape (C, H, W) in [0,1], float32.
    If grayscale=True, converts to 1 channel. Otherwise keeps RGB (3ch).
    """
    img = Image.open(path).convert("RGB")
    x = torch.from_numpy(
        (torch.ByteTensor(bytearray(img.tobytes())).numpy())
    )  # fast path to raw bytes
    w, h = img.size
    x = x.view(h, w, 3).permute(2, 0, 1).float() / 255.0  # (C,H,W)
    return x


def write_tensor_as_png(tensor: torch.Tensor, path: str):
    """
    Save a tensor of shape (C,H,W) in [0,1] as a PNG image.

    Args:
        tensor: torch.Tensor, shape (C,H,W), values in [0,1]
        path: output file path (should end with .png)
    """
    if tensor.ndim != 3:
        raise ValueError(f"Expected tensor of shape (C,H,W), got {tensor.shape}")

    # Ensure channel-first (C,H,W)
    C, H, W = tensor.shape
    if C not in (1, 3):
        raise ValueError(f"Only 1 or 3 channels supported, got {C}")

    # Clamp and convert to uint8
    arr = (tensor.clamp(0, 1) * 255).byte().cpu().permute(1, 2, 0).numpy()

    # Drop channel axis if grayscale
    if C == 1:
        arr = arr[:, :, 0]
        img = Image.fromarray(arr, mode="L")
    else:
        img = Image.fromarray(arr, mode="RGB")

    img.save(path)


def match_sizes(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ensure tensors have same (C,H,W). If channels differ (e.g. 3 vs 1),
    replicate single channel to 3 or take first channel from RGB to 1,
    then center-crop to the minimum H,W of the two.
    """
    Ca, Ha, Wa = a.shape
    Cb, Hb, Wb = b.shape

    # Harmonize channels
    if Ca != Cb:
        if Ca == 1 and Cb == 3:
            a = a.repeat(3, 1, 1)
        elif Ca == 3 and Cb == 1:
            b = b.repeat(3, 1, 1)
        else:
            # Fallback: take first channel of both to 1ch
            a = a[:1, ...]
            b = b[:1, ...]
    # Center-crop to min size
    H = min(a.shape[1], b.shape[1])
    W = min(a.shape[2], b.shape[2])

    def center_crop(x, H, W):
        h, w = x.shape[1], x.shape[2]
        top = (h - H) // 2
        left = (w - W) // 2
        return x[:, top:top + H, left:left + W]

    return center_crop(a, H, W), center_crop(b, H, W)


# -------------------------------
# FFT helpers & metrics
# -------------------------------

def fft2_ortho(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fft2(x, dim=(-2, -1), norm="ortho")

def mag_and_phase(z: torch.Tensor, eps: float = 1e-8):
    mag = torch.sqrt(z.real * z.real + z.imag * z.imag + eps)
    phase = torch.atan2(z.imag, z.real)  # [-pi, pi]
    return mag, phase

def wrap_phase_diff(phase_pred: torch.Tensor, phase_tgt: torch.Tensor) -> torch.Tensor:
    d = phase_pred - phase_tgt
    d = (d + math.pi) % (2 * math.pi) - math.pi  # wrap to [-pi, pi]
    return d

def spectral_metrics(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    *,
    magnitude_variant: str = "l1",   # 'l1' or 'l2'
    phase_weighted_by_mag: bool = True,
    eps: float = 1e-8
) -> Dict[str, float]:
    """
    pred, tgt: (C,H,W) in [0,1], float32
    Returns scalar metrics averaged over channels and spatial dims.
    """
    Fp = fft2_ortho(pred)
    Ft = fft2_ortho(tgt)

    mag_p, ph_p = mag_and_phase(Fp, eps)
    mag_t, ph_t = mag_and_phase(Ft, eps)

    # Magnitude error
    if magnitude_variant.lower() == "l2":
        mag_err = (mag_p - mag_t).pow(2)
        mag_key = "global_mag_l2"
    else:
        mag_err = (mag_p - mag_t).abs()
        mag_key = "global_mag_l1"

    # Phase error (wrapped)
    dphi = wrap_phase_diff(ph_p, ph_t).abs()
    dphi2 = dphi.pow(2)

    if phase_weighted_by_mag:
        # Weight by target magnitude (normalize by per-channel mean to balance scenes)
        w = mag_t / (mag_t.mean(dim=(-2, -1), keepdim=True) + eps)
        phase_mae = (w * dphi).mean(dim=(-2, -1))   # (C,)
        phase_mse = (w * dphi2).mean(dim=(-2, -1))  # (C,)
        mag_err   = mag_err.mean(dim=(-2, -1))      # (C,)
    else:
        phase_mae = dphi.mean(dim=(-2, -1))
        phase_mse = dphi2.mean(dim=(-2, -1))
        mag_err   = mag_err.mean(dim=(-2, -1))

    return {
        mag_key:   float(mag_err.mean().item()),
        "global_phase_mae": float(phase_mae.mean().item()),
        "global_phase_mse": float(phase_mse.mean().item()),
    }

def pixel_mse(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    return float(F.mse_loss(pred, tgt, reduction="mean").item())


# -------------------------------
# Shifts (integer & subpixel)
# -------------------------------

def integer_roll(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    # dx along H (vertical), dy along W (horizontal)
    return torch.roll(x, shifts=(dx, dy), dims=(-2, -1))

def subpixel_shift(x: torch.Tensor, dx: float, dy: float, mode: str = "bilinear") -> torch.Tensor:
    # Keep size via grid_sample
    C, H, W = x.shape
    tx = 2.0 * (dy / max(W - 1, 1))
    ty = 2.0 * (dx / max(H - 1, 1))
    theta = torch.tensor([[[1, 0, tx], [0, 1, ty]]], dtype=x.dtype, device=x.device)  # (1,2,3)
    grid = F.affine_grid(theta, size=(1, C, H, W), align_corners=False)
    y = F.grid_sample(x.unsqueeze(0), grid, mode=mode, padding_mode="zeros", align_corners=False)
    return y.squeeze(0)


def pinch_bulge_torch(img: torch.Tensor, center: tuple, radius: int, strength: float=0.2) -> torch.Tensor:
    """
    Pinch (strength>0) or bulge (strength<0) a circular region.

    Args:
        img: torch.Tensor, shape (C,H,W) or (B,C,H,W), float in [0,1].
        center: (cx, cy) in pixel coordinates.
        radius: radius in pixels.
        strength: positive=pinch, negative=bulge.
    Returns:
        warped: same shape as img.
    """
    is_batched = (img.dim() == 4)
    if not is_batched:
        img = img.unsqueeze(0)  # add batch dim

    B, C, H, W = img.shape
    device = img.device
    dtype = img.dtype
    cx, cy = center

    # Make base grid in normalized coords [-1,1]
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij"
    )
    dx = xx - cx
    dy = yy - cy
    r = torch.sqrt(dx**2 + dy**2)
    mask = r < radius

    # normalized radius in [0,1]
    r_norm = torch.zeros_like(r)
    r_norm[mask] = r[mask] / radius

    # Smooth falloff (cosine)
    falloff = 0.5 * (1 + torch.cos(torch.pi * r_norm))
    scale = 1 + strength * falloff

    # Inverse mapping
    srcX = cx + dx / scale.clamp_min(1e-6)
    srcY = cy + dy / scale.clamp_min(1e-6)

    # Outside circle -> identity
    srcX[~mask] = xx[~mask]
    srcY[~mask] = yy[~mask]

    # Convert pixel coords to [-1,1] range for grid_sample
    normX = (srcX / (W - 1)) * 2 - 1
    normY = (srcY / (H - 1)) * 2 - 1
    grid = torch.stack((normX, normY), dim=-1).unsqueeze(0)  # (1,H,W,2)

    # Sample
    warped = F.grid_sample(img, grid, mode="bilinear", padding_mode="reflection", align_corners=True)

    if not is_batched:
        warped = warped.squeeze(0)
    return warped


# -------------------------------
# Evaluation loop (no batches)
# -------------------------------

def parse_shifts(s: str, subpixel: bool) -> List[Tuple[float, float]]:
    """
    Example: "0,0 1,0 0,1 2,2" (whitespace-separated dx,dy pairs)
    If subpixel=True, accepts floats; otherwise they’ll be cast to ints.
    """
    out: List[Tuple[float, float]] = []
    for token in s.strip().split():
        dx_s, dy_s = token.split(",")
        dx = float(dx_s)
        dy = float(dy_s)
        if not subpixel:
            dx, dy = int(round(dx)), int(round(dy))
        out.append((dx, dy))
    return out

def evaluate_pair(
    gt: torch.Tensor,
    pred: torch.Tensor,
    shifts: List[Tuple[float, float]],
    magnitude_variant: str,
    phase_weighted_by_mag: bool,
    subpixel: bool,
) -> List[Dict[str, Any]]:
    """
    Returns list of rows (dict) with metrics for the original comparison
    and for each shifted prediction.
    """
    rows: List[Dict[str, Any]] = []

    # Baseline (no shift)
    base_metrics = spectral_metrics(
        pred, gt,
        magnitude_variant=magnitude_variant,
        phase_weighted_by_mag=phase_weighted_by_mag
    )
    base_px = pixel_mse(pred, gt)
    rows.append({
        "variant": "orig",
        "pixel_mse": base_px,
        **base_metrics
    })

    # Shifted versions
    for (dx, dy) in shifts:
        if (dx, dy) == (0, 0):
            continue
        pred_s = subpixel_shift(pred, dx, dy) if subpixel else integer_roll(pred, int(dx), int(dy))

        m = spectral_metrics(
            pred_s, gt,
            magnitude_variant=magnitude_variant,
            phase_weighted_by_mag=phase_weighted_by_mag
        )
        px = pixel_mse(pred_s, gt)
        rows.append({
            "variant": f"shift_dx{dx}_dy{dy}",
            "pixel_mse": px,
            **m
        })
    return rows


def evaluate_pinches(
    gt: torch.Tensor,
    iterations: int,
    radius_range: tuple = (40, 100),
    strenght_range: tuple = (-0.4, 0,4)
) -> tuple: #torch.Tensor, List[Dict[str, Any]]:
    """
    Returns list of rows (dict) with metrics for the original comparison
    and for each shifted prediction.
    """
    rows: List[Dict[str, Any]] = []

    # Baseline (no shift)
    #base_metrics = spectral_metrics(
    #    pred, gt
    #)
    #base_px = pixel_mse(pred, gt)
    #rows.append({
    #    "variant": "orig",
    #    "pixel_mse": base_px,
    #    **base_metrics
    #})

    distorced_gt = gt.clone()

    # Shifted versions
    for i in range(iterations):

        center = torch.randint(0, min(distorced_gt.shape[1], distorced_gt.shape[2]), (2,)).tolist()
        radius = torch.randint(radius_range[0], radius_range[1], (1,)).item()
        strength = torch.FloatTensor(1).uniform_(strenght_range[0], strenght_range[1]).item()

        distorced_gt = pinch_bulge_torch(distorced_gt, center=center, radius=radius, strength=strength)

        m = spectral_metrics(
            distorced_gt, gt,
        )
        px = pixel_mse(distorced_gt, gt)
        rows.append({
            "variant": f"c{i+1}_r{radius}_s{strength:.2f}",
            "pixel_mse": px,
            **m
        })
    return distorced_gt, rows


def print_table(rows: List[Dict[str, Any]]):
    if not rows:
        return
    # Determine columns & pretty print
    cols = ["variant"] + [k for k in rows[0].keys() if k != "variant"]
    # Header
    widths = [max(len(str(r.get(c, ""))) for r in rows + [{c: c}]) for c in cols]
    header = " | ".join(c.ljust(w) for c, w in zip(cols, widths))
    sep = "-+-".join("-" * w for w in widths)
    print(header)
    print(sep)
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(w) for c, w in zip(cols, widths)))


def write_csv(rows: List[Dict[str, Any]], out_path: str):
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -------------------------------
# Main
# -------------------------------

def main():
    p = argparse.ArgumentParser(description="Compare GT vs Generated: FFT magnitude/phase and pixel L2, with shifts.")
    p.add_argument("--gt", required=True, help="Path to ground-truth PNG.")
    p.add_argument("--pred", required=True, help="Path to generated PNG.")

    p.add_argument("--magnitude", choices=["l1", "l2"], default="l1", help="Magnitude error variant.")
    p.add_argument("--phase-weight", action="store_true", help="Weight phase error by target magnitude.")
    p.add_argument("--no-phase-weight", dest="phase_weight", action="store_false", help="Disable phase weighting.")
    p.set_defaults(phase_weight=True)
    
    p.add_argument("--shifts", default="0,0 1,0 0,1 2,2",
                   help='Whitespace-separated "dx,dy" pairs. E.g. "0,0 1,0 0,1 2,2"')
    p.add_argument("--subpixel", action="store_true", help="Interpret shifts as sub-pixel (floats) using grid_sample.")

    p.add_argument("--pinches", type=int, help="Number of random pinch/bulge distortions to apply to pred before comparison.")
    
    p.add_argument("--csv", default="", help="Optional path to write CSV with results.")
    args = p.parse_args()

    gt = load_png_as_tensor(args.gt)
    #pred = load_png_as_tensor(args.pred)

    #pred, gt = match_sizes(pred, gt)  # (C,H,W) same size/channels

    if args.pinches:

        # TODO: fare due programmi diversi perch equesto prende come pred la path di output, che per sbaglio non sovvrascriva l'altra eh

        print(f"Applying {args.pinches} random pinch/bulge distortions to prediction before comparison...")
        distorced_gt, rows = evaluate_pinches(
            gt=gt,
            iterations=args.pinches,
            radius_range=(40, 100),
            strenght_range=(-0.4, 0.4)
        )

        print(f"Saving distorted prediction as {args.pred}'...")
        write_tensor_as_png(distorced_gt, args.pred)
 
    else:
    
        print(f"Comparing GT vs Pred with shifts: {args.shifts} (subpixel={args.subpixel})")
        shift_list = parse_shifts(args.shifts, subpixel=args.subpixel)

        rows = evaluate_pair(
            gt=gt,
            pred=pred,
            shifts=shift_list,
            magnitude_variant=args.magnitude,
            phase_weighted_by_mag=args.phase_weight,
            subpixel=args.subpixel,
        )

    print_table(rows)

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_csv(rows, str(out))
        print(f"\nCSV saved to: {out.resolve()}")


if __name__ == "__main__":
    main()