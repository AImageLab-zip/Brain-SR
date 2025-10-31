#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import argparse
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import os

import torch
import torch.nn.functional as F
from PIL import Image
import csv

import math
import torch
import torch.nn.functional as F

import concurrent.futures
import multiprocessing as mp
from dataclasses import dataclass

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


def wave2d_distortion_torch(img: torch.Tensor,
                            amp_x: float, freq_x: float,
                            amp_y: float, freq_y: float,
                            mode: str = "bilinear") -> torch.Tensor:
    """
    Distorsione 2D combinata:
      x' = x + amp_x * sin(2π * y / freq_x)
      y' = y + amp_y * sin(2π * x / freq_y)

    img: (C,H,W) o (B,C,H,W) in [0,1]
    amp_*, freq_* in pixel
    """
    is_batched = (img.dim() == 4)
    if not is_batched:
        img = img.unsqueeze(0)  # (1,C,H,W)

    B, C, H, W = img.shape
    device, dtype = img.device, img.dtype

    # griglie in pixel
    yy = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1).expand(B, 1, H, W)
    xx = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, W).expand(B, 1, H, W)

    dx = amp_x * torch.sin(2 * math.pi * yy / max(freq_x, 1e-6))  # (B,1,H,W)
    dy = amp_y * torch.sin(2 * math.pi * xx / max(freq_y, 1e-6))  # (B,1,H,W)

    # coord sorgente (in pixel)
    srcX = xx + dx
    srcY = yy + dy

    # clamp opzionale per sicurezza (riflessione già gestita da padding_mode)
    # srcX = srcX.clamp(0, W-1)
    # srcY = srcY.clamp(0, H-1)

    # normalizza in [-1,1] per grid_sample (align_corners=True)
    normX = (srcX / (W - 1)) * 2 - 1
    normY = (srcY / (H - 1)) * 2 - 1
    grid = torch.stack((normX, normY), dim=-1)  # (B,1,H,W,2) -> (B,H,W,2)
    grid = grid.squeeze(1)

    warped = F.grid_sample(img, grid, mode=mode, padding_mode="reflection", align_corners=True)
    return warped.squeeze(0) if not is_batched else warped


from torch.nn import functional as F

@torch.no_grad()
def perlin_like_distortion_torch(img: torch.Tensor,
                                 scale: float = 20.0,
                                 sigma: float = 10.0,
                                 mode: str = "bilinear") -> torch.Tensor:
    """
    Distorsione tramite rumore Perlin-like:
      - Si genera un campo random dx, dy.
      - Lo si liscia con un blur gaussiano (sigma).
      - Lo si scala in pixel (scale).
    
    Args:
        img: torch.Tensor (C,H,W) o (B,C,H,W)
        scale: ampiezza massima spostamento in pixel
        sigma: smoothness del campo (blur gaussiano). Più alto = ondulazioni più larghe.
        mode: interpolazione (bilinear o nearest)
    """
    is_batched = (img.dim() == 4)
    if not is_batched:
        img = img.unsqueeze(0)  # (1,C,H,W)

    B, C, H, W = img.shape
    device, dtype = img.device, img.dtype

    # Rumore casuale
    dx = torch.randn(B, 1, H, W, device=device, dtype=dtype)
    dy = torch.randn(B, 1, H, W, device=device, dtype=dtype)

    # Applica blur gaussiano separabile
    def gaussian_blur(x, sigma):
        if sigma <= 0:
            return x
        radius = int(3 * sigma)
        ksize = 2 * radius + 1
        coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel /= kernel.sum()
        kx = kernel.view(1, 1, 1, -1)
        ky = kernel.view(1, 1, -1, 1)
        x = F.conv2d(x, ky, padding=(radius, 0), groups=1)
        x = F.conv2d(x, kx, padding=(0, radius), groups=1)
        return x

    dx = gaussian_blur(dx, sigma) * scale
    dy = gaussian_blur(dy, sigma) * scale

    # griglia base in pixel
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij"
    )
    xx = xx.unsqueeze(0).expand(B, -1, -1)
    yy = yy.unsqueeze(0).expand(B, -1, -1)

    srcX = xx + dx.squeeze(1)
    srcY = yy + dy.squeeze(1)

    # normalizza in [-1,1] per grid_sample
    normX = (srcX / (W - 1)) * 2 - 1
    normY = (srcY / (H - 1)) * 2 - 1
    grid = torch.stack((normX, normY), dim=-1)

    warped = F.grid_sample(img, grid, mode=mode,
                           padding_mode="reflection", align_corners=True)
    return warped.squeeze(0) if not is_batched else warped

### Aggiunti per il wave ripple locale
@torch.no_grad()
def _gaussian_kernel1d(sigma: float, truncate: float = 3.0, device=None, dtype=None):
    if sigma <= 0:
        return torch.tensor([1.0], device=device, dtype=dtype)
    radius = int(truncate * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    return k

@torch.no_grad()
def _gaussian_blur2d(x: torch.Tensor, sigma: float):
    if sigma <= 0:
        return x
    B, C, H, W = x.shape
    k1d = _gaussian_kernel1d(sigma, device=x.device, dtype=x.dtype)
    kx = k1d.view(1, 1, 1, -1)
    ky = k1d.view(1, 1, -1, 1)
    x = F.conv2d(x, ky.repeat(C,1,1,1), padding=(ky.shape[2]//2, 0), groups=C)
    x = F.conv2d(x, kx.repeat(C,1,1,1), padding=(0, kx.shape[3]//2), groups=C)
    return x

@torch.no_grad()
def local_wave_ripple_torch(
    img: torch.Tensor,
    amp_x: float, freq_x: float,
    amp_y: float, freq_y: float,
    patch_size: tuple[int,int] = (50, 50),
    top_left: tuple[int,int] | None = None,
    feather: float = 6.0,   # sigma per sfumare i bordi della patch (px)
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Applica un'onda 2D SOLO dentro una patch (con bordo sfumato).
    img: (C,H,W) o (B,C,H,W) in [0,1]
    amp_*: ampiezze in pixel
    freq_*: periodi in pixel
    patch_size: (ph, pw) in pixel (default 50x50)
    top_left: (y0, x0) opzionale; se None viene campionata random
    feather: sigma gaussiana per ottenere maschera morbida (0 = bordo netto)
    """
    is_batched = (img.dim() == 4)
    if not is_batched:
        img = img.unsqueeze(0)   # (1,C,H,W)
    B, C, H, W = img.shape
    device, dtype = img.device, img.dtype

    ph, pw = patch_size
    ph = min(ph, H)
    pw = min(pw, W)

    if top_left is None:
        y0 = int(torch.randint(0, max(H - ph + 1, 1), (1,)).item())
        x0 = int(torch.randint(0, max(W - pw + 1, 1), (1,)).item())
    else:
        y0, x0 = top_left
        y0 = max(0, min(y0, H - ph))
        x0 = max(0, min(x0, W - pw))

    # griglie in pixel
    yy_full = torch.arange(H, device=device, dtype=dtype).view(1,1,H,1).expand(B,1,H,W)
    xx_full = torch.arange(W, device=device, dtype=dtype).view(1,1,1,W).expand(B,1,H,W)

    # campo di spostamento "onda 2D" (sull'intera immagine)
    dx = amp_x * torch.sin(2 * math.pi * yy_full / max(freq_x, 1e-6))
    dy = amp_y * torch.sin(2 * math.pi * xx_full / max(freq_y, 1e-6))

    # maschera locale (1 dentro patch, 0 fuori) + feather gaussian
    mask = torch.zeros((B, 1, H, W), device=device, dtype=dtype)
    mask[:, :, y0:y0+ph, x0:x0+pw] = 1.0
    if feather > 0:
        mask = _gaussian_blur2d(mask, sigma=feather)
        # normalizza a [0,1] mantenendo il picco=1
        maxv = mask.amax(dim=(-2,-1), keepdim=True).clamp_min(1e-8)
        mask = (mask / maxv).clamp(0, 1)

    dx *= mask
    dy *= mask

    # sorgenti in pixel
    srcX = xx_full + dx
    srcY = yy_full + dy

    # normalizza in [-1,1] per grid_sample (align_corners=True)
    normX = (srcX / (W - 1)) * 2 - 1
    normY = (srcY / (H - 1)) * 2 - 1
    grid = torch.stack((normX, normY), dim=-1).squeeze(1)  # (B,H,W,2)

    warped = F.grid_sample(img, grid, mode=mode, padding_mode="reflection", align_corners=True)
    return warped.squeeze(0) if not is_batched else warped



# -------------------------------
# Evaluation loop (no batches)
# -------------------------------


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


def evaluate_waves(
    gt: torch.Tensor,
    iterations: int,
    amp_x_range: tuple = (2.0, 8.0),
    freq_x_range: tuple = (16.0, 48.0),
    amp_y_range: tuple = (2.0, 8.0),
    freq_y_range: tuple = (16.0, 48.0),
) -> tuple:
    """
    Applica iterativamente la distorsione 2D combined wave, calcola metriche ad ogni step.
    Ritorna (distorted_gt_finale, rows).
    """
    rows: List[Dict[str, Any]] = []

    # baseline (nessuna distorsione)
    base_metrics = spectral_metrics(gt, gt)
    base_px = pixel_mse(gt, gt)
    rows.append({
        "variant": "orig",
        "pixel_mse": base_px,
        **base_metrics
    })

    dist_gt = gt.clone()

    for i in range(iterations):
        amp_x = float(torch.empty(1).uniform_(amp_x_range[0], amp_x_range[1]).item())
        freq_x = float(torch.empty(1).uniform_(freq_x_range[0], freq_x_range[1]).item())
        amp_y = float(torch.empty(1).uniform_(amp_y_range[0], amp_y_range[1]).item())
        freq_y = float(torch.empty(1).uniform_(freq_y_range[0], freq_y_range[1]).item())

        dist_gt = wave2d_distortion_torch(
            dist_gt, amp_x=amp_x, freq_x=freq_x, amp_y=amp_y, freq_y=freq_y
        )

        m = spectral_metrics(dist_gt, gt)
        px = pixel_mse(dist_gt, gt)
        rows.append({
            "variant": f"i{i+1}_ax{amp_x:.2f}_fx{freq_x:.1f}_ay{amp_y:.2f}_fy{freq_y:.1f}",
            "pixel_mse": px,
            **m
        })

    return dist_gt, rows


def evaluate_perlin(
    gt: torch.Tensor,
    iterations: int,
    scale_range: tuple = (10.0, 30.0),
    sigma_range: tuple = (5.0, 20.0),
) -> tuple:
    """
    Applica iterativamente distorsione Perlin-like.
    """
    rows: List[Dict[str, Any]] = []

    # baseline
    base_metrics = spectral_metrics(gt, gt)
    base_px = pixel_mse(gt, gt)
    rows.append({
        "variant": "orig",
        "pixel_mse": base_px,
        **base_metrics
    })

    dist_gt = gt.clone()

    for i in range(iterations):
        scale = float(torch.empty(1).uniform_(*scale_range).item())
        sigma = float(torch.empty(1).uniform_(*sigma_range).item())

        dist_gt = perlin_like_distortion_torch(dist_gt, scale=scale, sigma=sigma)

        m = spectral_metrics(dist_gt, gt)
        px = pixel_mse(dist_gt, gt)
        rows.append({
            "variant": f"i{i+1}_scl{scale:.1f}_sig{sigma:.1f}",
            "pixel_mse": px,
            **m
        })

    return dist_gt, rows


def evaluate_local_wave(
    gt: torch.Tensor,
    iterations: int,
    amp_x_range: tuple = (2.0, 8.0),
    freq_x_range: tuple = (16.0, 48.0),
    amp_y_range: tuple = (2.0, 8.0),
    freq_y_range: tuple = (16.0, 48.0),
    patch_size: tuple[int,int] = (50, 50),
    feather_sigma: float = 6.0,
) -> tuple:
    """
    Applica iterativamente Local Wave Ripple su patch random 50x50 (di default).
    Ritorna (distorted_gt_finale, rows).
    """
    rows: List[Dict[str, Any]] = []

    # baseline
    base_metrics = spectral_metrics(gt, gt)
    base_px = pixel_mse(gt, gt)
    rows.append({
        "variant": "orig",
        "pixel_mse": base_px,
        **base_metrics
    })

    dist_gt = gt.clone()

    for i in range(iterations):
        amp_x = float(torch.empty(1).uniform_(*amp_x_range).item())
        freq_x = float(torch.empty(1).uniform_(*freq_x_range).item())
        amp_y = float(torch.empty(1).uniform_(*amp_y_range).item())
        freq_y = float(torch.empty(1).uniform_(*freq_y_range).item())

        dist_gt = local_wave_ripple_torch(
            dist_gt,
            amp_x=amp_x, freq_x=freq_x,
            amp_y=amp_y, freq_y=freq_y,
            patch_size=patch_size,
            top_left=None,           # random ad ogni iterazione
            feather=feather_sigma,
        )

        m = spectral_metrics(dist_gt, gt)
        px = pixel_mse(dist_gt, gt)
        rows.append({
            "variant": f"i{i+1}_ax{amp_x:.2f}_fx{freq_x:.1f}_ay{amp_y:.2f}_fy{freq_y:.1f}_ps{patch_size[0]}x{patch_size[1]}",
            "pixel_mse": px,
            **m
        })

    return dist_gt, rows


def convert_to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.drop(columns=["variant"], inplace=True)

    # Compute percent change vs original (first row)
    baseline = df.iloc[0]
    for col in df.columns:
        df[f"{col}_pct_change"] = 100.0 * (df[col] - baseline[col]) / (baseline[col] + 1e-8)

    return df


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
# Parallelize
# -------------------------------


def _process_one(img_path, outpath, pinches):
    # minimal prints to avoid stdout lockups
    gt = load_png_as_tensor(img_path)
    
    dist_gt, rows = evaluate_pinches(
        gt=gt,
        iterations=pinches,
        radius_range=(40, 100),
        strenght_range=(-0.4, 0.4),
    )

    #dist_gt, rows = evaluate_waves(
    #    gt=gt,
    #    iterations=pinches,
    #    amp_x_range=(1, 4),
    #    freq_x_range=(50, 100),
    #    amp_y_range=(1, 4),
    #    freq_y_range=(50, 100),
    #)


    #dist_gt, rows = evaluate_perlin(
    #    gt=gt,
    #    iterations=pinches,
    #    scale_range=(0, 1),
    #    sigma_range=(0,0.2),
    #)


    #dist_gt, rows = evaluate_local_wave(
    #    gt=gt,
    #    iterations=pinches,
    #    amp_x_range=(0.5, 2),
    #    freq_x_range=(20, 40),
    #    amp_y_range=(0, 0),
    #    freq_y_range=(40.0, 100.0),
    #    patch_size=(20, 20),
    #    feather_sigma=2.0,  # 0 => bordo netto; >0 => transizione morbida
    #)


    out_name = os.path.basename(img_path)
    write_tensor_as_png(dist_gt, os.path.join(outpath, out_name))

    # return just the light payload
    return rows


def run_simple_processpool(to_process, args, max_workers=None):
    if max_workers is None:
        max_workers = os.cpu_count() or 2

    overall_df = None

    # use spawn to avoid fork-related hangs
    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
    ) as ex:
        # map is simpler than submit/as_completed; chunksize=1 avoids worker starvation
        for rows in ex.map(_process_one, to_process,
                           [args.outpath]*len(to_process),
                           [args.pinches]*len(to_process),
                           chunksize=1):
            df = convert_to_df(rows)
            overall_df = df if overall_df is None else (overall_df + df)

    return overall_df


def run_serial(to_process, args):
    overall_df = None
    for p in to_process:
        rows = _process_one(p, args.outpath, args.pinches)
        df = convert_to_df(rows)
        overall_df = df if overall_df is None else (overall_df + df)
    return overall_df


# -------------------------------
# Main
# -------------------------------

def main():
    p = argparse.ArgumentParser(description="Compare GT vs Generated: FFT magnitude/phase and pixel L2, with shifts.")
    p.add_argument("--gt", required=True, help="Path to ground-truth PNG folder")
    p.add_argument("--outpath", required=True, help="Output path to save distorted images")

    p.add_argument("--magnitude", choices=["l1", "l2"], default="l1", help="Magnitude error variant.")
    p.add_argument("--phase-weight", action="store_true", help="Weight phase error by target magnitude.")
    p.add_argument("--no-phase-weight", dest="phase_weight", action="store_false", help="Disable phase weighting.")
    p.set_defaults(phase_weight=True)
    
    p.add_argument("--pinches", type=int, help="Number of random pinch/bulge distortions to apply to pred before comparison.")

    p.add_argument("--workers", type=int, default=0, help="Number of parallel workers (0=auto).")
    
    p.add_argument("--csv", default="", help="Optional path to write CSV with results.")
    args = p.parse_args()

    to_process = []
    if os.path.isfile(args.gt):
        to_process.append(args.gt)
    elif os.path.isdir(args.gt):
        files = os.listdir(args.gt)
        to_process = [str(os.path.join(args.gt, p)) for p in files if p.lower().endswith('.png')]

    if not to_process:
        print(f"No PNG files found in {args.gt}")
        return
    
    if args.workers <= 1:
        # Serial process
        overall_df = run_serial(to_process, args)
    else:
        # Parallell process
        overall_df = run_simple_processpool(to_process, args, max_workers=15)

    # Average overall metrics
    overall_df /= len(to_process)
    print("\nOverall average metrics:")
    print(overall_df)

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        overall_df.to_csv(out, index=False)
        print(f"\nCSV saved to: {out.resolve()}")


if __name__ == "__main__":
    # one-liner to enforce spawn everywhere (esp. on Linux)
    mp.set_start_method("spawn", force=True)

    # optional: reduce native lib threads (keeps it simple & stable)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    main()