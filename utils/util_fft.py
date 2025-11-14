import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


class FFTPatchLoss(nn.Module):
    def __init__(self, patch_size=16, stride=16, loss_type='l1', eps=1e-8):
        """
        Patch-wise FFT loss for spatially-aware frequency matching.

        Args:
            patch_size (int): Size of square patches to extract from latent space.
            stride (int): Stride between patches.
            loss_type (str): 'l1' or 'l2' distance between magnitude spectra.
            eps (float): Small constant for numerical stability in sqrt.
        """
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.loss_type = loss_type
        self.eps = eps

    def _extract_patches(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        return x.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride)
        # shape: (B, C, nH, nW, patch_H, patch_W)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Unfold into patches
        input_patches = self._extract_patches(input)
        target_patches = self._extract_patches(target)

        B, C, nH, nW, ph, pw = input_patches.shape
        N = nH * nW  # Number of patches

        # Reshape patches to (B, C, N, ph, pw)
        input_patches = input_patches.contiguous().view(B, C, N, ph, pw)
        target_patches = target_patches.contiguous().view(B, C, N, ph, pw)

        # FFT2 over each patch
        input_fft = torch.fft.fft2(input_patches, dim=(-2, -1), norm='ortho')
        target_fft = torch.fft.fft2(target_patches, dim=(-2, -1), norm='ortho')

        # Magnitude
        input_mag = torch.sqrt(input_fft.real ** 2 + input_fft.imag ** 2 + self.eps)
        target_mag = torch.sqrt(target_fft.real ** 2 + target_fft.imag ** 2 + self.eps)

        # Compute spectral distance
        if self.loss_type == 'l1':
            loss = torch.abs(input_mag - target_mag)
        elif self.loss_type == 'l2':
            loss = (input_mag - target_mag) ** 2

        # Mean over patch pixels: (B, C, N)
        loss = loss.view(B, C, N, -1).mean(dim=-1)

        # Mean over channels and patches: (B,)
        loss = loss.mean(dim=(1, 2))

        return loss
        

def generate_fft_loss_func(fft_conf: dict):
    
    def fft_loss_func(input, target):
        """
        Computes FFT patch loss based on the provided configuration.

        Args:
            input (torch.Tensor): Input tensor of shape (B, C, H, W).
            target (torch.Tensor): Target tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Computed FFT patch loss.
        """
        losses = []
        for conf in fft_conf:
            patch_size = conf.get('patch_size')
            patch_stride = conf.get('patch_stride')
            loss = FFTPatchLoss(patch_size=patch_size, stride=patch_stride)
            losses.append(loss(input, target))
        
        return sum(losses) / len(losses) if losses else torch.tensor(0.0)

    return fft_loss_func