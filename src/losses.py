"""
losses.py — Multi-component loss for semiconductor image restoration
=====================================================================
Loss = α·L1 + β·SSIM + γ·Perceptual(VGG)

Weights justified:
  - L1 (α=0.6): pixel accuracy, fast convergence
  - SSIM (β=0.3): structural similarity, critical for wafer inspection
  - Perceptual/VGG (γ=0.1): texture sharpness, penalizes blurry output

References:
  - Wang et al., "Image Quality Assessment: From Error Visibility to 
    Structural Similarity" (SSIM), IEEE TIP 2004.
  - Johnson et al., "Perceptual Losses for Real-Time Style Transfer", 
    ECCV 2016.
  - Zhao et al., "Loss Functions for Image Restoration with Neural Networks",
    IEEE TCI 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# SSIM Loss
# ------------------------------------------------------------------

def gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Create 2D Gaussian kernel for SSIM computation."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = g.outer(g)
    return kernel / kernel.sum()


class SSIMLoss(nn.Module):
    """
    Differentiable SSIM loss. Returns 1 - SSIM so lower = better.
    Handles single-channel (grayscale) images.
    """
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        kernel = gaussian_kernel(window_size, sigma)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        self.register_buffer('kernel', kernel)
        self.window_size = window_size
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: (B, 1, H, W) in [0, 1]
        Returns: scalar SSIM loss (1 - mean_ssim)
        """
        pad = self.window_size // 2
        kernel = self.kernel.to(pred.device)

        mu_x = F.conv2d(pred,   kernel, padding=pad, groups=1)
        mu_y = F.conv2d(target, kernel, padding=pad, groups=1)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(pred   * pred,   kernel, padding=pad, groups=1) - mu_x2
        sigma_y2 = F.conv2d(target * target, kernel, padding=pad, groups=1) - mu_y2
        sigma_xy = F.conv2d(pred   * target, kernel, padding=pad, groups=1) - mu_xy

        ssim_map = ((2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)) / \
                   ((mu_x2 + mu_y2 + self.C1) * (sigma_x2 + sigma_y2 + self.C2))

        return 1.0 - ssim_map.mean()


# ------------------------------------------------------------------
# Perceptual Loss (VGG16 features)
# ------------------------------------------------------------------

class PerceptualLoss(nn.Module):
    """
    VGG16-based perceptual loss using relu2_2 features.
    Images are converted from 1-channel to 3-channel by repeating.
    
    NOTE: Falls back gracefully to zero if torchvision is not installed.
    """
    def __init__(self):
        super().__init__()
        self.vgg = None
        try:
            import torchvision.models as models
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            # Use only up to relu2_2 (layer index 9)
            self.vgg = nn.Sequential(*list(vgg.features.children())[:10])
            for p in self.vgg.parameters():
                p.requires_grad = False
            self.vgg.eval()
            print("[Loss] Perceptual loss: VGG16 relu2_2 loaded OK")
        except Exception as e:
            print(f"[Loss] WARNING: VGG16 unavailable ({e}), perceptual loss=0")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.vgg is None:
            return torch.tensor(0.0, device=pred.device)
        # Grayscale -> 3 channel
        pred_3 = pred.repeat(1, 3, 1, 1)
        target_3 = target.repeat(1, 3, 1, 1)
        # Move VGG to same device
        self.vgg = self.vgg.to(pred.device)
        feat_pred   = self.vgg(pred_3)
        feat_target = self.vgg(target_3)
        return F.l1_loss(feat_pred, feat_target)


# ------------------------------------------------------------------
# Combined Loss
# ------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """
    Combined = α·L1 + β·SSIM + γ·Perceptual
    
    Default weights empirically tuned for semiconductor SEM restoration:
      α=0.60, β=0.30, γ=0.10
    
    Rationale:
      - High L1 weight for pixel-accurate reconstruction
      - SSIM for structural integrity of circuit features
      - Low perceptual weight to avoid over-texturizing noise
    """
    def __init__(
        self,
        alpha: float = 0.60,   # L1 weight
        beta:  float = 0.30,   # SSIM weight
        gamma: float = 0.10,   # Perceptual weight
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.perceptual_loss = PerceptualLoss()

        print(f"[Loss] CombinedLoss: a={alpha}*L1 + b={beta}*SSIM + g={gamma}*Perceptual")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple:
        """
        Returns: (total_loss, {l1, ssim, perceptual})
        """
        l1   = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        perc = self.perceptual_loss(pred, target)

        total = self.alpha * l1 + self.beta * ssim + self.gamma * perc

        return total, {
            'l1':         l1.item(),
            'ssim':       ssim.item(),
            'perceptual': perc.item(),
            'total':      total.item(),
        }


if __name__ == "__main__":
    loss_fn = CombinedLoss()
    pred   = torch.rand(2, 1, 256, 256)
    target = torch.rand(2, 1, 256, 256)
    total, components = loss_fn(pred, target)
    print(f"Total loss: {total.item():.4f}")
    for k, v in components.items():
        print(f"  {k}: {v:.4f}")
