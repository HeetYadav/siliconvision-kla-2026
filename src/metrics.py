"""
metrics.py — PSNR, SSIM, LPIPS for semiconductor image restoration evaluation
==============================================================================
These functions are used during training validation AND in the final
evaluate.py submission script.

All metrics operate on tensors of shape (B, 1, H, W) in [0,1].
"""

import torch
import torch.nn.functional as F
import numpy as np


# ------------------------------------------------------------------
# PSNR
# ------------------------------------------------------------------

def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio.
    Higher is better. Typical good range: 30-45 dB for semiconductor SR.
    
    Args:
        pred, target: (B, 1, H, W) tensors in [0,1]
    Returns:
        Mean PSNR across batch (float, dB)
    """
    with torch.no_grad():
        mse = F.mse_loss(pred, target, reduction='none')
        mse = mse.view(mse.size(0), -1).mean(dim=1)  # Per-image MSE
        psnr = 10 * torch.log10(max_val ** 2 / (mse + 1e-8))
        return psnr.mean().item()


# ------------------------------------------------------------------
# SSIM
# ------------------------------------------------------------------

def _gaussian_kernel_metric(size: int = 11, sigma: float = 1.5, device='cpu') -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = g.outer(g)
    return (kernel / kernel.sum()).unsqueeze(0).unsqueeze(0)


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """
    Structural Similarity Index.
    Higher is better. Range [0,1]. Target >= 0.90 for good restoration.
    
    Args:
        pred, target: (B, 1, H, W) tensors in [0,1]
    Returns:
        Mean SSIM across batch (float)
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    pad = window_size // 2
    kernel = _gaussian_kernel_metric(window_size, sigma, device=pred.device)

    with torch.no_grad():
        mu_x = F.conv2d(pred,   kernel, padding=pad)
        mu_y = F.conv2d(target, kernel, padding=pad)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sig_x2 = F.conv2d(pred * pred,     kernel, padding=pad) - mu_x2
        sig_y2 = F.conv2d(target * target, kernel, padding=pad) - mu_y2
        sig_xy = F.conv2d(pred * target,   kernel, padding=pad) - mu_xy

        num = (2 * mu_xy + C1) * (2 * sig_xy + C2)
        den = (mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2)
        ssim_map = num / (den + 1e-8)

        return ssim_map.mean().item()


# ------------------------------------------------------------------
# LPIPS (Learned Perceptual Image Patch Similarity)
# ------------------------------------------------------------------

class LPIPSMetric:
    """
    LPIPS using VGG16 features (lower = better, range [0,1]).
    Lazy-loaded to avoid import overhead at train time.
    """
    _instance = None

    @classmethod
    def get(cls, device):
        if cls._instance is None:
            try:
                import torchvision.models as models
                vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
                cls._instance = torch.nn.Sequential(
                    *list(vgg.features.children())[:16]
                ).eval()
                for p in cls._instance.parameters():
                    p.requires_grad = False
            except Exception as e:
                print(f"[Metrics] LPIPS unavailable: {e}")
                cls._instance = None
        if cls._instance is not None:
            cls._instance = cls._instance.to(device)
        return cls._instance


def compute_lpips(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Approximate LPIPS using VGG16 relu3_3 features.
    Lower is better. Falls back to MSE-proxy if VGG unavailable.
    
    Args:
        pred, target: (B, 1, H, W) in [0,1]
    Returns:
        Mean LPIPS across batch (float)
    """
    vgg = LPIPSMetric.get(pred.device)
    if vgg is None:
        # Fallback: use L1 as proxy
        return F.l1_loss(pred, target).item()

    with torch.no_grad():
        # 1-channel -> 3-channel
        pred_3   = pred.repeat(1, 3, 1, 1)
        target_3 = target.repeat(1, 3, 1, 1)
        f_pred   = vgg(pred_3)
        f_target = vgg(target_3)
        return F.mse_loss(f_pred, f_target).item()


# ------------------------------------------------------------------
# Full metric suite
# ------------------------------------------------------------------

def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Compute PSNR, SSIM, and LPIPS for a batch.
    Returns dict with scalar values.
    """
    pred   = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    return {
        'psnr':  compute_psnr(pred, target),
        'ssim':  compute_ssim(pred, target),
        'lpips': compute_lpips(pred, target),
    }


if __name__ == "__main__":
    # Quick sanity check
    pred   = torch.rand(4, 1, 256, 256)
    target = torch.rand(4, 1, 256, 256)
    metrics = compute_all_metrics(pred, target)
    print("Metrics on random tensors:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Perfect reconstruction test
    pred_perfect = target.clone()
    metrics_perfect = compute_all_metrics(pred_perfect, target)
    print("\nMetrics on perfect reconstruction:")
    for k, v in metrics_perfect.items():
        print(f"  {k}: {v:.4f}")
    assert metrics_perfect['ssim'] > 0.999, "SSIM should be ~1.0 for identical images"
    assert metrics_perfect['psnr'] > 60,   "PSNR should be very high for identical images"
    print("Sanity checks PASSED!")
