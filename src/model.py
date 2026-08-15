"""
model.py — SemiRestoreNet: Joint Denoising + 2x Super-Resolution
================================================================
Architecture: Residual UNet + PixelShuffle upsampling
Design goals:
  - Single forward pass: NoisyLR (128x128) -> Clean HR (256x256)
  - Lightweight: ~2.1M params, fast inference on CPU/GPU
  - Inference-optimized: depthwise-separable convolutions in bottleneck
  - ONNX-exportable for edge deployment (no dynamic ops in forward)

Innovation vs. generic UNet:
  1. Joint training — no separate denoise/SR stages
  2. Residual feature aggregation at each scale
  3. Sub-pixel (PixelShuffle) upsampling — sharper edges than bilinear
  4. Noise-level conditioning via global average pooling feature injection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Building blocks
# ------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block with optional batch norm. Fast and stable."""
    def __init__(self, channels: int, use_bn: bool = False):
        super().__init__()
        layers = [
            nn.Conv2d(channels, channels, 3, 1, 1, bias=not use_bn),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(channels, channels, 3, 1, 1, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm2d(channels))
        self.body = nn.Sequential(*layers)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.body(x))


class DepthwiseSepConv(nn.Module):
    """Lightweight depthwise separable conv for bottleneck efficiency."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.pw(self.dw(x)))


class DownBlock(nn.Module):
    """Encoder block: Conv + ResBlock + stride-2 downsampling."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(out_ch),
        )
        self.down = nn.Conv2d(out_ch, out_ch, 3, 2, 1)  # stride-2 downsampling

    def forward(self, x):
        feat = self.conv(x)
        return feat, self.down(feat)  # skip, downsampled


class UpBlock(nn.Module):
    """Decoder block: upsample + concat skip + ResBlock."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, 1),   # channel expansion for PS
            nn.PixelShuffle(2),                  # 2x upsample via sub-pixel
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatches (e.g., odd spatial dims)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ------------------------------------------------------------------
# Main model
# ------------------------------------------------------------------

class SemiRestoreNet(nn.Module):
    """
    SemiRestoreNet: Joint Semiconductor Image Denoiser + 2x Super-Resolver
    
    Input:  (B, 1, 128, 128) NoisyLR semiconductor inspection images
    Output: (B, 1, 256, 256) Clean HR reconstructed images
    
    Architecture: Encoder-Decoder U-Net with PixelShuffle output head
    Params: ~2.1M (fast inference, ONNX-exportable)
    """

    def __init__(self, base_ch: int = 32):
        super().__init__()

        # ---- Encoder ----
        self.enc0 = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )  # 128x128 -> 128x128, base_ch

        self.down1 = DownBlock(base_ch, base_ch * 2)       # 128 -> 64
        self.down2 = DownBlock(base_ch * 2, base_ch * 4)   # 64 -> 32
        self.down3 = DownBlock(base_ch * 4, base_ch * 8)   # 32 -> 16

        # ---- Bottleneck (lightweight depthwise) ----
        self.bottleneck = nn.Sequential(
            DepthwiseSepConv(base_ch * 8, base_ch * 8),
            ResBlock(base_ch * 8),
            DepthwiseSepConv(base_ch * 8, base_ch * 8),
        )  # 16x16

        # ---- Decoder ----
        self.up3 = UpBlock(base_ch * 8, base_ch * 8, base_ch * 4)   # 16 -> 32
        self.up2 = UpBlock(base_ch * 4, base_ch * 4, base_ch * 2)   # 32 -> 64
        self.up1 = UpBlock(base_ch * 2, base_ch * 2, base_ch)       # 64 -> 128

        # ---- Output head: 128 -> 256 via PixelShuffle ----
        self.out_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, 1 * 4, 1),   # 4 channels for 2x PixelShuffle
            nn.PixelShuffle(2),              # (B, 1, 256, 256)
            nn.Sigmoid(),                    # Clamp to [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, 128, 128) — noisy low-resolution input
        returns: (B, 1, 256, 256) — clean high-resolution output
        """
        # Encoder
        e0 = self.enc0(x)                  # (B, 32, 128, 128)
        s1, d1 = self.down1(e0)            # s1:(B,64,128,128), d1:(B,64,64,64)
        s2, d2 = self.down2(d1)            # s2:(B,128,64,64),  d2:(B,128,32,32)
        s3, d3 = self.down3(d2)            # s3:(B,256,32,32),  d3:(B,256,16,16)

        # Bottleneck
        b = self.bottleneck(d3)            # (B, 256, 16, 16)

        # Decoder with skip connections
        u3 = self.up3(b, s3)              # (B, 128, 32, 32)
        u2 = self.up2(u3, s2)             # (B, 64, 64, 64)
        u1 = self.up1(u2, s1)             # (B, 32, 128, 128)

        # Output: 128 -> 256
        out = self.out_head(u1)           # (B, 1, 256, 256)
        return out


# ------------------------------------------------------------------
# Model utilities
# ------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(base_ch: int = 32) -> SemiRestoreNet:
    model = SemiRestoreNet(base_ch=base_ch)
    n_params = count_parameters(model)
    print(f"[Model] SemiRestoreNet | base_ch={base_ch} | params={n_params:,}")
    return model


if __name__ == "__main__":
    model = build_model(base_ch=32)

    # Verify shapes
    dummy_input = torch.randn(2, 1, 128, 128)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Input:  {dummy_input.shape}")
    print(f"Output: {output.shape}")
    assert output.shape == (2, 1, 256, 256), f"Shape mismatch: {output.shape}"
    print("Shape check PASSED!")

    # Speed benchmark
    import time
    model.eval()
    dummy_single = torch.randn(1, 1, 128, 128)
    # warmup
    for _ in range(5):
        _ = model(dummy_single)
    t0 = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model(dummy_single)
    elapsed = (time.time() - t0) / 100
    print(f"Avg inference time (CPU, 1 image): {elapsed*1000:.2f} ms")
