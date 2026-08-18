import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# ==========================================
# 1. Model Architecture
# ==========================================
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.sg = SimpleGate()
        
        ffn_channel = FFN_Expand * c
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = nn.InstanceNorm2d(c)
        self.norm2 = nn.InstanceNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma

class NAFNetLarge(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=64):
        super().__init__()
        self.intro = nn.Conv2d(in_ch, base_ch, 3, 1, 1)
        
        # Level 1
        self.enc1 = nn.Sequential(*[NAFBlock(base_ch) for _ in range(2)])
        self.down1 = nn.Conv2d(base_ch, base_ch*2, 2, 2)
        
        # Level 2
        self.enc2 = nn.Sequential(*[NAFBlock(base_ch*2) for _ in range(2)])
        self.down2 = nn.Conv2d(base_ch*2, base_ch*4, 2, 2)
        
        # Level 3 
        self.enc3 = nn.Sequential(*[NAFBlock(base_ch*4) for _ in range(4)])
        self.down3 = nn.Conv2d(base_ch*4, base_ch*8, 2, 2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(*[NAFBlock(base_ch*8) for _ in range(8)])
        
        # Level 3 Up 
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, 2)
        self.dec3 = nn.Sequential(*[NAFBlock(base_ch*4) for _ in range(2)])
        
        # Level 2 Up
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, 2)
        self.dec2 = nn.Sequential(*[NAFBlock(base_ch*2) for _ in range(2)])
        
        # Level 1 Up
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, 2)
        self.dec1 = nn.Sequential(*[NAFBlock(base_ch) for _ in range(2)])
        
        # Super-resolution path (2x) using PixelShuffle
        self.up_sr = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 4, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.outro = nn.Conv2d(base_ch, out_ch, 3, 1, 1)

    def forward(self, x):
        x = self.intro(x)
        
        s1 = self.enc1(x)
        x = self.down1(s1)
        
        s2 = self.enc2(x)
        x = self.down2(s2)
        
        s3 = self.enc3(x)
        x = self.down3(s3)
        
        x = self.bottleneck(x)
        
        x = self.up3(x) + s3
        x = self.dec3(x)
        
        x = self.up2(x) + s2
        x = self.dec2(x)
        
        x = self.up1(x) + s1
        x = self.dec1(x)
        
        x = self.up_sr(x)
        x = self.outro(x)
        return torch.sigmoid(x)

# ==========================================
# 2. Inference Logic
# ==========================================
def run_inference(input_dir, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dynamically find the model in the 'models' folder relative to this script
    script_dir = Path(__file__).resolve().parent
    model_path = script_dir / 'models' / 'best_model.pth'

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}. Please ensure 'best_model.pth' is inside the 'models' folder.")

    # Create output directory and enforce a dedicated subfolder
    out_dir = Path(output_dir) / "restored_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory ready at: {out_dir}")

    # Load Model
    model = NAFNetLarge(base_ch=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Get all .npy files in the input directory
    in_dir = Path(input_dir)
    files = sorted(in_dir.glob("*.npy"))
    
    if len(files) == 0:
        print(f"No .npy files found in {in_dir}")
        return

    print(f"Starting inference on {len(files)} files...")

    # Fast Inference Loop
    with torch.inference_mode():
        for file_path in tqdm(files, desc="Processing Images"):
            # Load and normalize
            arr = np.load(file_path).astype(np.float32)
            if arr.max() > 2.0:
                arr /= 255.0
                
            # Prepare tensor
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
            
            # Predict with mixed precision for speed
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                pred = model(tensor)
                
            # Format output and save
            pred_arr = pred.squeeze().cpu().numpy()
            np.save(out_dir / file_path.name, pred_arr)

    print("Inference complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on NoisyLR images.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input directory containing NoisyLR .npy files")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory for restored images")
    
    args = parser.parse_args()
    
    run_inference(args.input_dir, args.output_dir)