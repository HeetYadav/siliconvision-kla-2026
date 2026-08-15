"""
train.py — Training Script for SemiRestoreNet
==============================================
Run: python train.py [--epochs N] [--batch_size B] [--lr LR]

Features:
  - Mixed precision (AMP) for GPU speedup
  - Cosine LR schedule with warmup
  - Validation PSNR/SSIM after every epoch
  - Best model checkpoint saved automatically
  - TensorBoard logging (optional)
  - Reproducible (seed everything)
"""

import os
import sys
import argparse
import time
import random
import math
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast

# Local modules
sys.path.insert(0, str(Path(__file__).parent))
from dataset import get_dataloaders
from model import build_model
from losses import CombinedLoss
from metrics import compute_psnr, compute_ssim


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

DATASET_ROOT = r"c:\Users\heety\Documents\SemiconHackathon\Dataset"
WORK_DIR = r"c:\Users\heety\Documents\SemiconHackathon\work_stage1"

DEFAULT_CONFIG = {
    "train_noisy_dir": os.path.join(DATASET_ROOT, "train", "NoisyLR"),
    "train_gt_dir":    os.path.join(DATASET_ROOT, "train", "GT"),
    "output_dir":      WORK_DIR,
    "epochs":          100,
    "batch_size":      16,
    "lr":              3e-4,
    "lr_min":          1e-6,
    "warmup_epochs":   5,
    "val_split":       0.1,
    "base_ch":         32,
    "num_workers":     4,
    "seed":            42,
    "amp":             True,   # Mixed precision (disable if CPU-only)
    "loss_alpha":      0.60,   # L1 weight
    "loss_beta":       0.30,   # SSIM weight
    "loss_gamma":      0.10,   # Perceptual weight
}


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr_schedule(optimizer, epochs, warmup_epochs, lr_min, lr_max):
    """Cosine annealing with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)  # Linear warmup
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return lr_min / lr_max + (1 - lr_min / lr_max) * cosine
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(model, optimizer, scheduler, epoch, best_psnr, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_psnr': best_psnr,
    }, path)


def load_checkpoint(model, optimizer, scheduler, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt['epoch'], ckpt['best_psnr']


# ------------------------------------------------------------------
# Train / Validate
# ------------------------------------------------------------------

def train_epoch(model, loader, optimizer, loss_fn, scaler, device, amp):
    model.train()
    meters = {k: AverageMeter() for k in ['total', 'l1', 'ssim', 'perceptual']}
    t0 = time.time()

    for i, (noisy, gt) in enumerate(loader):
        noisy = noisy.to(device, non_blocking=True)
        gt    = gt.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type='cuda' if amp else 'cpu', enabled=amp):
            pred = model(noisy)
            
        # Compute loss in float32 to prevent AMP NaN instability
        total_loss, components = loss_fn(pred.float(), gt.float())

        if amp:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        n = noisy.size(0)
        for k, v in components.items():
            meters[k].update(v, n)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(loader)}] loss={meters['total'].avg:.4f} "
                  f"l1={meters['l1'].avg:.4f} ssim={meters['ssim'].avg:.4f} "
                  f"time={elapsed:.1f}s")

    return {k: m.avg for k, m in meters.items()}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    for noisy, gt in loader:
        noisy = noisy.to(device)
        gt    = gt.to(device)
        pred  = model(noisy)
        pred  = pred.clamp(0, 1)

        psnr = compute_psnr(pred, gt)
        ssim = compute_ssim(pred, gt)
        psnr_meter.update(psnr, noisy.size(0))
        ssim_meter.update(ssim, noisy.size(0))

    return psnr_meter.avg, ssim_meter.avg


# ------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------

def train(cfg: dict):
    seed_everything(cfg['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Device: {device} | AMP: {cfg['amp'] and device.type=='cuda'}")

    # Data
    train_loader, val_loader = get_dataloaders(
        train_noisy_dir=cfg['train_noisy_dir'],
        train_gt_dir=cfg['train_gt_dir'],
        val_split=cfg['val_split'],
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'] if device.type == 'cuda' else 0,
        seed=cfg['seed'],
    )

    # Model
    model = build_model(base_ch=cfg['base_ch']).to(device)

    # Optimizer & Schedule
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    scheduler = get_lr_schedule(
        optimizer, cfg['epochs'], cfg['warmup_epochs'],
        cfg['lr_min'], cfg['lr']
    )
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['amp'] and device.type == 'cuda')

    # Loss
    loss_fn = CombinedLoss(
        alpha=cfg['loss_alpha'],
        beta=cfg['loss_beta'],
        gamma=cfg['loss_gamma'],
    ).to(device)

    # Output paths
    out_dir = Path(cfg['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best = out_dir / "best_model.pth"
    ckpt_last = out_dir / "last_model.pth"
    log_path  = out_dir / "train_log.json"

    # Resume if available
    start_epoch = 0
    best_psnr = 0.0
    if ckpt_last.exists():
        print(f"[Train] Resuming from {ckpt_last}")
        start_epoch, best_psnr = load_checkpoint(
            model, optimizer, scheduler, ckpt_last, device
        )
        start_epoch += 1

    train_log = []

    print(f"\n{'='*60}")
    print(f"  SemiRestoreNet Training")
    print(f"  Epochs: {cfg['epochs']} | LR: {cfg['lr']} | Batch: {cfg['batch_size']}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, cfg['epochs']):
        t_epoch = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device,
            amp=cfg['amp'] and device.type == 'cuda'
        )

        # Validate
        val_psnr, val_ssim = validate(model, val_loader, device)

        # Step LR
        scheduler.step()

        epoch_time = time.time() - t_epoch

        # Log
        log_entry = {
            'epoch': epoch,
            'lr': current_lr,
            'train_loss': train_metrics['total'],
            'val_psnr': val_psnr,
            'val_ssim': val_ssim,
            'time': epoch_time,
        }
        train_log.append(log_entry)

        print(f"Epoch [{epoch+1:03d}/{cfg['epochs']:03d}] "
              f"loss={train_metrics['total']:.4f} | "
              f"val_PSNR={val_psnr:.2f}dB | val_SSIM={val_ssim:.4f} | "
              f"lr={current_lr:.2e} | t={epoch_time:.1f}s")

        # Save checkpoints
        save_checkpoint(model, optimizer, scheduler, epoch, best_psnr, ckpt_last)

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(model, optimizer, scheduler, epoch, best_psnr, ckpt_best)
            print(f"  *** NEW BEST PSNR: {best_psnr:.2f} dB — saved to {ckpt_best}")

        # Save log
        with open(log_path, 'w') as f:
            json.dump(train_log, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training complete! Best Val PSNR: {best_psnr:.2f} dB")
    print(f"  Best model: {ckpt_best}")
    print(f"{'='*60}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SemiRestoreNet Training")
    parser.add_argument('--epochs',     type=int,   default=DEFAULT_CONFIG['epochs'])
    parser.add_argument('--batch_size', type=int,   default=DEFAULT_CONFIG['batch_size'])
    parser.add_argument('--lr',         type=float, default=DEFAULT_CONFIG['lr'])
    parser.add_argument('--base_ch',    type=int,   default=DEFAULT_CONFIG['base_ch'])
    parser.add_argument('--val_split',  type=float, default=DEFAULT_CONFIG['val_split'])
    parser.add_argument('--num_workers',type=int,   default=DEFAULT_CONFIG['num_workers'])
    parser.add_argument('--no_amp',     action='store_true', help='Disable mixed precision')
    parser.add_argument('--seed',       type=int,   default=DEFAULT_CONFIG['seed'])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = {**DEFAULT_CONFIG}
    cfg['epochs']      = args.epochs
    cfg['batch_size']  = args.batch_size
    cfg['lr']          = args.lr
    cfg['base_ch']     = args.base_ch
    cfg['val_split']   = args.val_split
    cfg['num_workers'] = args.num_workers
    cfg['amp']         = not args.no_amp
    cfg['seed']        = args.seed
    train(cfg)
