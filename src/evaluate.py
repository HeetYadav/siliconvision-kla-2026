"""
evaluate.py — STANDALONE Evaluation Script (Hackathon Submission)
=================================================================
USAGE:
    python evaluate.py --input_dir <NoisyLR_dir> --output_dir <output_dir>
                       [--gt_dir <GT_dir>]  [--model_path <path_to_pth>]

This script MUST run standalone on a fresh machine with no manual edits.
It handles:
  1. Loading the trained model from a checkpoint
  2. Running inference on all .npy files in input_dir
  3. Saving restored 256x256 .npy files to output_dir
  4. If GT provided: computes and prints PSNR / SSIM / LPIPS + per-image CSV
  5. Speed benchmarking (images/sec, ms/image)

KLA grading criteria addressed:
  - SSIM / PSNR / LPIPS: computed and reported
  - Inference speed: benchmarked and printed
  - Runs on new OOD data: no hardcoded assumptions about filenames
  - Clean standalone script: self-contained, pip-installable deps only
"""

import os
import sys
import argparse
import time
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).parent))
from model import build_model
from metrics import compute_psnr, compute_ssim, compute_lpips


# ------------------------------------------------------------------
# Default paths (edit these for your environment)
# ------------------------------------------------------------------

DEFAULT_MODEL_PATH = str(
    Path(__file__).parent.parent / "best_model.pth"
)

# ------------------------------------------------------------------
# Inference helpers
# ------------------------------------------------------------------

def load_model(model_path: str, device: torch.device, base_ch: int = 32) -> torch.nn.Module:
    """Load model from checkpoint. Handles both full-checkpoint and state-dict formats."""
    model = build_model(base_ch=base_ch)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}\n"
            f"Please run train.py first to generate best_model.pth"
        )

    ckpt = torch.load(model_path, map_location=device)

    # Handle both checkpoint dicts and raw state dicts
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"[Eval] Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
              f"(best PSNR: {ckpt.get('best_psnr', '?'):.2f} dB)")
    else:
        model.load_state_dict(ckpt)
        print(f"[Eval] Loaded state dict from {model_path}")

    model = model.to(device).eval()
    return model


def load_npy(path: str) -> torch.Tensor:
    """Load a .npy file and return a (1, 1, H, W) tensor in [0,1]."""
    arr = np.load(path).astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    if arr.ndim == 2:
        arr = arr[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
    elif arr.ndim == 3:
        arr = arr[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
    return torch.from_numpy(arr)


def save_npy(tensor: torch.Tensor, path: str):
    """Save a (1, 1, H, W) or (1, H, W) tensor as .npy file."""
    arr = tensor.squeeze().cpu().numpy().astype(np.float32)
    np.save(path, arr)


# ------------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------------

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  SemiRestoreNet — Evaluation")
    print(f"  Device: {device}")
    print(f"  Input:  {args.input_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  GT:     {args.gt_dir or 'Not provided (inference-only mode)'}")
    print(f"{'='*60}\n")

    # Load model
    model = load_model(args.model_path, device, base_ch=args.base_ch)

    # Prepare output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Get all .npy files
    input_files = sorted(
        [f for f in os.listdir(args.input_dir) if f.endswith('.npy')]
    )
    if len(input_files) == 0:
        raise RuntimeError(f"No .npy files found in {args.input_dir}")
    print(f"[Eval] Found {len(input_files)} input files")

    # Check GT
    has_gt = (args.gt_dir is not None) and os.path.isdir(args.gt_dir)

    # ---- Inference loop ----
    per_image_results = []
    total_psnr = total_ssim = total_lpips = 0.0
    total_time = 0.0

    for fname in input_files:
        input_path = os.path.join(args.input_dir, fname)
        output_path = os.path.join(args.output_dir, fname)

        # Load input
        noisy = load_npy(input_path).to(device)

        # Inference (timed)
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = model(noisy).clamp(0, 1)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_time += elapsed_ms

        # Save output
        save_npy(pred, output_path)

        # Metrics vs GT
        row = {'filename': fname, 'time_ms': round(elapsed_ms, 2)}
        if has_gt:
            gt_path = os.path.join(args.gt_dir, fname)
            if os.path.exists(gt_path):
                gt = load_npy(gt_path).to(device)
                psnr_val  = compute_psnr(pred, gt)
                ssim_val  = compute_ssim(pred, gt)
                lpips_val = compute_lpips(pred, gt)
                row.update({
                    'psnr':  round(psnr_val, 4),
                    'ssim':  round(ssim_val, 4),
                    'lpips': round(lpips_val, 6),
                })
                total_psnr  += psnr_val
                total_ssim  += ssim_val
                total_lpips += lpips_val

        per_image_results.append(row)

    # ---- Summary ----
    n = len(input_files)
    avg_ms = total_time / n
    throughput = 1000.0 / avg_ms  # images/sec

    print(f"\n{'='*60}")
    print(f"  INFERENCE RESULTS ({n} images)")
    print(f"{'='*60}")
    print(f"  Avg inference time : {avg_ms:.2f} ms/image")
    print(f"  Throughput         : {throughput:.1f} images/sec")

    summary = {
        'n_images': n,
        'avg_inference_ms': round(avg_ms, 2),
        'throughput_img_per_sec': round(throughput, 2),
    }

    if has_gt:
        mean_psnr  = total_psnr  / n
        mean_ssim  = total_ssim  / n
        mean_lpips = total_lpips / n
        print(f"\n  QUALITY METRICS (vs Ground Truth):")
        print(f"  PSNR  (dB, ↑) : {mean_psnr:.4f}")
        print(f"  SSIM  (↑)     : {mean_ssim:.4f}")
        print(f"  LPIPS (↓)     : {mean_lpips:.6f}")
        summary.update({
            'mean_psnr':  round(mean_psnr,  4),
            'mean_ssim':  round(mean_ssim,  4),
            'mean_lpips': round(mean_lpips, 6),
        })

    print(f"\n  Output saved to: {args.output_dir}")
    print(f"{'='*60}\n")

    # ---- Save results ----
    # Per-image CSV
    csv_path = os.path.join(args.output_dir, "results_per_image.csv")
    if per_image_results:
        fieldnames = list(per_image_results[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_image_results)
        print(f"  Per-image CSV : {csv_path}")

    # Summary JSON
    summary_path = os.path.join(args.output_dir, "eval_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON  : {summary_path}")

    return summary


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="SemiRestoreNet Evaluation — KLA Hackathon 2026 Track 1"
    )
    parser.add_argument(
        '--input_dir', type=str, required=True,
        help='Path to folder containing NoisyLR .npy files'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Path to folder where restored .npy files will be saved'
    )
    parser.add_argument(
        '--gt_dir', type=str, default=None,
        help='(Optional) Path to GT folder for metric computation'
    )
    parser.add_argument(
        '--model_path', type=str, default=DEFAULT_MODEL_PATH,
        help=f'Path to model checkpoint (default: {DEFAULT_MODEL_PATH})'
    )
    parser.add_argument(
        '--base_ch', type=int, default=32,
        help='Model base channels (must match training config)'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
