import argparse
import os
import glob
import numpy as np
import sys

# Add root to sys.path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.degradation import degrade_image

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset augmentation")
    parser.add_argument('--gt_dir', type=str, required=True, help="Path to original GT folder")
    parser.add_argument('--out_dir', type=str, required=True, help="Where to save the expanded dataset")
    parser.add_argument('--variants_per_image', type=int, default=5, help="Number of synthetic variants per GT")
    args = parser.parse_args()

    # Create output directories
    out_gt = os.path.join(args.out_dir, "GT")
    out_noisy = os.path.join(args.out_dir, "NoisyLR")
    os.makedirs(out_gt, exist_ok=True)
    os.makedirs(out_noisy, exist_ok=True)

    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, '*.npy')))
    print(f"Found {len(gt_files)} Ground Truth images.")
    if len(gt_files) == 0:
        print("No files found. Please check --gt_dir")
        return
        
    total_generated = 0
    for idx, gt_path in enumerate(gt_files):
        fname = os.path.basename(gt_path)
        base, ext = os.path.splitext(fname)
        
        gt = np.load(gt_path).astype(np.float32)
        
        for v in range(args.variants_per_image):
            # Vary severity between 0.5 (cleaner) to 1.5 (heavier noise)
            severity = np.random.uniform(0.5, 1.5)
            noisy = degrade_image(gt, severity=severity)
            
            # Save files with variant suffix
            new_fname = f"{base}_aug_v{v}{ext}"
            
            np.save(os.path.join(out_gt, new_fname), gt)
            np.save(os.path.join(out_noisy, new_fname), noisy)
            
            total_generated += 1
            
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx+1}/{len(gt_files)} images...")
            
    print(f"Done! Generated {total_generated} paired images in {args.out_dir}")
    print(f"Run cp commands or configure your training script to point to {args.out_dir} alongside your real data.")

if __name__ == "__main__":
    main()
