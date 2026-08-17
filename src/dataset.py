"""
dataset.py — KLA Semiconductor Image Restoration Dataset
=========================================================
Handles paired NoisyLR (128x128 float32) and GT (256x256 float32) .npy files.
Augmentation: random horizontal/vertical flip + 90° rotation.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random


class SemconDataset(Dataset):
    """
    Paired dataset: NoisyLR (128x128) -> GT (256x256)
    Files must be named identically (000000.npy, 000001.npy, ...)
    """

    def __init__(self, noisy_dir: str, gt_dir: str = None, augment: bool = True):
        """
        Args:
            noisy_dir: Path to NoisyLR folder.
            gt_dir: Path to GT folder. None for inference-only (test set).
            augment: Apply random flips and rotations.
        """
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir
        self.augment = augment and (gt_dir is not None)

        # Collect filenames
        self.filenames = sorted([
            f for f in os.listdir(noisy_dir) if f.endswith('.npy')
        ])

        if len(self.filenames) == 0:
            raise RuntimeError(f"No .npy files found in {noisy_dir}")

        print(f"[Dataset] Found {len(self.filenames)} files | "
              f"augment={self.augment} | gt={'yes' if gt_dir else 'no'}")

    def __len__(self):
        return len(self.filenames)

    def _load_npy(self, path: str) -> np.ndarray:
        """Load .npy, normalize to [0,1] per-sample via min-max.
        
        GT is already strictly [0,1]. NoisyLR can slightly exceed [0,1] 
        (e.g. up to 1.58 on test set) due to SEM noise characteristics.
        Per-sample min-max normalization preserves the full dynamic range.
        """
        arr = np.load(path).astype(np.float32)
        # Handle uint8 images (stored as 0-255)
        if arr.max() > 2.0:
            arr = arr / 255.0
        # Per-sample min-max normalization -> [0, 1]
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        else:
            arr = np.zeros_like(arr)  # Flat image edge case
        return arr

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """Convert HxW or HxWxC numpy array to CxHxW tensor."""
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]   # Add channel dim
        elif arr.ndim == 3:
            arr = arr.transpose(2, 0, 1)  # HWC -> CHW
        return torch.from_numpy(arr.copy())

    def _augment_pair(self, noisy: np.ndarray, gt: np.ndarray):
        """Apply identical random flips/rotations to both images."""
        # Random horizontal flip
        if random.random() > 0.5:
            noisy = np.fliplr(noisy)
            gt = np.fliplr(gt)
        # Random vertical flip
        if random.random() > 0.5:
            noisy = np.flipud(noisy)
            gt = np.flipud(gt)
        # Random 90-degree rotation (0, 90, 180, 270)
        k = random.randint(0, 3)
        noisy = np.rot90(noisy, k)
        gt = np.rot90(gt, k)
        return noisy, gt

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        noisy_path = os.path.join(self.noisy_dir, fname)
        noisy = self._load_npy(noisy_path)  # 128x128

        if self.gt_dir is not None:
            gt_path = os.path.join(self.gt_dir, fname)
            gt = self._load_npy(gt_path)  # 256x256

            if self.augment:
                noisy, gt = self._augment_pair(noisy, gt)

            return self._to_tensor(noisy), self._to_tensor(gt)
        else:
            # Inference mode
            return self._to_tensor(noisy), fname


def get_dataloaders(
    train_noisy_dir: str,
    train_gt_dir: str,
    val_split: float = 0.1,
    batch_size: int = 16,
    num_workers: int = 4,
    seed: int = 42,
):
    """
    Create train and validation dataloaders from a single dataset directory.
    Returns (train_loader, val_loader).
    """
    import torch
    pin_mem = torch.cuda.is_available()  # Only pin memory if GPU available
    full_dataset = SemconDataset(train_noisy_dir, train_gt_dir, augment=False)
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    rng = torch.Generator()
    rng.manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val], generator=rng
    )

    # Wrap training split with augmentation
    train_ds_aug = _AugmentedSubset(train_ds, train_noisy_dir, train_gt_dir)

    train_loader = DataLoader(
        train_ds_aug, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_mem, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem
    )

    print(f"[DataLoader] Train: {n_train} | Val: {n_val} | Batch: {batch_size} | pin_memory: {pin_mem}")
    return train_loader, val_loader


class _AugmentedSubset(Dataset):
    """Wrapper that re-loads files with augmentation enabled."""
    def __init__(self, subset, noisy_dir, gt_dir):
        self.subset = subset
        self.aug_dataset = SemconDataset(noisy_dir, gt_dir, augment=True)

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        original_idx = self.subset.indices[idx]
        return self.aug_dataset[original_idx]


if __name__ == "__main__":
    import sys
    # Quick sanity check
    DATASET_ROOT = "./Dataset"
    noisy_dir = os.path.join(DATASET_ROOT, "train", "NoisyLR")
    gt_dir = os.path.join(DATASET_ROOT, "train", "GT")

    ds = SemconDataset(noisy_dir, gt_dir, augment=True)
    noisy_t, gt_t = ds[0]
    print(f"NoisyLR shape: {noisy_t.shape}, dtype: {noisy_t.dtype}, range: [{noisy_t.min():.3f}, {noisy_t.max():.3f}]")
    print(f"GT shape:      {gt_t.shape}, dtype: {gt_t.dtype}, range: [{gt_t.min():.3f}, {gt_t.max():.3f}]")

    # Test set (no GT)
    test_noisy_dir = os.path.join(DATASET_ROOT, "NoisyLR")
    test_ds = SemconDataset(test_noisy_dir, gt_dir=None, augment=False)
    noisy_t2, fname = test_ds[0]
    print(f"Test NoisyLR shape: {noisy_t2.shape} | filename: {fname}")
