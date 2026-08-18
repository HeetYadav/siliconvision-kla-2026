import numpy as np

def degrade_image(gt: np.ndarray, severity: float = 1.0) -> np.ndarray:
    """
    Synthetically degrade a clean Ground Truth (256x256) image to a NoisyLR (128x128).
    Matches KLA Hackathon's noise profile:
    - Downsampling
    - Multiplicative speckle noise
    - Additive Gaussian noise
    
    Args:
        gt: (256, 256) numpy array in range [0, 1].
        severity: Noise multiplier (1.0 = normal, 1.5 = heavily degraded).
        
    Returns:
        (128, 128) float32 numpy array.
    """
    # 1. Downsample from 256x256 to 128x128
    # Using 2x2 average pooling to simulate CCD sensor binning/downsampling
    lr = gt.reshape(128, 2, 128, 2).mean(axis=(1, 3))
    
    # 2. Add Speckle Noise (multiplicative)
    # Sensor noise typically scales with signal intensity
    # Based on our analysis, stddev varies from ~0.04 to ~0.09
    std = np.random.uniform(0.04, 0.09) * severity
    speckle_noise = np.random.normal(0, std * 0.6, size=lr.shape) * lr
    
    # 3. Add Gaussian Noise (additive)
    # Background thermal/read noise
    gaussian_noise = np.random.normal(0, std * 0.8, size=lr.shape)
    
    # Combine (Note: Not strictly clamped to 1.0 as KLA's raw Noisy inputs can exceed 1.0)
    noisy = lr + speckle_noise + gaussian_noise
    
    return noisy.astype(np.float32)
