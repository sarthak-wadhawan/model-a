"""Debug baseline calculation."""

import sys
sys.path.insert(0, 'src')

import numpy as np
from model_a.preprocessing import calculate_majority_baseline

# Test with synthetic data
print("Test 1: Synthetic data")
train_targets = np.array([0, 0, 0, 1, 1])
val_targets = np.array([0, 1, 1, 0])

print(f"train_targets dtype: {train_targets.dtype}")
print(f"train_targets shape: {train_targets.shape}")
print(f"train_targets unique: {np.unique(train_targets, return_counts=True)}")
print(f"val_targets dtype: {val_targets.dtype}")
print(f"val_targets shape: {val_targets.shape}")
print(f"val_targets unique: {np.unique(val_targets, return_counts=True)}")

baseline = calculate_majority_baseline(train_targets, val_targets)
print(f"Baseline: {baseline}")
print(f"Expected: 0.5 (2/4)")

# Test with real corrected label counts
print("\nTest 2: Real corrected counts")
train_targets_real = np.array([0]*284 + [1]*280)
val_targets_real = np.array([0]*66 + [1]*75)

print(f"train_targets dtype: {train_targets_real.dtype}")
print(f"train_targets shape: {train_targets_real.shape}")
print(f"train_targets unique: {np.unique(train_targets_real, return_counts=True)}")
print(f"val_targets dtype: {val_targets_real.dtype}")
print(f"val_targets shape: {val_targets_real.shape}")
print(f"val_targets unique: {np.unique(val_targets_real, return_counts=True)}")

baseline_real = calculate_majority_baseline(train_targets_real, val_targets_real)
print(f"Baseline: {baseline_real}")
print(f"Expected: 0.4681 (66/141)")
