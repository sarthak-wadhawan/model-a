"""Training-only normalization and scaling."""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler


class Scaler:
    """Wrapper for sklearn StandardScaler with persistence."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, features: List[List[float]]) -> 'Scaler':
        """Fit scaler on training data only."""
        features_array = np.array(features)
        self.scaler.fit(features_array)
        self.is_fitted = True
        return self
    
    def transform(self, features: List[List[float]]) -> np.ndarray:
        """Transform features using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        features_array = np.array(features)
        return self.scaler.transform(features_array)
    
    def fit_transform(self, features: List[List[float]]) -> np.ndarray:
        """Fit and transform in one call (training only)."""
        return self.fit(features).transform(features)
    
    def save(self, path: Path) -> None:
        """Save scaler to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted scaler")
        
        scaler_data = {
            'mean': self.scaler.mean_.tolist(),
            'scale': self.scaler.scale_.tolist(),
            'var': self.scaler.var_.tolist(),
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'w') as f:
            json.dump(scaler_data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'Scaler':
        """Load scaler from disk."""
        with open(path, 'r') as f:
            scaler_data = json.load(f)
        
        scaler = cls()
        scaler.scaler.mean_ = np.array(scaler_data['mean'])
        scaler.scaler.scale_ = np.array(scaler_data['scale'])
        scaler.scaler.var_ = np.array(scaler_data['var'])
        scaler.is_fitted = scaler_data['is_fitted']
        
        return scaler


def prepare_training_data(sequences: List) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features and targets from sequences for training.
    
    Args:
        sequences: List of SequenceExample objects
        
    Returns:
        (features_array, targets_array)
    """
    features = [seq.features for seq in sequences]
    targets = [seq.target for seq in sequences]
    
    return np.array(features), np.array(targets)


def calculate_majority_baseline(train_targets: np.ndarray, val_targets: np.ndarray) -> float:
    """Calculate majority-class baseline accuracy.
    
    Uses training data to determine majority class, then applies to validation.
    
    Args:
        train_targets: Array of binary training targets (0=DOWN, 1=UP)
        val_targets: Array of binary validation targets (0=DOWN, 1=UP)
        
    Returns:
        Accuracy of always predicting the training majority class on validation
    """
    if len(train_targets) == 0 or len(val_targets) == 0:
        return 0.0
    
    # Determine majority class from training data
    majority_class = 1 if np.mean(train_targets) >= 0.5 else 0
    
    # Calculate accuracy on validation using training majority class
    baseline_accuracy = np.mean(val_targets == majority_class)
    
    return baseline_accuracy
