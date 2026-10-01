"""Model training with early stopping and persistence."""

import json
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .model import FixedMLP
from .preprocessing import Scaler
from .sequences import SequenceExample


class ModelTrainer:
    """Train the fixed MLP with early stopping."""
    
    def __init__(self, 
                 model: FixedMLP,
                 scaler: Scaler,
                 learning_rate: float = 0.01,
                 patience: int = 10,
                 seed: int = 42):
        self.model = model
        self.scaler = scaler
        self.learning_rate = learning_rate
        self.patience = patience
        self.seed = seed
        
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Training components
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Early stopping state
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.epochs_no_improve = 0
    
    def train(self,
              train_sequences: List[SequenceExample],
              val_sequences: List[SequenceExample],
              max_epochs: int = 100,
              batch_size: int = 32) -> dict:
        """Train model with early stopping on validation set.
        
        Args:
            train_sequences: Training sequences
            val_sequences: Validation sequences
            max_epochs: Maximum training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history dictionary
        """
        # Prepare data
        train_features, train_targets = self._prepare_data(train_sequences)
        
        # Fit scaler on training data only
        train_features_scaled = self.scaler.fit_transform(train_features.tolist())
        
        # Convert to tensors
        train_X = torch.FloatTensor(train_features_scaled)
        train_y = torch.FloatTensor(train_targets)
        
        # Handle empty validation set
        if len(val_sequences) > 0:
            val_features, val_targets = self._prepare_data(val_sequences)
            val_features_scaled = self.scaler.transform(val_features.tolist())
            val_X = torch.FloatTensor(val_features_scaled)
            val_y = torch.FloatTensor(val_targets)
        else:
            val_X = None
            val_y = None
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'epochs_trained': 0
        }
        
        for epoch in range(max_epochs):
            # Training phase
            self.model.train()
            train_loss = self._train_epoch(train_X, train_y, batch_size)
            
            # Validation phase
            self.model.eval()
            if val_X is not None and val_y is not None:
                val_loss, val_accuracy = self._validate(val_X, val_y)
            else:
                val_loss = train_loss  # Use training loss if no validation data
                val_accuracy = 0.0
            
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            history['epochs_trained'] = epoch + 1
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
            
            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return history
    
    def _prepare_data(self, sequences: List[SequenceExample]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and targets from sequences."""
        features = np.array([seq.features for seq in sequences])
        targets = np.array([seq.target for seq in sequences])
        return features, targets
    
    def prepare_data(self, sequences: List[SequenceExample]) -> Tuple[np.ndarray, np.ndarray]:
        """Public method to extract features and targets from sequences."""
        return self._prepare_data(sequences)
    
    def _train_epoch(self, X: torch.Tensor, y: torch.Tensor, batch_size: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        # Simple batching
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            
            # Ensure output shape matches target shape
            if outputs.shape != batch_y.shape:
                outputs = outputs.squeeze()
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
            
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def _validate(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        """Validate model and return loss and accuracy."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            
            # Ensure output shape matches target shape
            if outputs.shape != y.shape:
                outputs = outputs.squeeze()
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
            
            loss = self.criterion(outputs, y)
            
            # Calculate accuracy
            predictions = (outputs >= 0.5).long()
            accuracy = (predictions == y.long()).float().mean().item()
            
        return loss.item(), accuracy
    
    def save_model(self, model_path: Path, scaler_path: Path) -> None:
        """Save model and scaler to disk."""
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.model.input_dim,
            'hidden_dim': self.model.hidden_dim,
            'seed': self.seed
        }, model_path)
        
        # Save scaler
        self.scaler.save(scaler_path)
    
    @classmethod
    def load_model(cls, model_path: Path, scaler_path: Path, seed: int = 42) -> Tuple[FixedMLP, Scaler]:
        """Load model and scaler from disk."""
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        model = FixedMLP(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            seed=seed
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load scaler
        scaler = Scaler.load(scaler_path)
        
        return model, scaler


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """Calculate classification metrics.
    
    Args:
        predictions: Binary predictions (0=DOWN, 1=UP)
        targets: True targets (0=DOWN, 1=UP)
        
    Returns:
        Dictionary of metrics
    """
    accuracy = np.mean(predictions == targets)
    
    # Calculate per-class metrics
    up_mask = targets == 1
    down_mask = targets == 0
    
    if up_mask.sum() > 0:
        up_accuracy = np.mean(predictions[up_mask] == targets[up_mask])
    else:
        up_accuracy = 0.0
    
    if down_mask.sum() > 0:
        down_accuracy = np.mean(predictions[down_mask] == targets[down_mask])
    else:
        down_accuracy = 0.0
    
    return {
        'accuracy': accuracy,
        'up_accuracy': up_accuracy,
        'down_accuracy': down_accuracy,
        'up_samples': up_mask.sum(),
        'down_samples': down_mask.sum()
    }
