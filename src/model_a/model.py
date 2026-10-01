"""Fixed 20→8→1 MLP model as specified in PRD."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedMLP(nn.Module):
    """Fixed architecture: 20 inputs → 8 hidden (tanh) → 1 output (sigmoid)."""
    
    def __init__(self, input_dim: int = 20, hidden_dim: int = 8, seed: int = 42):
        super().__init__()
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Fixed architecture
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights for reproducibility
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using fixed scheme for reproducibility."""
        # Xavier/Glorot initialization for tanh
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        
        # Initialize output layer
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: 20 → 8 tanh → 1 sigmoid."""
        # First layer: 20 → 8 with tanh activation
        x = self.fc1(x)
        x = torch.tanh(x)
        
        # Output layer: 8 → 1 with sigmoid activation
        x = self.fc2(x)
        x = torch.sigmoid(x)
        
        # Ensure output is always 2D: (batch_size, 1)
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # (batch_size,) -> (batch_size, 1)
        
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probability of UP (P(UP))."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return binary predictions (1=UP, 0=DOWN)."""
        proba = self.predict_proba(x)
        return (proba >= threshold).long()


def create_model(seed: int = 42) -> FixedMLP:
    """Create a fixed MLP model with specified architecture."""
    return FixedMLP(input_dim=20, hidden_dim=8, seed=seed)
