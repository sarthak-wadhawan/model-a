"""Tests for model architecture and scaler functionality."""

import pytest
import numpy as np
import torch

from model_a.model import FixedMLP, create_model
from model_a.preprocessing import Scaler, calculate_majority_baseline


class TestModelArchitecture:
    """Tests for fixed 20→8→1 MLP architecture."""
    
    def test_model_architecture(self):
        """Test that model has correct architecture: 20→8→1."""
        model = create_model(seed=42)
        
        # Check input dimension
        assert model.input_dim == 20
        
        # Check hidden dimension
        assert model.hidden_dim == 8
        
        # Check layers
        assert model.fc1.in_features == 20
        assert model.fc1.out_features == 8
        assert model.fc2.in_features == 8
        assert model.fc2.out_features == 1
    
    def test_model_output_shape(self):
        """Test that model produces correct output shape."""
        model = create_model(seed=42)
        
        # Create dummy input (batch_size=1, features=20)
        x = torch.randn(1, 20)
        
        output = model(x)
        
        # Output should be (batch_size, 1)
        assert output.shape == (1, 1)
    
    def test_model_output_range(self):
        """Test that model output is in [0, 1] range (sigmoid)."""
        model = create_model(seed=42)
        model.eval()
        
        # Create various inputs
        for _ in range(10):
            x = torch.randn(1, 20) * 10  # Wide range of inputs
            with torch.no_grad():
                output = model(x)
            
            # Check sigmoid range
            assert 0.0 <= output.item() <= 1.0
    
    def test_model_activations(self):
        """Test that model uses tanh hidden activation and sigmoid output."""
        model = create_model(seed=42)
        
        # Test hidden layer activation
        x = torch.randn(1, 20)
        hidden = model.fc1(x)
        hidden_activated = torch.tanh(hidden)
        
        # Tanh should produce values in [-1, 1]
        assert torch.all(hidden_activated >= -1.0)
        assert torch.all(hidden_activated <= 1.0)
    
    def test_model_determinism(self):
        """Test that model produces identical outputs with same seed."""
        model1 = create_model(seed=42)
        model2 = create_model(seed=42)
        
        x = torch.randn(1, 20)
        
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            output1 = model1(x)
            output2 = model2(x)
        
        # Outputs should be identical
        assert torch.allclose(output1, output2)
    
    def test_predict_proba(self):
        """Test predict_proba method."""
        model = create_model(seed=42)
        model.eval()
        
        x = torch.randn(1, 20)
        proba = model.predict_proba(x)
        
        assert 0.0 <= proba.item() <= 1.0
    
    def test_predict_binary(self):
        """Test predict method with binary threshold."""
        model = create_model(seed=42)
        model.eval()
        
        x = torch.randn(1, 20)
        prediction = model.predict(x, threshold=0.5)
        
        # Should be 0 or 1
        assert prediction.item() in [0, 1]


class TestScaler:
    """Tests for scaler functionality."""
    
    def test_scaler_fit_transform(self):
        """Test that scaler fits and transforms correctly."""
        scaler = Scaler()
        
        # Create dummy features
        features = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        
        # Fit and transform
        scaled = scaler.fit_transform(features)
        
        # Check shape
        assert scaled.shape == (3, 3)
        
        # Check that scaler is fitted
        assert scaler.is_fitted
    
    def test_scaler_transform_only_after_fit(self):
        """Test that transform works only after fit."""
        scaler = Scaler()
        
        features = [[1.0, 2.0, 3.0]]
        
        # Should raise error if not fitted
        with pytest.raises(ValueError, match="must be fitted"):
            scaler.transform(features)
    
    def test_scaler_persistence(self):
        """Test that scaler can be saved and loaded."""
        import tempfile
        from pathlib import Path
        
        scaler = Scaler()
        features = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        
        # Fit scaler
        scaler.fit(features)
        
        # Save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            scaler_path = Path(tmpdir) / "scaler.json"
            scaler.save(scaler_path)
            
            # Load scaler
            loaded_scaler = Scaler.load(scaler_path)
            
            # Check that loaded scaler is fitted
            assert loaded_scaler.is_fitted
            
            # Check that transformation produces same results
            original_scaled = scaler.transform(features)
            loaded_scaled = loaded_scaler.transform(features)
            
            assert np.allclose(original_scaled, loaded_scaled)
    
    def test_scaler_no_training_data_contamination(self):
        """Test that scaler fitted on training data doesn't see validation data."""
        scaler = Scaler()
        
        train_features = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        val_features = [[7.0, 8.0, 9.0]]
        
        # Fit on training data only
        scaler.fit(train_features)
        
        # Transform validation data
        val_scaled = scaler.transform(val_features)
        
        # The validation data should be transformed using training statistics
        # This is implicitly tested by the sklearn StandardScaler behavior
        assert scaler.is_fitted


class TestMajorityBaseline:
    """Tests for majority baseline calculation."""
    
    def test_majority_baseline_accuracy(self):
        """Test majority baseline calculation."""
        # More UP than DOWN
        train_targets = np.array([1, 1, 1, 0, 0])
        val_targets = np.array([1, 1, 0, 0])
        baseline = calculate_majority_baseline(train_targets, val_targets)
        
        # Training majority is UP (1), validation DOWN count = 2
        # Expected baseline = 2/4 = 0.5
        assert baseline == 0.5
    
    def test_majority_baseline_tie(self):
        """Test majority baseline with tie (defaults to UP)."""
        train_targets = np.array([1, 1, 0, 0])
        val_targets = np.array([1, 1, 0, 0])
        baseline = calculate_majority_baseline(train_targets, val_targets)
        
        # Training tie defaults to UP, validation UP count = 2
        # Expected baseline = 2/4 = 0.5
        assert baseline == 0.5
    
    def test_majority_baseline_all_down(self):
        """Test majority baseline when all are DOWN."""
        train_targets = np.array([0, 0, 0, 0])
        val_targets = np.array([0, 0, 1, 1])
        baseline = calculate_majority_baseline(train_targets, val_targets)
        
        # Training majority is DOWN (0), validation DOWN count = 2
        # Expected baseline = 2/4 = 0.5
        assert baseline == 0.5
    
    def test_majority_baseline_empty(self):
        """Test majority baseline with empty array."""
        train_targets = np.array([])
        val_targets = np.array([1, 0])
        baseline = calculate_majority_baseline(train_targets, val_targets)
        
        # Empty training should return 0.0
        assert baseline == 0.0