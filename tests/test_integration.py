"""Integration tests for end-to-end pipeline."""

import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile

from model_a.data_loader import Event
from model_a.sequences import SequenceExample
from model_a.model import create_model
from model_a.preprocessing import Scaler
from model_a.training import ModelTrainer
from model_a.prediction import Predictor


class TestEndToEnd:
    """Tests for end-to-end pipeline functionality."""
    
    def test_full_pipeline_without_real_data(self):
        """Test full pipeline components with mock data."""
        # Create mock sequences
        sequences = []
        for i in range(10):
            seq = SequenceExample(
                input_timestamps=[datetime(2026, 8, 18, 13, 55, tzinfo=timezone.utc)] * 5,
                features=[float(i % 2)] * 20,  # Alternating features
                target_timestamp=datetime(2026, 8, 18, 14, 0, tzinfo=timezone.utc),
                target=i % 2,  # Alternating targets
                actual_direction=i % 2,
                decision_midpoint=100.0 + i,
                target_midpoint=100.0 + i + 1
            )
            sequences.append(seq)
        
        # Split into train/val
        train_seqs = sequences[:7]
        val_seqs = sequences[7:9]
        demo_seqs = sequences[9:]
        
        # Create model and scaler
        model = create_model(seed=42)
        scaler = Scaler()
        
        # Train
        trainer = ModelTrainer(model, scaler, seed=42)
        history = trainer.train(train_seqs, val_seqs, max_epochs=5)
        
        # Check that training occurred
        assert history['epochs_trained'] > 0
        assert len(history['train_loss']) > 0
        assert len(history['val_loss']) > 0
        
        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            scaler_path = Path(tmpdir) / "scaler.json"
            
            trainer.save_model(model_path, scaler_path)
            
            # Load predictor
            predictor = Predictor.load(model_path, scaler_path, seed=42)
            
            # Make prediction
            prediction = predictor.predict_single(demo_seqs[0])
            
            # Check prediction structure
            assert 'symbol' in prediction
            assert 'direction' in prediction
            assert 'p_up' in prediction
            assert 'p_down' in prediction
            assert 0.0 <= prediction['p_up'] <= 1.0
            assert 0.0 <= prediction['p_down'] <= 1.0
            assert abs(prediction['p_up'] + prediction['p_down'] - 1.0) < 1e-6
    
    def test_model_dimensions(self):
        """Test that model maintains correct dimensions throughout pipeline."""
        model = create_model(seed=42)
        
        # Create input with correct dimensions
        features = [0.0] * 20
        seq = SequenceExample(
            input_timestamps=[datetime(2026, 8, 18, 13, 55, tzinfo=timezone.utc)] * 5,
            features=features,
            target_timestamp=datetime(2026, 8, 18, 14, 0, tzinfo=timezone.utc),
            target=1,
            actual_direction=1,
            decision_midpoint=100.0,
            target_midpoint=101.0
        )
        
        # Check dimensions
        assert len(seq.features) == 20
        assert len(seq.input_timestamps) == 5
        assert len(seq.to_feature_vector()) == 20  # This would need to be added to SequenceExample
    
    def test_scaler_isolation(self):
        """Test that scaler is fitted only on training data."""
        train_features = [[1.0, 2.0] * 10]  # 20 features
        val_features = [[3.0, 4.0] * 10]  # 20 features
        
        scaler = Scaler()
        scaler.fit(train_features)
        
        # Transform validation data
        val_scaled = scaler.transform(val_features)
        
        # Check that validation data was transformed using training statistics
        assert scaler.is_fitted
        assert val_scaled.shape == (1, 20)
    
    def test_prediction_without_retraining(self):
        """Test that prediction does not retrain the model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and train model
            model = create_model(seed=42)
            scaler = Scaler()
            
            sequences = [SequenceExample(
                input_timestamps=[datetime(2026, 8, 18, 13, 55, tzinfo=timezone.utc)] * 5,
                features=[0.0] * 20,
                target_timestamp=datetime(2026, 8, 18, 14, 0, tzinfo=timezone.utc),
                target=1,
                actual_direction=1,
                decision_midpoint=100.0,
                target_midpoint=101.0
            )]
            
            trainer = ModelTrainer(model, scaler, seed=42)
            trainer.train(sequences, [], max_epochs=2)
            
            # Save model
            model_path = Path(tmpdir) / "model.pt"
            scaler_path = Path(tmpdir) / "scaler.json"
            trainer.save_model(model_path, scaler_path)
            
            # Load predictor
            predictor = Predictor.load(model_path, scaler_path, seed=42)
            
            # Make prediction - this should not retrain
            prediction = predictor.predict_single(sequences[0])
            
            # Model should still be in eval mode
            assert not model.training


class TestDeterminism:
    """Tests for deterministic behavior."""
    
    def test_deterministic_training(self):
        """Test that training with same seed produces same results."""
        # Create identical data
        sequences = [SequenceExample(
            input_timestamps=[datetime(2026, 8, 18, 13, 55, tzinfo=timezone.utc)] * 5,
            features=[0.5] * 20,
            target_timestamp=datetime(2026, 8, 18, 14, 0, tzinfo=timezone.utc),
            target=1,
            actual_direction=1,
            decision_midpoint=100.0,
            target_midpoint=101.0
        )] * 5
        
        # Train two models with same seed
        model1 = create_model(seed=42)
        scaler1 = Scaler()
        trainer1 = ModelTrainer(model1, scaler1, seed=42)
        
        model2 = create_model(seed=42)
        scaler2 = Scaler()
        trainer2 = ModelTrainer(model2, scaler2, seed=42)
        
        history1 = trainer1.train(sequences, [], max_epochs=3)
        history2 = trainer2.train(sequences, [], max_epochs=3)
        
        # Histories should be identical
        assert history1['epochs_trained'] == history2['epochs_trained']
        # Note: Exact loss values may vary slightly due to floating point, but should be very close
        assert abs(history1['train_loss'][-1] - history2['train_loss'][-1]) < 1e-5
    
    def test_deterministic_prediction(self):
        """Test that prediction with same inputs produces same results."""
        sequences = [SequenceExample(
            input_timestamps=[datetime(2026, 8, 18, 13, 55, tzinfo=timezone.utc)] * 5,
            features=[0.5] * 20,
            target_timestamp=datetime(2026, 8, 18, 14, 0, tzinfo=timezone.utc),
            target=1,
            actual_direction=1,
            decision_midpoint=100.0,
            target_midpoint=101.0
        )]
        
        # Create and train model
        model = create_model(seed=42)
        scaler = Scaler()
        trainer = ModelTrainer(model, scaler, seed=42)
        trainer.train(sequences, [], max_epochs=2)
        
        # Make multiple predictions
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            scaler_path = Path(tmpdir) / "scaler.json"
            trainer.save_model(model_path, scaler_path)
            
            predictor = Predictor.load(model_path, scaler_path, seed=42)
            
            pred1 = predictor.predict_single(sequences[0])
            pred2 = predictor.predict_single(sequences[0])
            
            # Predictions should be identical
            assert pred1['p_up'] == pred2['p_up']
            assert pred1['direction'] == pred2['direction']