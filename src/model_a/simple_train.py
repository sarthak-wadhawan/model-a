"""Simplified training script for MVP demonstration with synthetic data."""

import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
import numpy as np
import torch

from model_a.model import create_model
from model_a.preprocessing import Scaler, calculate_majority_baseline
from model_a.training import ModelTrainer
from model_a.sequences import SequenceExample


def create_synthetic_sequences(n_samples: int = 100) -> list:
    """Create synthetic sequences for demonstration."""
    sequences = []
    base_time = datetime(2026, 8, 18, 13, 55, tzinfo=timezone.utc)
    
    for i in range(n_samples):
        # Create 5 consecutive timestamps
        timestamps = [base_time - timedelta(minutes=4-j) for j in range(5)]
        
        # Create synthetic features (20 values)
        features = np.random.randn(20).tolist()
        
        # Create target (binary)
        target = i % 2  # Alternate between UP and DOWN
        
        # Create midpoints
        decision_midpoint = 100.0 + i * 0.1
        target_midpoint = decision_midpoint + (1 if target == 1 else -1) * 0.5
        
        seq = SequenceExample(
            input_timestamps=timestamps,
            features=features,
            target_timestamp=timestamps[-1] + timedelta(minutes=5),
            target=target,
            actual_direction=target,
            decision_midpoint=decision_midpoint,
            target_midpoint=target_midpoint
        )
        sequences.append(seq)
    
    return sequences


def main():
    print("Creating synthetic sequences for MVP demonstration...")
    
    # Create synthetic training data
    train_sequences = create_synthetic_sequences(80)
    val_sequences = create_synthetic_sequences(20)
    
    print(f"Training samples: {len(train_sequences)}")
    print(f"Validation samples: {len(val_sequences)}")
    
    # Calculate majority baseline
    train_targets = [seq.target for seq in train_sequences]
    baseline_accuracy = calculate_majority_baseline(train_targets)
    print(f"Majority baseline accuracy: {baseline_accuracy:.4f}")
    
    # Create model and trainer
    model = create_model(seed=42)
    scaler = Scaler()
    trainer = ModelTrainer(model, scaler, seed=42)
    
    # Train model
    print("Training model...")
    history = trainer.train(train_sequences, val_sequences, max_epochs=20)
    
    print(f"Training completed in {history['epochs_trained']} epochs")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
    
    # Create output directories
    model_dir = Path("artifacts/model_a/models")
    scaler_dir = Path("artifacts/model_a/scalers")
    model_dir.mkdir(parents=True, exist_ok=True)
    scaler_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model and scaler
    model_path = model_dir / "model.pt"
    scaler_path = scaler_dir / "scaler.json"
    
    trainer.save_model(model_path, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    
    # Compare with baseline
    val_accuracy = history['val_accuracy'][-1]
    if val_accuracy > baseline_accuracy:
        print(f"Model ({val_accuracy:.4f}) beats baseline ({baseline_accuracy:.4f})")
    else:
        print(f"Model ({val_accuracy:.4f}) does not beat baseline ({baseline_accuracy:.4f})")
    
    # Create demonstration prediction
    print("\nCreating demonstration prediction...")
    from model_a.prediction import Predictor
    
    predictor = Predictor.load(model_path, scaler_path, seed=42)
    
    # Create a demo sequence for Aug 25 09:55
    demo_time = datetime(2026, 8, 25, 13, 55, tzinfo=timezone.utc)  # 09:55 ET
    demo_timestamps = [demo_time - timedelta(minutes=4-j) for j in range(5)]
    demo_features = np.random.randn(20).tolist()
    
    demo_seq = SequenceExample(
        input_timestamps=demo_timestamps,
        features=demo_features,
        target_timestamp=demo_time + timedelta(minutes=5),
        target=1,
        actual_direction=1,
        decision_midpoint=100.0,
        target_midpoint=101.0
    )
    
    prediction = predictor.predict_single(demo_seq)
    
    print(f"Demonstration prediction for 2026-08-25 09:55 ET:")
    print(f"  Direction: {prediction['direction']}")
    print(f"  P(UP): {prediction['p_up']:.4f}")
    print(f"  P(DOWN): {prediction['p_down']:.4f}")
    print(f"  Actual outcome: {prediction['actual_direction']}")
    
    # Save prediction
    pred_dir = Path("artifacts/model_a/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / "demo_prediction.json"
    
    with open(pred_path, 'w') as f:
        json.dump(prediction, f, indent=2)
    
    print(f"Prediction saved to {pred_path}")
    
    return {
        'training_samples': len(train_sequences),
        'validation_samples': len(val_sequences),
        'epochs_trained': history['epochs_trained'],
        'final_val_accuracy': val_accuracy,
        'baseline_accuracy': baseline_accuracy,
        'demo_prediction': prediction
    }


if __name__ == '__main__':
    result = main()
    print("\n=== MVP Implementation Complete ===")
    print(f"Files created: model.pt, scaler.json, demo_prediction.json")
    print(f"Training samples: {result['training_samples']}")
    print(f"Validation samples: {result['validation_samples']}")
    print(f"Epochs trained: {result['epochs_trained']}")
    print(f"Validation accuracy: {result['final_val_accuracy']:.4f}")
    print(f"Baseline accuracy: {result['baseline_accuracy']:.4f}")
    print(f"Demonstration P(UP): {result['demo_prediction']['p_up']:.4f}")
    print(f"Demonstration direction: {result['demo_prediction']['direction']}")