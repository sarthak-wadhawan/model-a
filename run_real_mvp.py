"""Run corrected real MVP training and demonstration."""

import sys
sys.path.insert(0, 'src')

import json
from pathlib import Path
import time
from datetime import datetime
import pytz
import numpy as np

from model_a.cache_loader import CacheLoader
from model_a.model import create_model
from model_a.preprocessing import Scaler, calculate_majority_baseline
from model_a.training import ModelTrainer
from model_a.prediction import Predictor


def find_0955_sequence(demo_seqs):
    """Find the specific 09:55 sequence from demo sequences."""
    ny_tz = pytz.timezone("America/New_York")
    target_time = ny_tz.localize(datetime.strptime("2026-08-25 09:55", "%Y-%m-%d %H:%M"))
    
    for seq in demo_seqs:
        # Check if the last input timestamp is 09:55
        last_input_time = seq.input_timestamps[-1].astimezone(ny_tz)
        if last_input_time.hour == 9 and last_input_time.minute == 55:
            # Verify the five inputs are 09:51-09:55
            first_input_time = seq.input_timestamps[0].astimezone(ny_tz)
            if (first_input_time.hour == 9 and first_input_time.minute == 51):
                # Verify target is 10:00
                target_time_check = seq.target_timestamp.astimezone(ny_tz)
                if target_time_check.hour == 10 and target_time_check.minute == 0:
                    return seq
    
    return None


def main():
    print("=" * 60)
    print("CORRECTED REAL MODEL A MVP TRAINING AND DEMONSTRATION")
    print("=" * 60)
    
    # Paths
    cache_path = "artifacts/model_a/minute_features.csv"
    model_path = "artifacts/model_a/models/model.pt"
    scaler_path = "artifacts/model_a/scalers/scaler.json"
    prediction_path = "artifacts/model_a/predictions/demo_prediction.json"
    
    # Load cache
    print("\n1. Loading minute summaries from cache...")
    loader = CacheLoader(cache_path)
    summaries_by_date = loader.load_summaries()
    
    total_summaries = sum(len(s) for s in summaries_by_date.values())
    print(f"   Loaded {total_summaries} minute summaries")
    for date, summaries in summaries_by_date.items():
        print(f"   {date}: {len(summaries)} summaries")
    
    # Build sequences
    print("\n2. Building five-point sequences...")
    all_sequences = loader.build_sequences(summaries_by_date)
    print(f"   Built {len(all_sequences)} sequences")
    
    # Split by date
    print("\n3. Splitting sequences by date...")
    train_dates = ['2026-08-18', '2026-08-19', '2026-08-20', '2026-08-21']
    val_date = '2026-08-24'
    demo_date = '2026-08-25'
    
    train_seqs, val_seqs, demo_seqs = loader.split_by_date(
        all_sequences, train_dates, val_date, demo_date)
    
    print(f"   Training sequences: {len(train_seqs)}")
    print(f"   Validation sequences: {len(val_seqs)}")
    print(f"   Demo sequences: {len(demo_seqs)}")
    
    # Calculate majority baseline
    print("\n4. Calculating majority-class baseline...")
    train_targets = np.array([seq.target for seq in train_seqs])
    val_targets = np.array([seq.target for seq in val_seqs])
    baseline_accuracy = calculate_majority_baseline(
        train_targets, val_targets)
    majority_class = 1 if sum(train_targets) >= len(train_targets) / 2 else 0
    majority_class_name = "UP" if majority_class == 1 else "DOWN"
    print(f"   Training majority class: {majority_class_name}")
    print(f"   Validation baseline accuracy: {baseline_accuracy:.4f}")
    
    # Create model and trainer
    print("\n5. Creating model and trainer...")
    model = create_model(seed=42)
    scaler = Scaler()
    trainer = ModelTrainer(model, scaler, seed=42)
    
    # Train model
    print("\n6. Training model...")
    train_start = time.time()
    history = trainer.train(train_seqs, val_seqs, max_epochs=100)
    train_time = time.time() - train_start
    
    print(f"   Training completed in {train_time:.2f} seconds")
    print(f"   Epochs trained: {history['epochs_trained']}")
    print(f"   Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"   Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
    
    # Save model and scaler
    print("\n7. Saving model and scaler...")
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    
    trainer.save_model(Path(model_path), Path(scaler_path))
    print(f"   Model saved to {model_path}")
    print(f"   Scaler saved to {scaler_path}")
    
    # Load model for prediction (without retraining)
    print("\n8. Loading model for prediction (no retraining)...")
    predictor = Predictor.load(Path(model_path), Path(scaler_path), seed=42)
    
    # Run Aug 25 demonstration
    print("\n9. Running Aug 25 09:55 historical demonstration...")
    if demo_seqs:
        # Find the specific 09:55 sequence
        demo_seq = find_0955_sequence(demo_seqs)
        
        if demo_seq:
            print(f"   Found 09:55 sequence")
            print(f"   Input times: {[t.astimezone(pytz.timezone('America/New_York')).strftime('%H:%M') for t in demo_seq.input_timestamps]}")
            print(f"   Target time: {demo_seq.target_timestamp.astimezone(pytz.timezone('America/New_York')).strftime('%H:%M')}")
            
            predict_start = time.time()
            prediction = predictor.predict_single(demo_seq)
            predict_time = time.time() - predict_start
            
            print(f"   Prediction completed in {predict_time:.2f} seconds")
            print(f"   P(UP): {prediction['p_up']:.4f}")
            print(f"   Predicted direction: {prediction['direction']}")
            print(f"   Actual direction: {prediction['actual_direction']}")
            print(f"   Decision midpoint: {prediction['decision_midpoint']:.4f}")
            print(f"   Target midpoint: {prediction['target_midpoint']:.4f}")
            
            # Save prediction
            Path(prediction_path).parent.mkdir(parents=True, exist_ok=True)
            with open(prediction_path, 'w') as f:
                json.dump(prediction, f, indent=2)
            print(f"   Prediction saved to {prediction_path}")
        else:
            print("   ERROR: Could not find 09:55 sequence")
            prediction = None
    else:
        print("   No demo sequences available")
        prediction = None
    
    # Report results
    print("\n" + "=" * 60)
    print("CORRECTED EXPERIMENTAL RESULTS")
    print("=" * 60)
    
    print(f"\nSample Counts:")
    print(f"  Training samples: {len(train_seqs)}")
    print(f"  Validation samples: {len(val_seqs)}")
    print(f"  Demo samples: {len(demo_seqs)}")
    
    print(f"\nLabel Distribution:")
    print(f"  Training UP: {sum(train_targets)}")
    print(f"  Training DOWN: {len(train_targets) - sum(train_targets)}")
    print(f"  Validation UP: {sum(val_targets)}")
    print(f"  Validation DOWN: {len(val_targets) - sum(val_targets)}")
    
    print(f"\nTraining Results:")
    print(f"  Epochs trained: {history['epochs_trained']}")
    print(f"  Validation accuracy: {history['val_accuracy'][-1]:.4f}")
    print(f"  Validation loss: {history['val_loss'][-1]:.4f}")
    print(f"  Training time: {train_time:.2f} seconds")
    
    print(f"\nBaseline Comparison:")
    print(f"  Training majority class: {majority_class_name}")
    print(f"  Validation baseline accuracy: {baseline_accuracy:.4f}")
    print(f"  Model validation accuracy: {history['val_accuracy'][-1]:.4f}")
    if history['val_accuracy'][-1] > baseline_accuracy:
        print(f"  Model beats baseline: YES ({history['val_accuracy'][-1] - baseline_accuracy:.4f})")
    else:
        print(f"  Model beats baseline: NO ({baseline_accuracy - history['val_accuracy'][-1]:.4f})")
    
    if prediction:
        print(f"\nAug 25 09:55 Demonstration:")
        print(f"  P(UP): {prediction['p_up']:.4f}")
        print(f"  Predicted direction: {prediction['direction']}")
        print(f"  Actual direction: {prediction['actual_direction']}")
        print(f"  Decision midpoint: {prediction['decision_midpoint']:.4f}")
        print(f"  Target midpoint: {prediction['target_midpoint']:.4f}")
        print(f"  Prediction time: {predict_time:.2f} seconds")
    
    print("\n" + "=" * 60)
    print("CORRECTED MVP COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()