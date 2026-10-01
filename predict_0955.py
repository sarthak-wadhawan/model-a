"""Prediction-only Aug 25 09:55 demonstration from saved model."""

import sys
sys.path.insert(0, 'src')

import json
from pathlib import Path
import time
from datetime import datetime
import pytz

from model_a.cache_loader import CacheLoader
from model_a.prediction import Predictor


def find_0955_sequence(demo_seqs):
    """Find the specific 09:55 sequence from demo sequences."""
    ny_tz = pytz.timezone("America/New_York")
    
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
    print("PREDICTION-ONLY AUG 25 09:55 DEMONSTRATION")
    print("=" * 60)
    
    # Paths
    cache_path = "artifacts/model_a/minute_features.csv"
    model_path = "artifacts/model_a/models/model.pt"
    scaler_path = "artifacts/model_a/scalers/scaler.json"
    
    # Load cache
    print("\n1. Loading minute summaries from cache...")
    loader = CacheLoader(cache_path)
    summaries_by_date = loader.load_summaries()
    
    # Build sequences
    print("\n2. Building five-point sequences...")
    all_sequences = loader.build_sequences(summaries_by_date)
    
    # Split by date
    print("\n3. Splitting sequences by date...")
    train_dates = ['2026-08-18', '2026-08-19', '2026-08-20', '2026-08-21']
    val_date = '2026-08-24'
    demo_date = '2026-08-25'
    
    train_seqs, val_seqs, demo_seqs = loader.split_by_date(
        all_sequences, train_dates, val_date, demo_date)
    
    print(f"   Demo sequences: {len(demo_seqs)}")
    
    # Load saved model (no retraining)
    print("\n4. Loading saved model and scaler (no retraining)...")
    predictor = Predictor.load(Path(model_path), Path(scaler_path), seed=42)
    print("   Model loaded successfully")
    
    # Run Aug 25 demonstration
    print("\n5. Running Aug 25 09:55 historical demonstration...")
    if demo_seqs:
        # Find the specific 09:55 sequence
        demo_seq = find_0955_sequence(demo_seqs)
        
        if demo_seq:
            ny_tz = pytz.timezone("America/New_York")
            
            print(f"   Found 09:55 sequence")
            print(f"   Input times: {[t.astimezone(ny_tz).strftime('%H:%M') for t in demo_seq.input_timestamps]}")
            print(f"   Target time: {demo_seq.target_timestamp.astimezone(ny_tz).strftime('%H:%M')}")
            
            predict_start = time.time()
            prediction = predictor.predict_single(demo_seq)
            predict_time = time.time() - predict_start
            
            print(f"\n   Prediction completed in {predict_time:.2f} seconds")
            
            # Extract detailed information
            decision_time = demo_seq.input_timestamps[-1].astimezone(ny_tz)
            target_time = demo_seq.target_timestamp.astimezone(ny_tz)
            
            print(f"\n" + "=" * 60)
            print("DETAILED PREDICTION RESULTS")
            print("=" * 60)
            print(f"\nDecision Information:")
            print(f"  Canonical boundary: {decision_time.strftime('%H:%M:%S')} ET")
            print(f"  Actual quote timestamp: {prediction['decision_quote_timestamp']}")
            print(f"  Bid: {prediction['decision_bid']:.2f}")
            print(f"  Ask: {prediction['decision_ask']:.2f}")
            print(f"  Calculated midpoint: {prediction['decision_midpoint']:.4f}")
            
            print(f"\nTarget Information:")
            print(f"  Canonical boundary: {target_time.strftime('%H:%M:%S')} ET")
            print(f"  Actual quote timestamp: {prediction['target_quote_timestamp']}")
            print(f"  Bid: {prediction['target_bid']:.2f}")
            print(f"  Ask: {prediction['target_ask']:.2f}")
            print(f"  Calculated midpoint: {prediction['target_midpoint']:.4f}")
            
            print(f"\nPrediction:")
            print(f"  P(UP): {prediction['p_up']:.4f}")
            print(f"  Predicted direction: {prediction['direction']}")
            print(f"  Actual direction: {prediction['actual_direction']}")
            
            print(f"\nMidpoint Change:")
            print(f"  Change: {prediction['target_midpoint'] - prediction['decision_midpoint']:.4f}")
            print(f"  Direction: {'UP' if prediction['target_midpoint'] > prediction['decision_midpoint'] else 'DOWN'}")
        else:
            print("   ERROR: Could not find 09:55 sequence")
    else:
        print("   No demo sequences available")
    
    print("\n" + "=" * 60)
    print("PREDICTION-ONLY DEMONSTRATION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()