"""Command-line interface for Model A MVP."""

import argparse
from pathlib import Path
import sys

import torch

from .sequences import SequenceBuilder, split_by_date
from .training import ModelTrainer, calculate_metrics
from .prediction import Predictor, save_predictions
from .preprocessing import calculate_majority_baseline


def main():
    parser = argparse.ArgumentParser(description='Model A - Five-point SPY/IEX microstructure MVP')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # micro-cache command
    cache_parser = subparsers.add_parser('micro-cache', help='Build one-minute feature cache')
    cache_parser.add_argument('--events', required=True, help='Path to events.csv')
    cache_parser.add_argument('--output', required=True, help='Output path for cached features')
    cache_parser.add_argument('--start-date', default='2026-08-18', help='Start date (YYYY-MM-DD)')
    cache_parser.add_argument('--end-date', default='2026-08-25', help='End date (YYYY-MM-DD)')
    
    # micro-train command
    train_parser = subparsers.add_parser('micro-train', help='Train the fixed MLP model')
    train_parser.add_argument('--events', required=True, help='Path to events.csv')
    train_parser.add_argument('--model-output', required=True, help='Output path for model')
    train_parser.add_argument('--scaler-output', required=True, help='Output path for scaler')
    train_parser.add_argument('--train-dates', nargs='+', default=['2026-08-18', '2026-08-19', '2026-08-20', '2026-08-21'], 
                            help='Training dates')
    train_parser.add_argument('--val-date', default='2026-08-24', help='Validation date')
    train_parser.add_argument('--demo-date', default='2026-08-25', help='Historical demonstration date')
    train_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # micro-predict command
    predict_parser = subparsers.add_parser('micro-predict', help='Make historical prediction')
    predict_parser.add_argument('--events', required=True, help='Path to events.csv')
    predict_parser.add_argument('--model', required=True, help='Path to saved model')
    predict_parser.add_argument('--scaler', required=True, help='Path to saved scaler')
    predict_parser.add_argument('--date', required=True, help='Date to predict (YYYY-MM-DD)')
    predict_parser.add_argument('--time', required=True, help='Prediction time (HH:MM)')
    predict_parser.add_argument('--output', required=True, help='Output path for predictions')
    predict_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    if args.command == 'micro-cache':
        cmd_cache(args)
    elif args.command == 'micro-train':
        cmd_train(args)
    elif args.command == 'micro-predict':
        cmd_predict(args)
    else:
        parser.print_help()
        sys.exit(1)


def cmd_cache(args):
    """Build one-minute feature cache from real events.csv."""
    print(f"Building feature cache from {args.events}")
    print(f"Date range: {args.start_date} to {args.end_date}")
    
    from .cache_builder import IncrementalCacheBuilder
    
    cache_builder = IncrementalCacheBuilder(args.events)
    output_path = Path(args.output)
    
    stats = cache_builder.build_cache(output_path)
    
    print(f"\nCache Statistics:")
    print(f"Total events processed: {stats['total_events_processed']}")
    print(f"Events by date:")
    for date, count in stats['events_by_date'].items():
        print(f"  {date}: {count}")
    print(f"Valid minute summaries by date:")
    for date, count in stats['valid_summaries_by_date'].items():
        print(f"  {date}: {count}")
    print(f"Total valid minute summaries: {stats['total_valid_summaries']}")


def cmd_train(args):
    """Train the fixed MLP model."""
    print(f"Training model from {args.events}")
    print(f"Training dates: {args.train_dates}")
    print(f"Validation date: {args.val_date}")
    print(f"Demo date: {args.demo_date}")
    
    # Build sequences for all dates
    builder = SequenceBuilder(args.events)
    
    # Get date range (need to include all dates)
    all_dates = args.train_dates + [args.val_date, args.demo_date]
    start_date = min(all_dates)
    end_date = max(all_dates)
    
    print(f"Building sequences from {start_date} to {end_date}...")
    all_sequences = builder.build_sequences(start_date, end_date)
    
    # Split chronologically
    train_seqs, val_seqs, demo_seqs = split_by_date(
        all_sequences, args.train_dates, args.val_date, args.demo_date)
    
    print(f"Training samples: {len(train_seqs)}")
    print(f"Validation samples: {len(val_seqs)}")
    print(f"Demonstration samples: {len(demo_seqs)}")
    
    # Calculate majority baseline
    train_targets = [seq.target for seq in train_seqs]
    baseline_accuracy = calculate_majority_baseline(train_targets)
    print(f"Majority baseline accuracy (training): {baseline_accuracy:.4f}")
    
    # Create model and trainer
    from .model import create_model
    from .preprocessing import Scaler
    
    model = create_model(seed=args.seed)
    scaler = Scaler()
    trainer = ModelTrainer(model, scaler, seed=args.seed)
    
    # Train model
    print("Training model...")
    history = trainer.train(train_seqs, val_seqs, max_epochs=100)
    
    print(f"Training completed in {history['epochs_trained']} epochs")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
    
    # Evaluate on validation set
    if len(val_seqs) > 0:
        val_features, val_targets = trainer._prepare_data(val_seqs)
        val_X = torch.FloatTensor(val_features)
        val_y = torch.FloatTensor(val_targets)
        
        _, val_accuracy = trainer._validate(val_X, val_y)
        print(f"Validation accuracy: {val_accuracy:.4f}")
    else:
        val_accuracy = 0.0
        print("No validation samples available")
    
    # Save model and scaler
    model_path = Path(args.model_output)
    scaler_path = Path(args.scaler_output)
    
    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    
    trainer.save_model(model_path, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    
    # Compare with baseline
    if val_accuracy > baseline_accuracy:
        print(f"Model ({val_accuracy:.4f}) beats baseline ({baseline_accuracy:.4f})")
    else:
        print(f"Model ({val_accuracy:.4f}) does not beat baseline ({baseline_accuracy:.4f})")


def cmd_predict(args):
    """Make historical prediction without retraining."""
    print(f"Loading model from {args.model}")
    print(f"Loading scaler from {args.scaler}")
    
    # Load predictor
    model_path = Path(args.model)
    scaler_path = Path(args.scaler)
    predictor = Predictor.load(model_path, scaler_path, seed=args.seed)
    
    print(f"Making prediction for {args.date} at {args.time}")
    
    # Build sequences for the specific date
    builder = SequenceBuilder(args.events)
    sequences = builder.build_sequences(args.date, args.date)
    
    # Find sequence matching the specific time
    target_time_str = f"{args.date} {args.time}"
    target_sequence = None
    
    for seq in sequences:
        if seq.input_timestamps[-1].strftime("%Y-%m-%d %H:%M") == target_time_str:
            target_sequence = seq
            break
    
    if target_sequence is None:
        print(f"No sequence found for {target_time_str}")
        sys.exit(1)
    
    # Make prediction
    prediction = predictor.predict_single(target_sequence)
    
    print(f"Prediction: {prediction['direction']}")
    print(f"P(UP): {prediction['p_up']:.4f}")
    print(f"P(DOWN): {prediction['p_down']:.4f}")
    print(f"Actual outcome: {prediction['actual_direction']}")
    print(f"Decision midpoint: {prediction['decision_midpoint']:.4f}")
    print(f"Target midpoint: {prediction['target_midpoint']:.4f}")
    
    # Save prediction
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_predictions([prediction], output_path)
    print(f"Prediction saved to {output_path}")


if __name__ == '__main__':
    main()
