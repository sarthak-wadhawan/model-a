"""Historical prediction without retraining."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch

from .model import FixedMLP
from .preprocessing import Scaler
from .sequences import SequenceExample


class Predictor:
    """Load saved model and make predictions without retraining."""
    
    def __init__(self, model: FixedMLP, scaler: Scaler):
        self.model = model
        self.scaler = scaler
        self.model.eval()
    
    @classmethod
    def load(cls, model_path: Path, scaler_path: Path, seed: int = 42) -> 'Predictor':
        """Load predictor from saved artifacts."""
        from .training import ModelTrainer
        model, scaler = ModelTrainer.load_model(model_path, scaler_path, seed)
        return cls(model, scaler)
    
    def predict(self, sequences: List[SequenceExample]) -> List[Dict[str, Any]]:
        """Make predictions on sequences without retraining.
        
        Args:
            sequences: List of SequenceExample objects
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        for seq in sequences:
            # Extract features
            features = np.array([seq.features])
            
            # Scale using loaded scaler
            features_scaled = self.scaler.transform(features.tolist())
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features_scaled)
            
            # Get prediction
            with torch.no_grad():
                proba = self.model.predict_proba(features_tensor).item()
                prediction = 1 if proba >= 0.5 else 0
            
            # Build prediction result
            result = {
                'symbol': 'SPY',
                'prediction_timestamp': seq.input_timestamps[-1].isoformat(),
                'direction': 'UP' if prediction == 1 else 'DOWN',
                'p_up': proba,
                'p_down': 1.0 - proba,
                'input_timestamps': [ts.isoformat() for ts in seq.input_timestamps],
                'model_inputs': seq.features,
                'model_version': 'model-a-0.1.0',
                'is_historical': True,
                'target_timestamp': seq.target_timestamp.isoformat(),
                'actual_direction': 'UP' if seq.actual_direction == 1 else 'DOWN',
                'decision_midpoint': seq.decision_midpoint,
                'target_midpoint': seq.target_midpoint,
                'decision_quote_timestamp': seq.decision_quote_timestamp.isoformat() if seq.decision_quote_timestamp else None,
                'decision_bid': seq.decision_bid,
                'decision_ask': seq.decision_ask,
                'target_quote_timestamp': seq.target_quote_timestamp.isoformat() if seq.target_quote_timestamp else None,
                'target_bid': seq.target_bid,
                'target_ask': seq.target_ask
            }
            
            predictions.append(result)
        
        return predictions
    
    def predict_single(self, sequence: SequenceExample) -> Dict[str, Any]:
        """Make prediction on a single sequence."""
        return self.predict([sequence])[0]


def save_predictions(predictions: List[Dict[str, Any]], output_path: Path) -> None:
    """Save predictions to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)


def load_predictions(input_path: Path) -> List[Dict[str, Any]]:
    """Load predictions from JSON file."""
    with open(input_path, 'r') as f:
        return json.load(f)
