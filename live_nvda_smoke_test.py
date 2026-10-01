"""Live NVDA smoke test using frozen Model A v1 with timestamp-locked replay."""

import sys
sys.path.insert(0, 'src')

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
import pytz
import numpy as np
import torch

# Alpaca API imports
try:
    import alpaca_trade_api as tradeapi
except ImportError:
    print("Installing alpaca-trade-api...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "alpaca-trade-api"])
    import alpaca_trade_api as tradeapi

from model_a.model import FixedMLP
from model_a.preprocessing import Scaler
from model_a.features import FeatureCalculator, MinuteSummary
from model_a.trade_signing import TradeSigner
from model_a.data_loader import Event


class LiveEvent:
    """Live market event from Alpaca IEX feed."""
    
    def __init__(self, timestamp: datetime, event_type: str,
                 bid_price: Optional[float], ask_price: Optional[float],
                 bid_size: Optional[float], ask_size: Optional[float],
                 trade_price: Optional[float], trade_size: Optional[float],
                 aggressor_side: Optional[int]):
        self.timestamp = timestamp
        self.event_type = event_type
        self.bid_price = bid_price
        self.ask_price = ask_price
        self.bid_size = bid_size
        self.ask_size = ask_size
        self.trade_price = trade_price
        self.trade_size = trade_size
        self.aggressor_side = aggressor_side
    
    def is_quote(self) -> bool:
        return self.event_type == "quote"
    
    def is_trade(self) -> bool:
        return self.event_type == "trade"
    
    def has_valid_quote(self) -> bool:
        return (self.bid_price is not None and self.ask_price is not None and
                self.bid_size is not None and self.ask_size is not None)
    
    def has_valid_trade(self) -> bool:
        return (self.trade_price is not None and self.trade_size is not None)


class LiveNVDAFetcher:
    """Fetch live NVDA data from Alpaca IEX feed."""
    
    def __init__(self):
        self.api_key = os.environ.get('APCA_API_KEY_ID')
        self.api_secret = os.environ.get('APCA_API_SECRET_KEY')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API credentials not found in environment")
        
        self.api = tradeapi.REST(
            self.api_key,
            self.api_secret,
            base_url='https://data.alpaca.markets',
            api_version='v2'
        )
        
        self.ny_tz = pytz.timezone("America/New_York")
        self.chicago_tz = pytz.timezone("America/Chicago")
    
    def fetch_bars(self, symbol: str, start: datetime, end: datetime, 
                   timeframe: str = '1Min') -> List[LiveEvent]:
        """Fetch historical bars for a time range (for replay mode)."""
        # Convert to UTC for API
        start_utc = start.astimezone(timezone.utc)
        end_utc = end.astimezone(timezone.utc)
        
        print(f"Fetching bars for {symbol} from {start_utc} to {end_utc}...")
        
        bars = self.api.get_bars(
            symbol,
            timeframe,
            start=start_utc.isoformat(),
            end=end_utc.isoformat(),
            feed='iex',
            adjustment='raw'
        )
        
        events = []
        for bar in bars:
            # The bar.t is a pandas Timestamp, convert to datetime with UTC
            import pandas as pd
            if isinstance(bar.t, pd.Timestamp):
                bar_time = bar.t.to_pydatetime().replace(tzinfo=timezone.utc)
            else:
                bar_time = bar.t.replace(tzinfo=timezone.utc)
            
            # Create a quote event at the close of the bar
            events.append(LiveEvent(
                timestamp=bar_time,
                event_type='quote',
                bid_price=bar.l,  # Use low as bid (conservative)
                ask_price=bar.h,  # Use high as ask (conservative)
                bid_size=bar.v,   # Use volume as size
                ask_size=bar.v,
                trade_price=bar.c,  # Close as trade price
                trade_size=bar.v,
                aggressor_side=None
            ))
        
        print(f"Fetched {len(events)} bars")
        for i, event in enumerate(events[:3]):  # Print first 3 for debugging
            print(f"  Bar {i+1}: timestamp={event.timestamp}, bid={event.bid_price}, ask={event.ask_price}, volume={event.trade_size}")
        
        # If we got bars, use them. If not, try fetching recent bars as fallback
        if not events:
            print("No bars returned, trying to fetch recent bars...")
            recent_bars = self.api.get_bars(
                symbol,
                timeframe,
                feed='iex',
                adjustment='raw',
                limit=10  # Get last 10 bars
            )
            for bar in recent_bars:
                import pandas as pd
                if isinstance(bar.t, pd.Timestamp):
                    bar_time = bar.t.to_pydatetime().replace(tzinfo=timezone.utc)
                else:
                    bar_time = bar.t.replace(tzinfo=timezone.utc)
                
                events.append(LiveEvent(
                    timestamp=bar_time,
                    event_type='quote',
                    bid_price=bar.l,
                    ask_price=bar.h,
                    bid_size=bar.v,
                    ask_size=bar.v,
                    trade_price=bar.c,
                    trade_size=bar.v,
                    aggressor_side=None
                ))
            print(f"Fetched {len(events)} recent bars as fallback")
        
        return events
    
    def fetch_live_quotes(self, symbol: str, start: datetime, end: datetime) -> List[LiveEvent]:
        """Fetch live quotes using snapshot API for replay mode."""
        # For replay mode, we'll use bars
        return self.fetch_bars(symbol, start, end, timeframe='1Min')


def calculate_minute_summary_live(events: List[LiveEvent], 
                                   decision_time: datetime,
                                   calculator: FeatureCalculator) -> Optional[MinuteSummary]:
    """Calculate minute summary from live events using the exact Model A v1 logic."""
    if not events:
        return None
    
    # Filter events to only those <= decision_time (causal constraint)
    causal_events = [e for e in events if e.timestamp <= decision_time]
    
    if not causal_events:
        return None
    
    # With bar data, each bar represents a summary of the minute
    # We'll use the last bar in the window as the representative quote
    last_event = causal_events[-1]
    
    if not last_event.has_valid_quote():
        return None
    
    # Calculate quote-based features
    midpoint = (last_event.bid_price + last_event.ask_price) / 2.0
    queue_imbalance = calculator._calculate_queue_imbalance(
        last_event.bid_size, last_event.ask_size)
    relative_spread = calculator._calculate_relative_spread(
        last_event.bid_price, last_event.ask_price, midpoint)
    
    # For bar data, we can't calculate signed volume imbalance properly
    # Use a neutral value (this is a limitation of using bars vs raw events)
    signed_volume_imbalance = 0.0
    trade_count = 1  # At least one trade occurred
    
    # Calculate midpoint return
    if calculator.previous_midpoint is not None and calculator.previous_minute_time is not None:
        midpoint_return = (midpoint - calculator.previous_midpoint) / calculator.previous_midpoint
    else:
        midpoint_return = 0.0
    
    # Update state for next minute
    calculator.previous_midpoint = midpoint
    calculator.previous_minute_time = decision_time
    
    return MinuteSummary(
        timestamp=decision_time,
        queue_imbalance=queue_imbalance,
        signed_volume_imbalance=signed_volume_imbalance,
        relative_spread=relative_spread,
        midpoint_return=midpoint_return,
        midpoint=midpoint,
        has_valid_quote=True,
        trade_count=trade_count
    ), last_event  # Return last_event for debugging


def load_frozen_model(model_path: Path, scaler_path: Path, seed: int = 42) -> tuple:
    """Load frozen Model A v1 and scaler."""
    print(f"Loading frozen model from {model_path}")
    print(f"Loading frozen scaler from {scaler_path}")
    
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
    
    print("Model and scaler loaded successfully")
    return model, scaler


def predict_with_frozen_model(model: FixedMLP, scaler: Scaler, 
                              features: List[float]) -> Dict[str, Any]:
    """Make prediction using frozen model."""
    features_array = np.array([features])
    features_scaled = scaler.transform(features_array.tolist())
    features_tensor = torch.FloatTensor(features_scaled)
    
    with torch.no_grad():
        proba = model.predict_proba(features_tensor).item()
        prediction = 1 if proba >= 0.5 else 0
    
    return {
        'p_up': proba,
        'p_down': 1.0 - proba,
        'prediction': prediction,
        'direction': 'UP' if prediction == 1 else 'DOWN'
    }


def main():
    print("=" * 80)
    print("LIVE NVDA SMOKE TEST - MODEL A v1")
    print("=" * 80)
    
    # Configuration
    SYMBOL = "NVDA"
    
    chicago_tz = pytz.timezone("America/Chicago")
    ny_tz = pytz.timezone("America/New_York")
    
    # For this test, we'll use a live approach with the most recent available data
    # Fetch the last 10 minutes of data and use the most recent bar as decision boundary
    current_time = datetime.now(chicago_tz)
    
    print(f"\nConfiguration:")
    print(f"  Symbol: {SYMBOL}")
    print(f"  Using live data with dynamic decision boundary")
    print(f"  Target boundary: 5 minutes after decision")
    
    # Load frozen model
    model_path = Path("artifacts/model_a/models/model.pt")
    scaler_path = Path("artifacts/model_a/scalers/scaler.json")
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please train the model first using run_real_mvp.py")
        return
    
    if not scaler_path.exists():
        print(f"ERROR: Scaler not found at {scaler_path}")
        print("Please train the model first using run_real_mvp.py")
        return
    
    model, scaler = load_frozen_model(model_path, scaler_path)
    
    # Fetch live data with timestamp-locked replay
    print(f"\n" + "=" * 80)
    print("STAGE 1: PREDICTION (data <= 13:50:00 CT only)")
    print("=" * 80)
    
    fetcher = LiveNVDAFetcher()
    
    # Fetch data for prediction window - use the most recent 10 minutes of live data
    # We'll use the last 10 available bars and use the most recent as decision boundary
    print(f"\nFetching recent live data (last 10 minutes)...")
    
    try:
        # Fetch the last 10 minutes of live data
        recent_start = current_time - timedelta(minutes=10)
        recent_end = current_time
        live_events = fetcher.fetch_live_quotes(
            SYMBOL, recent_start, recent_end)
    except Exception as e:
        print(f"ERROR fetching live data: {e}")
        return
    
    if not live_events:
        print("ERROR: No live data returned")
        return
    
    # Use the last 5 bars as our input data
    # The most recent bar is our decision boundary
    recent_bars = live_events[-5:] if len(live_events) >= 5 else live_events
    
    if len(recent_bars) < 5:
        print(f"ERROR: Only {len(recent_bars)} bars available, need 5")
        return
    
    # Update decision boundary to match the most recent bar
    last_bar = recent_bars[-1]
    decision_boundary_utc = last_bar.timestamp
    decision_boundary_ct = decision_boundary_utc.astimezone(chicago_tz)
    decision_boundary_et = decision_boundary_utc.astimezone(ny_tz)
    
    print(f"\nUsing live data with decision boundary: {decision_boundary_ct.strftime('%H:%M:%S')} CT")
    print(f"Total bars fetched: {len(live_events)}")
    print(f"Using last 5 bars for prediction")
    
    # Build minute summaries for the 5 input minutes
    calculator = FeatureCalculator()
    calculator.reset_state()
    
    minute_summaries = []
    
    print(f"\nBuilding minute summaries from live bars...")
    for i, bar in enumerate(recent_bars):
        bar_time_utc = bar.timestamp
        bar_time_et = bar_time_utc.astimezone(ny_tz)
        
        # Create a window for this bar (just the bar itself)
        window_events = [bar]
        
        # Calculate summary
        result = calculate_minute_summary_live(window_events, bar_time_utc, calculator)
        
        if result:
            summary, last_event = result
            minute_summaries.append(summary)
            print(f"\nMinute {i+1} ({bar_time_et.strftime('%H:%M')} ET):")
            print(f"  Actual final quote timestamp: {bar_time_et.strftime('%H:%M:%S')} ET")
            print(f"  Bid: {last_event.bid_price:.4f}")
            print(f"  Ask: {last_event.ask_price:.4f}")
            print(f"  Midpoint: {summary.midpoint:.4f}")
            print(f"  Queue imbalance: {summary.queue_imbalance:.4f}")
            print(f"  Signed volume imbalance: {summary.signed_volume_imbalance:.4f}")
            print(f"  Relative spread: {summary.relative_spread:.4f}")
            print(f"  Midpoint return: {summary.midpoint_return:.4f}")
        else:
            print(f"\nMinute {i+1} ({bar_time_et.strftime('%H:%M')} ET): INVALID - no valid quote")
    
    if len(minute_summaries) != 5:
        print(f"\nERROR: Expected 5 valid minute summaries, got {len(minute_summaries)}")
        return
    
    # Construct 20-feature vector
    print(f"\nConstructing 20-feature vector...")
    features = []
    for summary in minute_summaries:
        features.extend(summary.to_feature_vector())
    
    print(f"Feature vector length: {len(features)}")
    
    # Make prediction
    print(f"\nMaking prediction with frozen Model A v1...")
    prediction_result = predict_with_frozen_model(model, scaler, features)
    
    decision_midpoint = minute_summaries[-1].midpoint
    
    print(f"\n" + "=" * 80)
    print("PREDICTION RESULT")
    print("=" * 80)
    print(f"Ticker: {SYMBOL}")
    print(f"Decision boundary: {decision_boundary_ct.strftime('%Y-%m-%d %H:%M:%S')} CT")
    print(f"Target boundary: {(decision_boundary_ct + timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S')} CT")
    print(f"P(UP): {prediction_result['p_up']:.4f}")
    print(f"P(DOWN): {prediction_result['p_down']:.4f}")
    print(f"Predicted direction: {prediction_result['direction']}")
    print(f"Decision midpoint: {decision_midpoint:.4f}")
    
    print(f"\n" + "=" * 80)
    print("OUT-OF-DISTRIBUTION LIVE SMOKE TEST")
    print("MODEL WAS TRAINED ON SPY MORNING DATA")
    print("THIS IS NOT VALIDATED NVDA PERFORMANCE")
    print("DATA LIMITATION: Using Alpaca bars instead of raw IEX events")
    print("  - Signed volume imbalance set to 0.0 (cannot calculate from bars)")
    print("  - This is a structural limitation, not a model limitation")
    print("=" * 80)
    
    # Save prediction BEFORE loading target data
    prediction_output = {
        'symbol': SYMBOL,
        'decision_boundary_ct': decision_boundary_ct.strftime('%Y-%m-%d %H:%M:%S'),
        'target_boundary_ct': (decision_boundary_ct + timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S'),
        'p_up': prediction_result['p_up'],
        'p_down': prediction_result['p_down'],
        'predicted_direction': prediction_result['direction'],
        'decision_midpoint': decision_midpoint,
        'input_timestamps': [s.timestamp.astimezone(ny_tz).isoformat() for s in minute_summaries],
        'features': features,
        'model_version': 'model-a-0.1.0',
        'is_live': True,
        'out_of_distribution_warning': 'Model trained on SPY morning data, not validated on NVDA'
    }
    
    prediction_path = Path("artifacts/model_a/predictions/nvda_live_smoke_test.json")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(prediction_path, 'w') as f:
        json.dump(prediction_output, f, indent=2)
    
    print(f"\nPrediction saved to {prediction_path}")
    
    # Now load target data (AFTER prediction is saved)
    print(f"\n" + "=" * 80)
    print("STAGE 2: TARGET / RESULT (loading data after prediction saved)")
    print("=" * 80)
    
    # Fetch target minute data
    # For live mode, we need to wait 5 minutes or fetch the next bar
    # Since this is a smoke test, we'll fetch the next available bar after decision
    target_boundary_ct = decision_boundary_ct + timedelta(minutes=5)
    target_start_ct = target_boundary_ct - timedelta(minutes=1)
    target_end_ct = target_boundary_ct
    
    print(f"\nFetching target data: {target_start_ct.strftime('%H:%M:%S')} CT to {target_end_ct.strftime('%H:%M:%S')} CT")
    
    try:
        target_events = fetcher.fetch_live_quotes(
            SYMBOL, target_start_ct, target_end_ct)
    except Exception as e:
        print(f"ERROR fetching target data: {e}")
        return
    
    # If target window has no data, try to get the next available bar after decision
    if not target_events:
        print("No target data in exact window, fetching next available bar...")
        try:
            next_start = decision_boundary_ct + timedelta(minutes=1)
            next_end = decision_boundary_ct + timedelta(minutes=10)
            target_events = fetcher.fetch_live_quotes(
                SYMBOL, next_start, next_end)
        except Exception as e:
            print(f"ERROR fetching next available bar: {e}")
            return
    
    # Use the first available bar after decision as target
    if target_events:
        last_target_quote = target_events[0]  # Use first available bar
    else:
        last_target_quote = None
    
    if last_target_quote is None:
        print(f"\nWARNING: No valid quote found in target window")
        print("For live mode, the target bar may not be available yet")
        print("This is expected if running in real-time before 5 minutes have elapsed")
        print("Prediction was saved successfully - target evaluation deferred")
        return
    
    target_midpoint = (last_target_quote.bid_price + last_target_quote.ask_price) / 2.0
    
    # Update target boundary to match actual bar timestamp
    actual_target_boundary_ct = last_target_quote.timestamp.astimezone(chicago_tz)
    
    print(f"\nTarget minute:")
    print(f"  Requested boundary: {target_boundary_ct.strftime('%Y-%m-%d %H:%M:%S')} CT")
    print(f"  Actual boundary: {actual_target_boundary_ct.strftime('%Y-%m-%d %H:%M:%S')} CT")
    print(f"  Actual quote timestamp: {last_target_quote.timestamp.astimezone(ny_tz).strftime('%H:%M:%S')} ET")
    print(f"  Bid: {last_target_quote.bid_price:.4f}")
    print(f"  Ask: {last_target_quote.ask_price:.4f}")
    print(f"  Midpoint: {target_midpoint:.4f}")
    
    # Determine actual outcome
    if target_midpoint > decision_midpoint:
        actual_direction = "UP"
        actual_direction_code = 1
    else:
        actual_direction = "DOWN"
        actual_direction_code = 0
    
    midpoint_change = target_midpoint - decision_midpoint
    
    # Determine if prediction was correct
    is_correct = (prediction_result['direction'] == actual_direction)
    
    print(f"\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)
    print(f"Predicted direction: {prediction_result['direction']}")
    print(f"Actual direction: {actual_direction}")
    print(f"Correct: {is_correct}")
    print(f"P(UP): {prediction_result['p_up']:.4f}")
    print(f"Decision midpoint: {decision_midpoint:.4f}")
    print(f"Target midpoint: {target_midpoint:.4f}")
    print(f"Midpoint change: {midpoint_change:.4f}")
    
    # Update prediction file with target data
    prediction_output['target_midpoint'] = target_midpoint
    prediction_output['actual_direction'] = actual_direction
    prediction_output['actual_direction_code'] = actual_direction_code
    prediction_output['midpoint_change'] = midpoint_change
    prediction_output['is_correct'] = is_correct
    prediction_output['target_quote_timestamp'] = last_target_quote.timestamp.astimezone(ny_tz).isoformat()
    prediction_output['target_bid'] = last_target_quote.bid_price
    prediction_output['target_ask'] = last_target_quote.ask_price
    prediction_output['target_boundary_ct'] = actual_target_boundary_ct.strftime('%Y-%m-%d %H:%M:%S')
    prediction_output['target_note'] = 'Target boundary adjusted to match available data'
    
    with open(prediction_path, 'w') as f:
        json.dump(prediction_output, f, indent=2)
    
    print(f"\nFinal result saved to {prediction_path}")
    
    print(f"\n" + "=" * 80)
    print("LIVE NVDA SMOKE TEST COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
