"""Replay Model A v1 on historical Alpaca IEX quote/trade events."""

import sys
sys.path.insert(0, 'src')

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pytz
import numpy as np
import torch
import argparse

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


def parse_timestamp(ts) -> datetime:
    """Parse various timestamp formats to timezone-aware UTC datetime."""
    import pandas as pd
    
    if isinstance(ts, pd.Timestamp):
        result = ts.to_pydatetime()
        if result.tzinfo is None:
            result = result.replace(tzinfo=timezone.utc)
        return result
    elif isinstance(ts, str):
        # Simple approach: strip everything after microseconds, use fromisoformat
        ts_str = ts.replace('Z', '+00:00')
        
        # Remove nanoseconds and beyond (keep only 6 digits after decimal)
        if '.' in ts_str:
            # Replace everything after the decimal with first 6 digits
            match = re.match(r'(.*\.\d{6}).*', ts_str)
            if match:
                ts_str = match.group(1)
        
        try:
            result = datetime.fromisoformat(ts_str)
        except Exception as e:
            # Final fallback: use pandas
            try:
                result = pd.to_datetime(ts_str).to_pydatetime()
                if result.tzinfo is None:
                    result = result.replace(tzinfo=timezone.utc)
            except Exception as e2:
                print(f"Failed to parse timestamp: {ts_str}, error: {e}, {e2}")
                raise
        
        if result.tzinfo is None:
            result = result.replace(tzinfo=timezone.utc)
        return result
    else:
        result = ts
        if hasattr(result, 'tzinfo') and result.tzinfo is None:
            result = result.replace(tzinfo=timezone.utc)
        return result


class HistoricalEventFetcher:
    """Fetch historical Alpaca IEX quote/trade events."""
    
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
    
    def fetch_quotes(self, symbol: str, start: datetime, end: datetime) -> List[Event]:
        """Fetch historical IEX quotes for a time range."""
        print(f"\nFetching quotes for {symbol} from {start} to {end}...")
        
        # Ensure inputs are timezone-aware
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        
        start_utc = start.astimezone(timezone.utc)
        end_utc = end.astimezone(timezone.utc)
        
        quotes = self.api.get_crypto_quotes(
            symbol,
            start=start_utc.isoformat(),
            end=end_utc.isoformat(),
            feed='iex'
        )
        
        events = []
        for quote in quotes:
            # Convert to timezone-aware UTC using the helper function
            timestamp = parse_timestamp(quote.t)
            
            events.append(Event(
                timestamp=timestamp,
                event_type='quote',
                bid_price=quote.bp,
                ask_price=quote.ap,
                bid_size=quote.bs,
                ask_size=quote.as_,
                trade_price=None,
                trade_size=None,
                aggressor_side=None
            ))
        
        print(f"Fetched {len(events)} quote events")
        return events
    
    def fetch_trades(self, symbol: str, start: datetime, end: datetime) -> List[Event]:
        """Fetch historical IEX trades for a time range."""
        print(f"\nFetching trades for {symbol} from {start} to {end}...")
        
        # Ensure inputs are timezone-aware
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        
        start_utc = start.astimezone(timezone.utc)
        end_utc = end.astimezone(timezone.utc)
        
        trades = self.api.get_crypto_trades(
            symbol,
            start=start_utc.isoformat(),
            end=end_utc.isoformat(),
            feed='iex'
        )
        
        events = []
        for trade in trades:
            # Convert to timezone-aware UTC using the helper function
            timestamp = parse_timestamp(trade.t)
            
            events.append(Event(
                timestamp=timestamp,
                event_type='trade',
                bid_price=None,
                ask_price=None,
                bid_size=None,
                ask_size=None,
                trade_price=trade.p,
                trade_size=trade.s,
                aggressor_side=None
            ))
        
        print(f"Fetched {len(events)} trade events")
        return events
    
    def fetch_stock_quotes(self, symbol: str, start: datetime, end: datetime) -> List[Event]:
        """Fetch historical IEX stock quotes for a time range."""
        print(f"\nFetching stock quotes for {symbol} from {start} to {end}...")
        
        # Ensure inputs are timezone-aware
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        
        start_utc = start.astimezone(timezone.utc)
        end_utc = end.astimezone(timezone.utc)
        
        try:
            quotes = self.api.get_quotes(
                symbol,
                start=start_utc.isoformat(),
                end=end_utc.isoformat(),
                feed='iex'
            )
        except Exception as e:
            print(f"Error fetching quotes: {e}")
            # Try with sip feed as fallback
            print("Trying with sip feed...")
            quotes = self.api.get_quotes(
                symbol,
                start=start_utc.isoformat(),
                end=end_utc.isoformat(),
                feed='sip'
            )
        
        events = []
        for quote in quotes:
            # Convert to timezone-aware UTC using the helper function
            timestamp = parse_timestamp(quote.t)
            
            # Handle different field names depending on API version
            bid_price = getattr(quote, 'bp', None)
            ask_price = getattr(quote, 'ap', None)
            bid_size = getattr(quote, 'bs', None)
            ask_size = getattr(quote, 'as_', None) if hasattr(quote, 'as_') else getattr(quote, 'as', None)
            
            events.append(Event(
                timestamp=timestamp,
                event_type='quote',
                bid_price=bid_price,
                ask_price=ask_price,
                bid_size=bid_size,
                ask_size=ask_size,
                trade_price=None,
                trade_size=None,
                aggressor_side=None
            ))
        
        print(f"Fetched {len(events)} stock quote events")
        return events
    
    def fetch_stock_trades(self, symbol: str, start: datetime, end: datetime) -> List[Event]:
        """Fetch historical IEX stock trades for a time range."""
        print(f"\nFetching stock trades for {symbol} from {start} to {end}...")
        
        # Ensure inputs are timezone-aware
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        
        start_utc = start.astimezone(timezone.utc)
        end_utc = end.astimezone(timezone.utc)
        
        try:
            trades = self.api.get_trades(
                symbol,
                start=start_utc.isoformat(),
                end=end_utc.isoformat(),
                feed='iex'
            )
        except Exception as e:
            print(f"Error fetching trades: {e}")
            # Try with sip feed as fallback
            print("Trying with sip feed...")
            trades = self.api.get_trades(
                symbol,
                start=start_utc.isoformat(),
                end=end_utc.isoformat(),
                feed='sip'
            )
        
        events = []
        for trade in trades:
            # Convert to timezone-aware UTC using the helper function
            timestamp = parse_timestamp(trade.t)
            
            # Handle different field names depending on API version
            trade_price = getattr(trade, 'p', None)
            trade_size = getattr(trade, 's', None)
            
            events.append(Event(
                timestamp=timestamp,
                event_type='trade',
                bid_price=None,
                ask_price=None,
                bid_size=None,
                ask_size=None,
                trade_price=trade_price,
                trade_size=trade_size,
                aggressor_side=None  # Alpaca trades don't have aggressor side in IEX feed
            ))
        
        print(f"Fetched {len(events)} stock trade events")
        return events


def calculate_minute_summary_from_events(events: List[Event], 
                                           decision_time: datetime,
                                           calculator: FeatureCalculator) -> Optional[MinuteSummary]:
    """Calculate minute summary using exact Model A v1 logic with quote/trade events."""
    if not events:
        return None
    
    # Filter events to only those <= decision_time (causal constraint)
    causal_events = [e for e in events if e.timestamp <= decision_time]
    
    if not causal_events:
        return None
    
    # Find the last valid quote at or before decision_time
    last_quote = None
    for event in reversed(causal_events):
        if event.is_quote() and event.has_valid_quote():
            last_quote = event
            break
    
    if last_quote is None:
        return None
    
    # Calculate quote-based features
    midpoint = (last_quote.bid_price + last_quote.ask_price) / 2.0
    queue_imbalance = calculator._calculate_queue_imbalance(
        last_quote.bid_size, last_quote.ask_size)
    relative_spread = calculator._calculate_relative_spread(
        last_quote.bid_price, last_quote.ask_price, midpoint)
    
    # Calculate trade-based features with causal signing
    signed_volume, trade_count = calculator._calculate_signed_volume(
        causal_events, last_quote.timestamp)
    
    # Normalize signed volume by total volume
    total_volume = sum(e.trade_size for e in causal_events 
                      if e.is_trade() and e.has_valid_trade())
    if total_volume > 0:
        signed_volume_imbalance = signed_volume / total_volume
    else:
        signed_volume_imbalance = 0.0
    
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
    )


def load_frozen_model(model_path: Path, scaler_path: Path, seed: int = 42) -> tuple:
    """Load frozen Model A v1 and scaler."""
    print(f"\nLoading frozen model from {model_path}")
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


def run_replay(symbol: str, date: str, decision_time: str, timezone_str: str):
    """Run Model A v1 replay on historical data."""
    print("=" * 80)
    print("MODEL A v1 REPLAY RUNNER")
    print("=" * 80)
    
    # Parse timezone
    user_tz = pytz.timezone(timezone_str)
    
    # Parse decision time
    decision_dt = user_tz.localize(
        datetime.strptime(f"{date} {decision_time}", "%Y-%m-%d %H:%M"))
    
    # Calculate target time
    target_dt = decision_dt + timedelta(minutes=5)
    
    # Print timezone sanity check
    print(f"\nTIMEZONE SANITY CHECK:")
    print(f"Decision {timezone_str}: {decision_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Decision ET: {decision_dt.astimezone(pytz.timezone('America/New_York')).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Decision UTC: {decision_dt.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Target {timezone_str}: {target_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Target ET: {target_dt.astimezone(pytz.timezone('America/New_York')).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Target UTC: {target_dt.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    # Load frozen model
    model_path = Path("artifacts/model_a/models/model.pt")
    scaler_path = Path("artifacts/model_a/scalers/scaler.json")
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return
    
    if not scaler_path.exists():
        print(f"ERROR: Scaler not found at {scaler_path}")
        return
    
    model, scaler = load_frozen_model(model_path, scaler_path)
    
    # Fetch historical events
    fetcher = HistoricalEventFetcher()
    
    # Calculate data window: need events from (T-5min - 1min) to T+5min
    # Input summaries: (T-4min, T-3min, T-2min, T-1min, T]
    # Each summary needs data from (boundary-1min, boundary]
    # For first input (T-4min), we need prior midpoint from (T-5min, T-4min]
    # So we need data from T-6min to T+5min for full causal support
    data_start = decision_dt - timedelta(minutes=6)  # Extra minute for pre-roll
    data_end = target_dt + timedelta(minutes=1)  # Buffer for target
    
    print(f"\nFetching historical events: {data_start} to {data_end}")
    
    try:
        quotes = fetcher.fetch_stock_quotes(symbol, data_start, data_end)
        trades = fetcher.fetch_stock_trades(symbol, data_start, data_end)
    except Exception as e:
        print(f"ERROR fetching historical data: {e}")
        return
    
    print(f"\nTotal events retrieved: {len(quotes)} quotes, {len(trades)} trades")
    
    # Combine and sort events
    all_events = quotes + trades
    all_events.sort(key=lambda e: e.timestamp)
    
    # Build five minute summaries
    calculator = FeatureCalculator()
    calculator.reset_state()
    
    # First, calculate pre-roll midpoint for the first input (T-4min)
    # This is the midpoint from (T-5min, T-4min] CT
    pre_roll_boundary_dt = decision_dt - timedelta(minutes=5)
    pre_roll_boundary_utc = pre_roll_boundary_dt.astimezone(timezone.utc)
    pre_roll_window_start = pre_roll_boundary_utc - timedelta(minutes=1)
    
    pre_roll_events = [
        e for e in all_events 
        if pre_roll_window_start < e.timestamp <= pre_roll_boundary_utc
    ]
    
    pre_roll_summary = calculate_minute_summary_from_events(pre_roll_events, pre_roll_boundary_utc, calculator)
    
    if pre_roll_summary:
        print(f"\nPre-roll midpoint calculation (for first input's return):")
        print(f"  Pre-roll boundary: {pre_roll_boundary_dt.strftime('%H:%M')} {timezone_str}")
        print(f"  Pre-roll midpoint: {pre_roll_summary.midpoint:.4f}")
        # Reset calculator state so it only has the pre-roll midpoint
        calculator.previous_midpoint = pre_roll_summary.midpoint
        calculator.previous_minute_time = pre_roll_boundary_utc
    else:
        print(f"\nWARNING: No pre-roll midpoint available from {pre_roll_boundary_dt.strftime('%H:%M')} {timezone_str}")
        print(f"  First input's midpoint_return will be defaulted to 0.0")
    
    minute_summaries = []
    
    print(f"\nBuilding five minute summaries...")
    for i in range(5):
        offset = 4 - i  # 4, 3, 2, 1, 0 minutes before decision
        boundary_dt = decision_dt - timedelta(minutes=offset)
        boundary_utc = boundary_dt.astimezone(timezone.utc)
        
        # Get events in the trailing window (boundary - 1min, boundary]
        window_start = boundary_utc - timedelta(minutes=1)
        window_events = [
            e for e in all_events 
            if window_start < e.timestamp <= boundary_utc
        ]
        
        # Calculate summary
        summary = calculate_minute_summary_from_events(window_events, boundary_utc, calculator)
        
        if summary:
            minute_summaries.append(summary)
            # Get the last quote for debugging
            last_quote = None
            for event in reversed(window_events):
                if event.is_quote() and event.has_valid_quote():
                    last_quote = event
                    break
            
            # Check if this is the first input (12:56 CT = 13:56 ET) and show return calculation
            first_input_boundary_dt = decision_dt - timedelta(minutes=4)
            if boundary_dt == first_input_boundary_dt:
                if pre_roll_summary:
                    calculated_return = (summary.midpoint - pre_roll_summary.midpoint) / pre_roll_summary.midpoint
                    print(f"\nMinute {i+1} ({boundary_dt.strftime('%H:%M')} {timezone_str}) - FIRST INPUT:")
                    print(f"  Pre-roll midpoint (12:55 CT): {pre_roll_summary.midpoint:.4f}")
                    print(f"  Current midpoint (12:56 CT): {summary.midpoint:.4f}")
                    print(f"  Calculated return: {calculated_return:.6f}")
                    print(f"  Final actual quote timestamp: {boundary_utc.astimezone(pytz.timezone('America/New_York')).strftime('%H:%M:%S')} ET")
                else:
                    print(f"\nMinute {i+1} ({boundary_dt.strftime('%H:%M')} {timezone_str}) - FIRST INPUT:")
                    print(f"  WARNING: No pre-roll midpoint, return defaulted to: {summary.midpoint_return:.6f}")
                    print(f"  Final actual quote timestamp: {boundary_utc.astimezone(pytz.timezone('America/New_York')).strftime('%H:%M:%S')} ET")
            else:
                print(f"\nMinute {i+1} ({boundary_dt.strftime('%H:%M')} {timezone_str}):")
                print(f"  Final actual quote timestamp: {boundary_utc.astimezone(pytz.timezone('America/New_York')).strftime('%H:%M:%S')} ET")
            
            # Find the last quote for debugging
            last_quote = None
            for event in reversed(window_events):
                if event.is_quote() and event.has_valid_quote():
                    last_quote = event
                    break
            
            if last_quote:
                print(f"  Bid: {last_quote.bid_price:.4f}")
                print(f"  Ask: {last_quote.ask_price:.4f}")
                print(f"  Bid size: {last_quote.bid_size}")
                print(f"  Ask size: {last_quote.ask_size}")
                print(f"  Midpoint: {summary.midpoint:.4f}")
                print(f"  Queue imbalance: {summary.queue_imbalance:.4f}")
                print(f"  Signed volume imbalance: {summary.signed_volume_imbalance:.4f}")
                print(f"  Relative spread: {summary.relative_spread:.4f}")
                print(f"  Midpoint return: {summary.midpoint_return:.4f}")
            else:
                print(f"\nMinute {i+1} ({boundary_dt.strftime('%H:%M')} {timezone_str}):")
                print(f"  Final actual quote timestamp: {boundary_utc.astimezone(pytz.timezone('America/New_York')).strftime('%H:%M:%S')} ET")
        else:
            print(f"\nMinute {i+1} ({boundary_dt.strftime('%H:%M')} {timezone_str}): INVALID - no valid quote")
    
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
    print(f"Symbol: {symbol}")
    print(f"Decision boundary: {decision_dt.strftime('%Y-%m-%d %H:%M:%S')} {timezone_str}")
    print(f"Target boundary: {target_dt.strftime('%Y-%m-%d %H:%M:%S')} {timezone_str}")
    print(f"P(UP): {prediction_result['p_up']:.4f}")
    print(f"P(DOWN): {prediction_result['p_down']:.4f}")
    print(f"Predicted direction: {prediction_result['direction']}")
    print(f"Decision midpoint: {decision_midpoint:.4f}")
    
    # Save prediction BEFORE loading target data
    prediction_output = {
        'symbol': symbol,
        'decision_boundary': decision_dt.isoformat(),
        'target_boundary': target_dt.isoformat(),
        'timezone': timezone_str,
        'p_up': prediction_result['p_up'],
        'p_down': prediction_result['p_down'],
        'predicted_direction': prediction_result['direction'],
        'decision_midpoint': decision_midpoint,
        'input_timestamps': [s.timestamp.astimezone(pytz.timezone('America/New_York')).isoformat() for s in minute_summaries],
        'features': features,
        'model_version': 'model-a-0.1.0',
        'is_replay': True
    }
    
    prediction_path = Path(f"artifacts/model_a/predictions/{symbol}_{date}_{decision_time.replace(':', '')}_replay.json")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(prediction_path, 'w') as f:
        json.dump(prediction_output, f, indent=2)
    
    print(f"\nPrediction saved to {prediction_path}")
    
    # Now load target data (AFTER prediction is saved)
    print(f"\n" + "=" * 80)
    print("TARGET / RESULT (loading data after prediction saved)")
    print("=" * 80)
    
    # Find target midpoint from (T+4, T+5]
    target_utc = target_dt.astimezone(timezone.utc)
    target_window_start = target_utc - timedelta(minutes=1)
    
    target_window_events = [
        e for e in all_events
        if target_window_start < e.timestamp <= target_utc
    ]
    
    # Find last valid quote in target window
    last_target_quote = None
    for event in reversed(target_window_events):
        if event.is_quote() and event.has_valid_quote():
            last_target_quote = event
            break
    
    if last_target_quote is None:
        print(f"\nERROR: No valid quote found in target window ({target_window_start} to {target_utc})")
        print(f"Expected window: (18:04, 18:05] UTC for 13:05 CT target")
        print("This may indicate data gaps or the target boundary is outside available data")
        return
    
    target_midpoint = (last_target_quote.bid_price + last_target_quote.ask_price) / 2.0
    
    print(f"\nTarget minute:")
    print(f"  Boundary: {target_dt.strftime('%Y-%m-%d %H:%M:%S')} {timezone_str}")
    print(f"  Expected UTC window: ({target_window_start.strftime('%H:%M')}, {target_utc.strftime('%H:%M')}] UTC")
    print(f"  Actual quote timestamp: {last_target_quote.timestamp.astimezone(pytz.timezone('America/New_York')).strftime('%H:%M:%S')} ET")
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
    prediction_output['target_quote_timestamp'] = last_target_quote.timestamp.astimezone(pytz.timezone('America/New_York')).isoformat()
    prediction_output['target_bid'] = last_target_quote.bid_price
    prediction_output['target_ask'] = last_target_quote.ask_price
    
    with open(prediction_path, 'w') as f:
        json.dump(prediction_output, f, indent=2)
    
    print(f"\nFinal result saved to {prediction_path}")
    
    print(f"\n" + "=" * 80)
    print("REPLAY COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Replay Model A v1 on historical Alpaca IEX data')
    parser.add_argument('--symbol', required=True, help='Symbol (e.g., NVDA)')
    parser.add_argument('--date', required=True, help='Date (YYYY-MM-DD)')
    parser.add_argument('--decision-time', required=True, help='Decision time (HH:MM)')
    parser.add_argument('--timezone', default='America/Chicago', help='Timezone (default: America/Chicago)')
    
    args = parser.parse_args()
    
    run_replay(args.symbol, args.date, args.decision_time, args.timezone)


if __name__ == '__main__':
    main()
