"""Immediate live prediction using frozen Model A v1 for any ticker."""

import sys
sys.path.insert(0, 'src')

import json
import os
import re
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
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


def parse_timestamp(ts) -> datetime:
    """Parse various timestamp formats to timezone-aware UTC datetime."""
    import pandas as pd
    
    if isinstance(ts, pd.Timestamp):
        result = ts.to_pydatetime()
        if result.tzinfo is None:
            result = result.replace(tzinfo=timezone.utc)
        return result
    elif isinstance(ts, str):
        ts_str = ts.replace('Z', '+00:00')
        
        if '.' in ts_str:
            match = re.match(r'(.*\.\d{6}).*', ts_str)
            if match:
                ts_str = match.group(1)
        
        try:
            result = datetime.fromisoformat(ts_str)
        except Exception as e:
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
    
    def fetch_stock_quotes(self, symbol: str, start: datetime, end: datetime) -> List[Event]:
        """Fetch historical IEX stock quotes for a time range."""
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
            print(f"Error fetching quotes with iex, trying sip: {e}")
            quotes = self.api.get_quotes(
                symbol,
                start=start_utc.isoformat(),
                end=end_utc.isoformat(),
                feed='sip'
            )
        
        events = []
        for quote in quotes:
            timestamp = parse_timestamp(quote.t)
            
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
        
        return events
    
    def fetch_stock_trades(self, symbol: str, start: datetime, end: datetime) -> List[Event]:
        """Fetch historical IEX stock trades for a time range."""
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
            print(f"Error fetching trades with iex, trying sip: {e}")
            trades = self.api.get_trades(
                symbol,
                start=start_utc.isoformat(),
                end=end_utc.isoformat(),
                feed='sip'
            )
        
        events = []
        for trade in trades:
            timestamp = parse_timestamp(trade.t)
            
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
                aggressor_side=None
            ))
        
        return events


def calculate_minute_summary_from_events(events: List[Event], 
                                           decision_time: datetime,
                                           calculator: FeatureCalculator) -> Optional[MinuteSummary]:
    """Calculate minute summary using exact Model A v1 logic with quote/trade events."""
    if not events:
        return None
    
    causal_events = [e for e in events if e.timestamp <= decision_time]
    
    if not causal_events:
        return None
    
    last_quote = None
    for event in reversed(causal_events):
        if event.is_quote() and event.has_valid_quote():
            last_quote = event
            break
    
    if last_quote is None:
        return None
    
    midpoint = (last_quote.bid_price + last_quote.ask_price) / 2.0
    queue_imbalance = calculator._calculate_queue_imbalance(
        last_quote.bid_size, last_quote.ask_size)
    relative_spread = calculator._calculate_relative_spread(
        last_quote.bid_price, last_quote.ask_price, midpoint)
    
    signed_volume, trade_count = calculator._calculate_signed_volume(
        causal_events, last_quote.timestamp)
    
    total_volume = sum(e.trade_size for e in causal_events 
                      if e.is_trade() and e.has_valid_trade())
    if total_volume > 0:
        signed_volume_imbalance = signed_volume / total_volume
    else:
        signed_volume_imbalance = 0.0
    
    if calculator.previous_midpoint is not None and calculator.previous_minute_time is not None:
        midpoint_return = (midpoint - calculator.previous_midpoint) / calculator.previous_midpoint
    else:
        midpoint_return = 0.0
    
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
    checkpoint = torch.load(model_path, map_location='cpu')
    model = FixedMLP(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        seed=seed
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler = Scaler.load(scaler_path)
    
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


def validate_ticker_data(fetcher, symbol: str, initial_time: datetime) -> bool:
    """Validate that the ticker returns usable Alpaca IEX data."""
    chicago_tz = pytz.timezone("America/Chicago")
    
    current_chicago = initial_time.astimezone(chicago_tz)
    test_boundary = current_chicago.replace(second=0, microsecond=0) - timedelta(minutes=1)
    test_boundary_utc = test_boundary.astimezone(timezone.utc)
    
    window_start = test_boundary_utc - timedelta(minutes=2)
    window_end = test_boundary_utc + timedelta(seconds=30)
    
    print(f"Validating ticker {symbol}...")
    
    try:
        quotes = fetcher.fetch_stock_quotes(symbol, window_start, window_end)
        trades = fetcher.fetch_stock_trades(symbol, window_start, window_end)
        
        if not quotes and not trades:
            print(f"ERROR: No usable Alpaca IEX data found for {symbol}")
            return False
        
        print(f"Ticker validation passed: {len(quotes)} quotes, {len(trades)} trades")
        return True
    except Exception as e:
        print(f"ERROR: No usable Alpaca IEX data found for {symbol}: {e}")
        return False


def find_usable_decision_boundary(fetcher, symbol: str, initial_time: datetime) -> datetime:
    """Find the most recent completed minute with available data."""
    chicago_tz = pytz.timezone("America/Chicago")
    
    current_chicago = initial_time.astimezone(chicago_tz)
    decision_boundary = current_chicago.replace(second=0, microsecond=0) - timedelta(minutes=1)
    
    print(f"Searching for usable decision boundary starting from {decision_boundary.strftime('%H:%M:%S')} CT...")
    
    for offset in range(10):
        test_boundary = decision_boundary - timedelta(minutes=offset)
        test_boundary_utc = test_boundary.astimezone(timezone.utc)
        
        window_start = test_boundary_utc - timedelta(minutes=1)
        window_end = test_boundary_utc + timedelta(seconds=30)
        
        try:
            quotes = fetcher.fetch_stock_quotes(symbol, window_start, window_end)
            if quotes:
                print(f"Found usable boundary: {test_boundary.strftime('%H:%M:%S')} CT")
                return test_boundary
        except Exception as e:
            print(f"Data unavailable for {test_boundary.strftime('%H:%M:%S')} CT: {e}")
            continue
    
    raise ValueError("Could not find a usable decision boundary with available data")


def run_live_prediction(symbol: str):
    """Run immediate live prediction for the given symbol."""
    symbol = symbol.upper()
    
    print("=" * 80)
    print(f"LIVE {symbol} PREDICTION - FROZEN MODEL A v1")
    print("=" * 80)
    
    chicago_tz = pytz.timezone("America/Chicago")
    ny_tz = pytz.timezone("America/New_York")
    
    now = datetime.now(timezone.utc)
    now_chicago = now.astimezone(chicago_tz)
    print(f"\nCurrent time: {now_chicago.strftime('%Y-%m-%d %H:%M:%S')} CT")
    
    fetcher = HistoricalEventFetcher()
    
    # Validate ticker data first
    if not validate_ticker_data(fetcher, symbol, now):
        sys.exit(1)
    
    decision_boundary = find_usable_decision_boundary(fetcher, symbol, now)
    decision_boundary_utc = decision_boundary.astimezone(timezone.utc)
    decision_boundary_et = decision_boundary.astimezone(ny_tz)
    
    print(f"\nSelected decision boundary: {decision_boundary.strftime('%Y-%m-%d %H:%M:%S')} CT")
    print(f"Equivalent ET: {decision_boundary_et.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Equivalent UTC: {decision_boundary_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    target_boundary = decision_boundary + timedelta(minutes=5)
    target_boundary_utc = target_boundary.astimezone(timezone.utc)
    
    print(f"Target boundary: {target_boundary.strftime('%Y-%m-%d %H:%M:%S')} CT")
    print(f"Equivalent UTC: {target_boundary_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    data_start = decision_boundary - timedelta(minutes=6)
    data_end = decision_boundary + timedelta(minutes=1)
    
    print(f"\nFetching data from {data_start.strftime('%H:%M')} CT to {data_end.strftime('%H:%M')} CT...")
    
    quotes = fetcher.fetch_stock_quotes(symbol, data_start, data_end)
    trades = fetcher.fetch_stock_trades(symbol, data_start, data_end)
    
    print(f"Fetched {len(quotes)} quotes, {len(trades)} trades")
    
    all_events = quotes + trades
    all_events.sort(key=lambda e: e.timestamp)
    
    calculator = FeatureCalculator()
    calculator.reset_state()
    
    pre_roll_boundary = decision_boundary - timedelta(minutes=5)
    pre_roll_boundary_utc = pre_roll_boundary.astimezone(timezone.utc)
    pre_roll_window_start = pre_roll_boundary_utc - timedelta(minutes=1)
    
    pre_roll_events = [
        e for e in all_events 
        if pre_roll_window_start < e.timestamp <= pre_roll_boundary_utc
    ]
    
    pre_roll_summary = calculate_minute_summary_from_events(pre_roll_events, pre_roll_boundary_utc, calculator)
    
    if pre_roll_summary:
        calculator.previous_midpoint = pre_roll_summary.midpoint
        calculator.previous_minute_time = pre_roll_boundary_utc
        print(f"Pre-roll midpoint ({pre_roll_boundary.strftime('%H:%M')} CT): {pre_roll_summary.midpoint:.4f}")
    else:
        print(f"WARNING: No pre-roll midpoint available")
    
    minute_summaries = []
    input_boundaries = []
    
    for i in range(5):
        offset = 4 - i
        boundary_dt = decision_boundary - timedelta(minutes=offset)
        boundary_utc = boundary_dt.astimezone(timezone.utc)
        input_boundaries.append(boundary_dt)
        
        window_start = boundary_utc - timedelta(minutes=1)
        window_events = [
            e for e in all_events 
            if window_start < e.timestamp <= boundary_utc
        ]
        
        summary = calculate_minute_summary_from_events(window_events, boundary_utc, calculator)
        
        if summary:
            minute_summaries.append(summary)
        else:
            print(f"ERROR: No valid quote for {boundary_dt.strftime('%H:%M')} CT")
            return
    
    if len(minute_summaries) != 5:
        print(f"ERROR: Expected 5 summaries, got {len(minute_summaries)}")
        return
    
    features = []
    for summary in minute_summaries:
        features.extend(summary.to_feature_vector())
    
    model_path = Path("artifacts/model_a/models/model.pt")
    scaler_path = Path("artifacts/model_a/scalers/scaler.json")
    
    model, scaler = load_frozen_model(model_path, scaler_path)
    
    prediction_result = predict_with_frozen_model(model, scaler, features)
    decision_midpoint = minute_summaries[-1].midpoint
    
    prediction_created = datetime.now(timezone.utc)
    
    # Check if decision time is outside training window (09:35-12:00 ET)
    training_start_et = decision_boundary_et.replace(hour=9, minute=35, second=0, microsecond=0)
    training_end_et = decision_boundary_et.replace(hour=12, minute=0, second=0, microsecond=0)
    is_out_of_distribution_time = decision_boundary_et < training_start_et or decision_boundary_et > training_end_et
    
    # PRINT PREDICTION PROMINENTLY
    print("\n" + "=" * 80)
    print(f"LIVE {symbol} PREDICTION")
    print("=" * 80)
    print(f"Decision time CT: {decision_boundary.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target time CT: {target_boundary.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nDecision midpoint: {decision_midpoint:.4f}")
    print(f"\nP(UP): {prediction_result['p_up']:.4f}")
    print(f"P(DOWN): {prediction_result['p_down']:.4f}")
    print(f"\nPREDICTION: {prediction_result['direction']}")
    print("=" * 80)
    
    # Build Markdown report
    md_content = f"""# Live {symbol} Prediction

## Summary

- Symbol: {symbol}
- Decision time CT: {decision_boundary.strftime('%Y-%m-%d %H:%M:%S')}
- Decision time ET: {decision_boundary_et.strftime('%Y-%m-%d %H:%M:%S')}
- Decision time UTC: {decision_boundary_utc.strftime('%Y-%m-%d %H:%M:%S')}
- Target time CT: {target_boundary.strftime('%Y-%m-%d %H:%M:%S')}
- Prediction-created timestamp: {prediction_created.strftime('%Y-%m-%d %H:%M:%S')} UTC
- Decision midpoint: {decision_midpoint:.4f}
- P(UP): {prediction_result['p_up']:.4f}
- P(DOWN): {prediction_result['p_down']:.4f}
- Predicted direction: {prediction_result['direction']}

## Input Minutes

|| Boundary CT | Quote Timestamp | Midpoint | Queue Imbalance | Signed Volume Imbalance | Relative Spread | Midpoint Return |
||---|---|---:|---:|---:|---:|---:|
"""
    
    for boundary, summary in zip(input_boundaries, minute_summaries):
        quote_ts = summary.timestamp.astimezone(ny_tz).strftime('%H:%M:%S')
        md_content += f"| {boundary.strftime('%H:%M')} | {quote_ts} ET | {summary.midpoint:.4f} | {summary.queue_imbalance:.4f} | {summary.signed_volume_imbalance:.4f} | {summary.relative_spread:.4f} | {summary.midpoint_return:.4f} |\n"
    
    md_content += f"""
## Raw Feature Vector

Complete ordered 20-feature vector:

"""
    
    for i, f in enumerate(features):
        md_content += f"Feature {i+1}: {f:.6f}\n"
    
    md_content += f"""
## Model

- Frozen Model A v1
- Existing saved scaler
- Existing saved MLP
- No retraining
- No tuning

## Status

LIVE {symbol} TEST

"""
    
    if symbol == "SPY":
        md_content += "TRAINED SYMBOL: SPY\n"
    else:
        md_content += f"""
OUT-OF-DISTRIBUTION SYMBOL
MODEL A v1 WAS TRAINED ON SPY
PREDICTION IS EXPERIMENTAL
"""
    
    if is_out_of_distribution_time:
        md_content += f"""
TIME-OF-DAY OUT-OF-DISTRIBUTION

Decision time {decision_boundary_et.strftime('%H:%M')} ET is outside the original training window (09:35-12:00 ET).
"""
    
    md_content += f"""
TARGET OUTCOME HAS NOT YET BEEN USED.
"""
    
    # Save latest file
    latest_path = Path(f"artifacts/model_a/predictions/latest_{symbol}_live.md")
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(latest_path, 'w') as f:
        f.write(md_content)
    
    print(f"\nSaved latest report to: {latest_path}")
    
    # Save timestamped archival file
    timestamp_str = prediction_created.strftime('%Y-%m-%d_%H%M')
    archival_path = Path(f"artifacts/model_a/predictions/{symbol}_{timestamp_str}_live.md")
    
    with open(archival_path, 'w') as f:
        f.write(md_content)
    
    print(f"Saved archival report to: {archival_path}")
    
    print("\nPrediction complete. Stopping (not waiting for target).")


def main():
    parser = argparse.ArgumentParser(description='Live prediction using frozen Model A v1 for any ticker')
    parser.add_argument('--symbol', required=True, help='Stock/ETF ticker symbol (e.g., SPY, NVDA, AAPL)')
    args = parser.parse_args()
    
    run_live_prediction(args.symbol)


if __name__ == '__main__':
    main()
