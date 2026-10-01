"""Microstructure feature calculation for one-minute trailing summaries."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Tuple

import pytz

from .data_loader import Event
from .trade_signing import TradeSigner


@dataclass
class MinuteSummary:
    """One-minute trailing summary ending at decision time."""
    timestamp: datetime  # Decision time (end of trailing window)
    queue_imbalance: float
    signed_volume_imbalance: float
    relative_spread: float
    midpoint_return: float
    midpoint: float  # Actual midpoint value
    has_valid_quote: bool
    trade_count: int
    
    def to_feature_vector(self) -> List[float]:
        """Convert to 4-element feature vector."""
        return [
            self.queue_imbalance,
            self.signed_volume_imbalance,
            self.relative_spread,
            self.midpoint_return
        ]


class FeatureCalculator:
    """Calculates microstructure features with causal constraints."""
    
    def __init__(self):
        self.ny_tz = pytz.timezone("America/New_York")
        self.trade_signer = TradeSigner()
        self.previous_midpoint: Optional[float] = None
        self.previous_minute_time: Optional[datetime] = None
    
    def calculate_minute_summary(self, 
                                 events: List[Event],
                                 decision_time: datetime) -> Optional[MinuteSummary]:
        """Calculate one-minute trailing summary ending at decision_time.
        
        The summary includes only events with timestamp <= decision_time.
        This is a trailing window: (decision_time - 1 minute, decision_time].
        
        Args:
            events: List of events in the minute window
            decision_time: End of the trailing window (decision time)
            
        Returns:
            MinuteSummary or None if no valid quote
        """
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
            # No valid quote in the window - invalid minute
            return None
        
        # Calculate quote-based features
        midpoint = (last_quote.bid_price + last_quote.ask_price) / 2.0
        queue_imbalance = self._calculate_queue_imbalance(
            last_quote.bid_size, last_quote.ask_size)
        relative_spread = self._calculate_relative_spread(
            last_quote.bid_price, last_quote.ask_price, midpoint)
        
        # Calculate trade-based features with causal signing
        signed_volume, trade_count = self._calculate_signed_volume(
            causal_events, last_quote.timestamp)
        
        # Normalize signed volume by total volume
        total_volume = sum(e.trade_size for e in causal_events 
                          if e.is_trade() and e.has_valid_trade())
        if total_volume > 0:
            signed_volume_imbalance = signed_volume / total_volume
        else:
            signed_volume_imbalance = 0.0
        
        # Calculate midpoint return (requires previous minute's midpoint)
        if self.previous_midpoint is not None and self.previous_minute_time is not None:
            midpoint_return = (midpoint - self.previous_midpoint) / self.previous_midpoint
        else:
            midpoint_return = 0.0  # First observation
        
        # Update state for next minute
        self.previous_midpoint = midpoint
        self.previous_minute_time = decision_time
        
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
    
    def _calculate_queue_imbalance(self, bid_size: float, ask_size: float) -> float:
        """Calculate queue imbalance from bid/ask sizes.
        
        Uses raw Alpaca IEX quote-size units as provided.
        """
        if bid_size + ask_size == 0:
            return 0.0
        return (bid_size - ask_size) / (bid_size + ask_size)
    
    def _calculate_relative_spread(self, bid_price: float, 
                                    ask_price: float, midpoint: float) -> float:
        """Calculate relative spread."""
        if midpoint == 0:
            return 0.0
        return (ask_price - bid_price) / midpoint
    
    def _calculate_signed_volume(self, events: List[Event], 
                                 quote_time: datetime) -> Tuple[float, int]:
        """Calculate signed trade volume with causal signing.
        
        Each trade is signed using the most recent quote at or before 
        that specific trade's timestamp.
        """
        signed_volume = 0.0
        trade_count = 0
        
        # Reset trade signer for this minute
        self.trade_signer = TradeSigner()
        
        for event in events:
            if event.is_quote() and event.has_valid_quote():
                self.trade_signer.update_quote(event)
            elif event.is_trade() and event.has_valid_trade():
                sign = self.trade_signer.sign_trade(event)
                if sign is not None:
                    signed_volume += sign * event.trade_size
                    trade_count += 1
        
        return signed_volume, trade_count
    
    def reset_state(self) -> None:
        """Reset calculator state for new session."""
        self.previous_midpoint = None
        self.previous_minute_time = None
        self.trade_signer = TradeSigner()


def create_minute_boundaries(start_date: str, end_date: str,
                            start_hour: int = 9, start_minute: int = 35,
                            end_hour: int = 12, end_minute: int = 0) -> List[datetime]:
    """Create list of minute decision times for observation window.
    
    Args:
        start_date: YYYY-MM-DD start date
        end_date: YYYY-MM-DD end date  
        start_hour: Start hour in ET
        start_minute: Start minute in ET
        end_hour: End hour in ET
        end_minute: End minute in ET
        
    Returns:
        List of datetime objects in ET representing decision times
    """
    ny_tz = pytz.timezone("America/New_York")
    
    start_dt = ny_tz.localize(
        datetime.strptime(start_date, "%Y-%m-%d").replace(
            hour=start_hour, minute=start_minute, second=0, microsecond=0))
    end_dt = ny_tz.localize(
        datetime.strptime(end_date, "%Y-%m-%d").replace(
            hour=end_hour, minute=end_minute, second=0, microsecond=0))
    
    boundaries = []
    current = start_dt
    while current <= end_dt:
        boundaries.append(current)
        current += timedelta(minutes=1)
    
    return boundaries
