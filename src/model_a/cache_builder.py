"""Incremental feature cache builder that processes events.csv once without accumulating Event objects."""

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, List
import json

import pytz

from .data_loader import Event
from .trade_signing import TradeSigner
from .features import FeatureCalculator


class MinuteState:
    """Maintains minimal state for a single minute's feature calculation."""
    
    def __init__(self, decision_time: datetime):
        self.decision_time = decision_time
        self.window_start = decision_time - timedelta(minutes=1)
        
        # Store minimal quote data (best quote at decision time)
        self.best_bid_price: Optional[float] = None
        self.best_ask_price: Optional[float] = None
        self.best_bid_size: Optional[float] = None
        self.best_ask_size: Optional[float] = None
        
        # Trade signing state
        self.trade_signer = TradeSigner()
        
        # Accumulated trade values
        self.signed_volume: float = 0.0
        self.trade_count: int = 0
        self.total_volume: float = 0.0
        
        # Buffer for events in this minute (needed for chronological processing)
        # This is necessary because trade signing requires chronological order
        # But we clear it after each minute to keep memory bounded
        self.minute_buffer: List[Event] = []


class IncrementalCacheBuilder:
    """Builds minute feature cache with single-pass streaming, no Event accumulation."""
    
    def __init__(self, events_csv_path: str):
        self.events_csv_path = Path(events_csv_path)
        self.ny_tz = pytz.timezone("America/New_York")
        
        # Required sessions
        self.required_dates = ['2026-08-18', '2026-08-19', '2026-08-20', '2026-08-21', '2026-08-24', '2026-08-25']
        
        # Observation window: 09:35-12:00 ET
        self.start_hour = 9
        self.start_minute = 35
        self.end_hour = 12
        self.end_minute = 0
        
        # Buffer: load 10 minutes before first observation for causal state
        self.buffer_minutes = 10
        
        # Store only completed minute summaries
        self.minute_summaries: Dict[str, List[Dict]] = {date: [] for date in self.required_dates}
        
        # Current minute state being built
        self.current_minute_state: Optional[MinuteState] = None
        self.current_decision_time: Optional[datetime] = None
        
        # Previous midpoint for return calculation
        self.previous_midpoint: Optional[float] = None
        
        # Statistics
        self.total_events_processed = 0
        self.events_by_date: Dict[str, int] = {date: 0 for date in self.required_dates}
        self.valid_summaries_by_date: Dict[str, int] = {date: 0 for date in self.required_dates}
    
    def _parse_timestamp(self, ts_str: str) -> datetime:
        """Parse UTC timestamp string to datetime."""
        ts_str = ts_str.rstrip('Z')
        if '.' in ts_str:
            parts = ts_str.split('.')
            if len(parts) > 1:
                microsecond_part = parts[1][:6]
                ts_str = f"{parts[0]}.{microsecond_part}"
        
        try:
            dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%f")
        except ValueError:
            try:
                dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                raise ValueError(f"Unable to parse timestamp: {ts_str}")
        return dt.replace(tzinfo=timezone.utc)
    
    def _parse_float(self, val: str) -> Optional[float]:
        """Parse float from string, return None if empty."""
        if not val or val.strip() == '':
            return None
        try:
            return float(val)
        except ValueError:
            return None
    
    def _parse_int(self, val: str) -> Optional[int]:
        """Parse int from string, return None if empty."""
        if not val or val.strip() == '':
            return None
        try:
            return int(val)
        except ValueError:
            return None
    
    def _is_in_observation_window(self, utc_dt: datetime) -> bool:
        """Check if datetime is in observation window (09:35-12:00 ET)."""
        et_time = utc_dt.astimezone(self.ny_tz)
        hour = et_time.hour
        minute = et_time.minute
        
        # Convert to 24-hour format for comparison
        start_hour_24 = self.start_hour if self.start_hour != 12 else 0
        end_hour_24 = self.end_hour if self.end_hour != 12 else 0
        
        if hour < start_hour_24 or hour > end_hour_24:
            return False
        if hour == start_hour_24 and minute < self.start_minute:
            return False
        if hour == end_hour_24 and minute >= self.end_minute:
            return False
        return True
    
    def _get_minute_key(self, dt: datetime) -> str:
        """Get minute key for datetime."""
        et_time = dt.astimezone(self.ny_tz)
        return et_time.strftime("%Y-%m-%d %H:%M")
    
    def _initialize_minute_state(self, decision_time: datetime):
        """Initialize state for a new minute."""
        self.current_minute_state = MinuteState(decision_time)
        self.current_decision_time = decision_time
    
    def _process_event_in_current_minute(self, event: Event):
        """Store event in minute buffer for chronological processing."""
        if event.timestamp <= self.current_minute_state.decision_time and \
           event.timestamp > self.current_minute_state.window_start:
            
            # Store in buffer for chronological processing
            self.current_minute_state.minute_buffer.append(event)
            
            # Track best quote (last valid quote at/before decision time)
            if event.is_quote() and event.has_valid_quote():
                if event.timestamp <= self.current_minute_state.decision_time:
                    self.current_minute_state.best_bid_price = event.bid_price
                    self.current_minute_state.best_ask_price = event.ask_price
                    self.current_minute_state.best_bid_size = event.bid_size
                    self.current_minute_state.best_ask_size = event.ask_size
    
    def _finalize_minute_summary(self) -> Optional[Dict]:
        """Finalize the current minute and produce a summary."""
        if self.current_minute_state is None or self.current_minute_state.best_bid_price is None:
            return None
        
        state = self.current_minute_state
        
        # Process buffered events chronologically for trade signing
        sorted_events = sorted(state.minute_buffer, key=lambda e: e.timestamp)
        
        for event in sorted_events:
            if event.is_quote() and event.has_valid_quote():
                state.trade_signer.update_quote(event)
            elif event.is_trade() and event.has_valid_trade():
                sign = state.trade_signer.sign_trade(event)
                if sign is not None:
                    state.signed_volume += sign * event.trade_size
                    state.trade_count += 1
                state.total_volume += event.trade_size
        
        # Clear the buffer to free memory
        state.minute_buffer.clear()
        
        # Calculate features from accumulated state
        midpoint = (state.best_bid_price + state.best_ask_price) / 2.0
        
        calculator = FeatureCalculator()
        queue_imbalance = calculator._calculate_queue_imbalance(
            state.best_bid_size, state.best_ask_size)
        relative_spread = calculator._calculate_relative_spread(
            state.best_bid_price, state.best_ask_price, midpoint)
        
        # Normalize signed volume
        if state.total_volume > 0:
            signed_volume_imbalance = state.signed_volume / state.total_volume
        else:
            signed_volume_imbalance = 0.0
        
        # Calculate midpoint return
        if self.previous_midpoint is not None:
            midpoint_return = (midpoint - self.previous_midpoint) / self.previous_midpoint
        else:
            midpoint_return = 0.0
        
        # Update state for next minute
        self.previous_midpoint = midpoint
        
        # Build summary dictionary
        summary = {
            'timestamp': state.decision_time.isoformat(),
            'date': state.decision_time.astimezone(self.ny_tz).strftime("%Y-%m-%d"),
            'time': state.decision_time.astimezone(self.ny_tz).strftime("%H:%M"),
            'queue_imbalance': queue_imbalance,
            'signed_volume_imbalance': signed_volume_imbalance,
            'relative_spread': relative_spread,
            'midpoint_return': midpoint_return,
            'midpoint': midpoint,
            'trade_count': state.trade_count,
            'has_valid_quote': True
        }
        
        return summary
    
    def build_cache(self, output_path: Path) -> Dict:
        """Build feature cache from events.csv with single-pass streaming.
        
        Args:
            output_path: Path to save the cache CSV
            
        Returns:
            Statistics dictionary
        """
        print(f"Building feature cache from {self.events_csv_path}")
        print(f"Required dates: {self.required_dates}")
        print(f"Observation window: {self.start_hour:02d}:{self.start_minute:02d}-{self.end_hour:02d}:{self.end_minute:02d} ET")
        
        # Calculate start date with buffer
        first_date = min(self.required_dates)
        start_dt = self.ny_tz.localize(datetime.strptime(first_date, "%Y-%m-%d").replace(
            hour=self.start_hour, minute=self.start_minute, second=0, microsecond=0))
        buffer_dt = start_dt - timedelta(minutes=self.buffer_minutes)
        buffer_date = buffer_dt.strftime("%Y-%m-%d")
        
        last_date = max(self.required_dates)
        
        print(f"Processing events from {buffer_date} to {last_date}...")
        
        # Process events incrementally
        with open(self.events_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                self.total_events_processed += 1
                
                if self.total_events_processed % 1000000 == 0:
                    print(f"Processed {self.total_events_processed} events...")
                
                # Parse timestamp
                timestamp = self._parse_timestamp(row['timestamp'])
                date_str = timestamp.strftime("%Y-%m-%d")
                
                # Skip if before buffer date or after last date
                if date_str < buffer_date or date_str > last_date:
                    continue
                
                # Count events by date
                if date_str in self.events_by_date:
                    self.events_by_date[date_str] += 1
                
                # Create event object
                event = Event(
                    timestamp=timestamp,
                    event_type=row['event_type'],
                    bid_price=self._parse_float(row['bid_price_1']),
                    ask_price=self._parse_float(row['ask_price_1']),
                    bid_size=self._parse_float(row['bid_size_1']),
                    ask_size=self._parse_float(row['ask_size_1']),
                    trade_price=self._parse_float(row['trade_price']),
                    trade_size=self._parse_float(row['trade_size']),
                    aggressor_side=self._parse_int(row['aggressor_side'])
                )
                
                # Check if we need to start a new minute
                current_minute_key = self._get_minute_key(timestamp)
                
                if self.current_decision_time is None:
                    # First event - initialize to start of first required date
                    first_required_dt = self.ny_tz.localize(datetime.strptime(first_date, "%Y-%m-%d").replace(
                        hour=self.start_hour, minute=self.start_minute, second=0, microsecond=0))
                    self._initialize_minute_state(first_required_dt)
                
                current_minute_key_expected = self._get_minute_key(self.current_decision_time)
                
                # Check if we've moved to a new minute
                if current_minute_key != current_minute_key_expected:
                    # Finalize current minute if it's in a required date
                    if self.current_decision_time.astimezone(self.ny_tz).strftime("%Y-%m-%d") in self.required_dates:
                        summary = self._finalize_minute_summary()
                        if summary is not None:
                            date_key = summary['date']
                            self.minute_summaries[date_key].append(summary)
                            self.valid_summaries_by_date[date_key] += 1
                    
                    # Move to next minute (advance until we catch up to current timestamp)
                    # Safety limit: don't advance more than 1000 minutes
                    advance_count = 0
                    max_advances = 1000
                    while self._get_minute_key(self.current_decision_time) != current_minute_key and advance_count < max_advances:
                        next_decision_time = self.current_decision_time + timedelta(minutes=1)
                        self._initialize_minute_state(next_decision_time)
                        advance_count += 1
                
                # Process event in current minute
                if date_str in self.required_dates and self._is_in_observation_window(timestamp):
                    self._process_event_in_current_minute(event)
        
        # Finalize last minute
        if self.current_decision_time and self.current_decision_time.astimezone(self.ny_tz).strftime("%Y-%m-%d") in self.required_dates:
            summary = self._finalize_minute_summary()
            if summary is not None:
                date_key = summary['date']
                self.minute_summaries[date_key].append(summary)
                self.valid_summaries_by_date[date_key] += 1
        
        print(f"Total events processed: {self.total_events_processed}")
        
        # Print statistics
        for date in self.required_dates:
            print(f"{date}: {self.events_by_date.get(date, 0)} events, {self.valid_summaries_by_date[date]} valid minute summaries")
        
        # Save cache to CSV
        print(f"Saving cache to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp', 'date', 'time', 'queue_imbalance', 'signed_volume_imbalance',
                'relative_spread', 'midpoint_return', 'midpoint', 'trade_count', 'has_valid_quote'
            ])
            writer.writeheader()
            
            for date in self.required_dates:
                for summary in self.minute_summaries[date]:
                    writer.writerow(summary)
        
        print(f"Cache saved successfully")
        
        return {
            'total_events_processed': self.total_events_processed,
            'events_by_date': self.events_by_date,
            'valid_summaries_by_date': self.valid_summaries_by_date,
            'total_valid_summaries': sum(self.valid_summaries_by_date.values())
        }