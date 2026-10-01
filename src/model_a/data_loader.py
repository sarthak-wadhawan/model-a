"""Data loader for streaming events.csv with memory-efficient chunked reading."""

import csv
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterator, Optional

import pytz


class Event:
    """Single market event from events.csv."""
    
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


class EventStream:
    """Memory-efficient streaming reader for events.csv."""
    
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.ny_tz = pytz.timezone("America/New_York")
    
    def _parse_timestamp(self, ts_str: str) -> datetime:
        """Parse UTC timestamp string to datetime."""
        # Remove 'Z' suffix if present and parse as UTC
        ts_str = ts_str.rstrip('Z')
        # Handle timestamps with extra precision (more than 6 decimal places)
        if '.' in ts_str:
            # Split at decimal point
            parts = ts_str.split('.')
            # Keep only first 6 digits of microseconds
            if len(parts) > 1:
                microsecond_part = parts[1][:6]
                ts_str = f"{parts[0]}.{microsecond_part}"
        
        # Use datetime.strptime for parsing
        try:
            dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%f")
        except ValueError:
            try:
                dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                raise ValueError(f"Unable to parse timestamp: {ts_str}")
        return dt.replace(tzinfo=timezone.utc)
    
    def _to_et(self, utc_dt: datetime) -> datetime:
        """Convert UTC datetime to Eastern Time."""
        return utc_dt.astimezone(self.ny_tz)
    
    def events(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Iterator[Event]:
        """Stream events from CSV one at a time, optionally filtered by date.
        
        Args:
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            
        Yields:
            Events within the specified date range
        """
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamp = self._parse_timestamp(row['timestamp'])
                
                # Filter by date if specified
                if start_date or end_date:
                    date_str = timestamp.strftime("%Y-%m-%d")
                    if start_date and date_str < start_date:
                        continue
                    if end_date and date_str > end_date:
                        continue
                
                # Parse numeric fields
                bid_price = self._parse_float(row['bid_price_1'])
                ask_price = self._parse_float(row['ask_price_1'])
                bid_size = self._parse_float(row['bid_size_1'])
                ask_size = self._parse_float(row['ask_size_1'])
                trade_price = self._parse_float(row['trade_price'])
                trade_size = self._parse_float(row['trade_size'])
                aggressor_side = self._parse_int(row['aggressor_side'])
                
                yield Event(
                    timestamp=timestamp,
                    event_type=row['event_type'],
                    bid_price=bid_price,
                    ask_price=ask_price,
                    bid_size=bid_size,
                    ask_size=ask_size,
                    trade_price=trade_price,
                    trade_size=trade_size,
                    aggressor_side=aggressor_side
                )
    
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


def filter_by_session(events: Iterator[Event], 
                      start_date: str, 
                      end_date: str,
                      start_hour: int = 9,
                      start_minute: int = 35,
                      end_hour: int = 12,
                      end_minute: int = 0) -> Iterator[Event]:
    """Filter events to specific date range and observation window.
    
    Args:
        events: Event iterator
        start_date: YYYY-MM-DD start date (inclusive)
        end_date: YYYY-MM-DD end date (inclusive)
        start_hour: Start hour in ET (default 9 for 09:35)
        start_minute: Start minute in ET (default 35 for 09:35)
        end_hour: End hour in ET (default 12 for 12:00)
        end_minute: End minute in ET (default 0 for 12:00)
    
    Yields:
        Events within the specified observation window
    """
    ny_tz = pytz.timezone("America/New_York")
    
    start_dt = ny_tz.localize(datetime.strptime(start_date, "%Y-%m-%d").replace(
        hour=start_hour, minute=start_minute, second=0, microsecond=0))
    end_dt = ny_tz.localize(datetime.strptime(end_date, "%Y-%m-%d").replace(
        hour=end_hour, minute=end_minute, second=0, microsecond=0))
    
    for event in events:
        et_time = event.timestamp.astimezone(ny_tz)
        if start_dt <= et_time <= end_dt:
            yield event
