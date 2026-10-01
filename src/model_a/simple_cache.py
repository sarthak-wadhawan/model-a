"""Cache builder with corrected minute boundary semantics."""

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import pytz

from .trade_signing import TradeSigner


class Event:
    """Lightweight event representation for cache building."""
    
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


class SimpleCacheBuilder:
    """Builds minute feature cache with corrected boundary semantics."""
    
    def __init__(self, events_csv_path: str):
        self.events_csv_path = Path(events_csv_path)
        self.ny_tz = pytz.timezone("America/New_York")
        
        # Required sessions
        self.required_dates = ['2026-08-18', '2026-08-19', '2026-08-20', '2026-08-21', '2026-08-24', '2026-08-25']
        
        # Observation window: 09:35-12:00 ET (inclusive)
        self.start_hour = 9
        self.start_minute = 35
        self.end_hour = 12
        self.end_minute = 0
        
        # Target window: 12:00-12:05 ET (for target lookup)
        self.target_end_hour = 12
        self.target_end_minute = 5
        
        # Pre-roll: load 1 minute before observation window for 09:35 construction
        self.pre_roll_minutes = 1
        
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
    
    def _is_in_extended_window(self, utc_dt: datetime) -> bool:
        """Check if datetime is in extended window (09:34-12:05 ET inclusive for pre-roll + targets)."""
        et_time = utc_dt.astimezone(self.ny_tz)
        hour = et_time.hour
        minute = et_time.minute
        
        # Window: 09:34 to 12:05 ET (inclusive)
        # 09:34 for pre-roll to construct 09:35 summary
        # 12:05 for targets
        if hour < 9 or hour > 12:
            return False
        if hour == 9 and minute < 34:
            return False
        if hour == 12 and minute > 5:
            return False
        return True
    
    def build_cache(self, output_path: Path) -> Dict:
        """Build feature cache with corrected semantics."""
        print(f"Building feature cache from {self.events_csv_path}")
        print(f"Required dates: {self.required_dates}")
        print(f"Observation window: {self.start_hour:02d}:{self.start_minute:02d}-{self.end_hour:02d}:{self.end_minute:02d} ET")
        print(f"Target window: through {self.target_end_hour:02d}:{self.target_end_minute:02d} ET")
        
        minute_summaries: Dict[str, List[Dict]] = {date: [] for date in self.required_dates}
        
        # Load all required dates at once using pandas
        start_date = min(self.required_dates)
        end_date = max(self.required_dates)
        
        print(f"Loading events from {start_date} to {end_date} using pandas...")
        
        # Read in chunks for memory efficiency
        chunksize = 100000
        all_events = []
        
        for chunk in pd.read_csv(self.events_csv_path, chunksize=chunksize):
            # Filter by date
            chunk['date'] = chunk['timestamp'].str[:10]
            chunk = chunk[chunk['date'].isin(self.required_dates)]
            
            if not chunk.empty:
                all_events.append(chunk)
                self.total_events_processed += len(chunk)
        
        if all_events:
            full_df = pd.concat(all_events, ignore_index=True)
            
            # Count events by date
            for date in self.required_dates:
                date_count = len(full_df[full_df['date'] == date])
                self.events_by_date[date] = date_count
                print(f"{date}: {date_count} events")
        else:
            print("No events found in date range")
            return {
                'total_events_processed': 0,
                'events_by_date': self.events_by_date,
                'valid_summaries_by_date': self.valid_summaries_by_date,
                'total_valid_summaries': 0
            }
        
        # Process each date
        for date in self.required_dates:
            print(f"Processing {date}...")
            
            date_df = full_df[full_df['date'] == date].copy()
            print(f"  Total events: {len(date_df)}")
            
            # Parse timestamps
            date_df['timestamp_dt'] = date_df['timestamp'].apply(self._parse_timestamp)
            
            # Filter to extended window (observation + target)
            date_df['in_extended_window'] = date_df['timestamp_dt'].apply(self._is_in_extended_window)
            date_df = date_df[date_df['in_extended_window']]
            
            print(f"  Events in extended window: {len(date_df)}")
            
            if date_df.empty:
                print(f"  No events in extended window for {date}")
                continue
            
            # Group by minute for trailing window semantics
            # Summary labelled T represents events in (T-1, T]
            # So events at 09:35:00.016 should go to 09:36 summary (representing 09:35-09:36)
            # Events at 09:35:00.000 should go to 09:35 summary (representing 09:34-09:35)
            def get_trailing_minute_key(dt):
                et = dt.astimezone(self.ny_tz)
                # If event is exactly on the minute boundary, it belongs to current minute's summary
                if et.second == 0 and et.microsecond == 0:
                    return et.strftime("%Y-%m-%d %H:%M")
                else:
                    # Event is after the boundary, belongs to next minute's summary
                    et_next = et + timedelta(minutes=1)
                    et_next = et_next.replace(second=0, microsecond=0)
                    return et_next.strftime("%Y-%m-%d %H:%M")
            
            date_df['minute_key'] = date_df['timestamp_dt'].apply(get_trailing_minute_key)
            
            print(f"  Unique minutes: {date_df['minute_key'].nunique()}")
            
            # Process each minute
            for minute_key, minute_df in date_df.groupby('minute_key'):
                # Determine if this is a target-only row
                et_time = self.ny_tz.localize(datetime.strptime(minute_key, "%Y-%m-%d %H:%M"))
                # Input-eligible: 09:35-12:00
                # Target-only: 12:01-12:05
                # 12:06 should be excluded entirely (after target window)
                is_target_only = (et_time.hour == 12 and et_time.minute >= 1 and et_time.minute <= 5)
                is_excluded = (et_time.hour == 12 and et_time.minute > 5)
                
                if is_excluded:
                    continue  # Skip 12:06 and beyond
                
                summary = self._process_minute(minute_df, minute_key, is_target_only)
                if summary is not None:
                    minute_summaries[date].append(summary)
                    # Only count input-eligible summaries as valid
                    if not is_target_only:
                        self.valid_summaries_by_date[date] += 1
        
        # Save cache
        print(f"Saving cache to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp', 'date', 'time', 'queue_imbalance', 'signed_volume_imbalance',
                'relative_spread', 'midpoint_return', 'midpoint', 'trade_count', 'has_valid_quote', 'is_target_only',
                'quote_timestamp', 'bid', 'ask'
            ])
            writer.writeheader()
            
            for date in self.required_dates:
                for summary in minute_summaries[date]:
                    writer.writerow(summary)
        
        print(f"Cache saved successfully")
        
        return {
            'total_events_processed': self.total_events_processed,
            'events_by_date': self.events_by_date,
            'valid_summaries_by_date': self.valid_summaries_by_date,
            'total_valid_summaries': sum(self.valid_summaries_by_date.values())
        }
    
    def _process_minute(self, minute_df: pd.DataFrame, minute_key: str, is_target_only: bool = False) -> Optional[Dict]:
        """Process a single minute's events to produce a summary.
        
        Args:
            minute_df: DataFrame of events for this minute
            minute_key: Minute key in format "YYYY-MM-DD HH:MM"
            is_target_only: Whether this row is for target lookup only (not model input)
        """
        # Sort by timestamp
        minute_df = minute_df.sort_values('timestamp_dt')
        
        # Parse the canonical boundary from the key
        et_dt = self.ny_tz.localize(datetime.strptime(minute_key, "%Y-%m-%d %H:%M"))
        boundary_time = et_dt.astimezone(timezone.utc)
        window_start = boundary_time - timedelta(minutes=1)
        
        # Filter to trailing window (T-1, T]
        # Keep only events with timestamp > T-1 and <= T
        minute_df = minute_df[
            (minute_df['timestamp_dt'] > window_start) &
            (minute_df['timestamp_dt'] <= boundary_time)
        ]
        
        if minute_df.empty:
            return None
        
        # Find last valid quote within the trailing window
        quotes = minute_df[minute_df['event_type'] == 'quote'].copy()
        quotes = quotes[
            (quotes['bid_price_1'].notna()) & 
            (quotes['ask_price_1'].notna()) &
            (quotes['bid_size_1'].notna()) &
            (quotes['ask_size_1'].notna())
        ]
        
        if quotes.empty:
            return None
        
        # Get last quote at decision time (within trailing window)
        last_quote = quotes.iloc[-1]
        
        # Calculate features
        midpoint = (last_quote['bid_price_1'] + last_quote['ask_price_1']) / 2.0
        queue_imbalance = (last_quote['bid_size_1'] - last_quote['ask_size_1']) / (last_quote['bid_size_1'] + last_quote['ask_size_1'])
        relative_spread = (last_quote['ask_price_1'] - last_quote['bid_price_1']) / midpoint
        
        # Process trades with signing
        trades = minute_df[minute_df['event_type'] == 'trade'].copy()
        trades = trades[
            (trades['trade_price'].notna()) & 
            (trades['trade_size'].notna())
        ]
        
        trade_signer = TradeSigner()
        signed_volume = 0.0
        trade_count = 0
        total_volume = 0.0
        
        # Process chronologically
        all_events = pd.concat([quotes, trades]).sort_values('timestamp_dt')
        
        for _, event in all_events.iterrows():
            if event['event_type'] == 'quote':
                quote_event = Event(
                    timestamp=event['timestamp_dt'],
                    event_type='quote',
                    bid_price=event['bid_price_1'],
                    ask_price=event['ask_price_1'],
                    bid_size=event['bid_size_1'],
                    ask_size=event['ask_size_1'],
                    trade_price=None,
                    trade_size=None,
                    aggressor_side=None
                )
                trade_signer.update_quote(quote_event)
            elif event['event_type'] == 'trade':
                trade_event = Event(
                    timestamp=event['timestamp_dt'],
                    event_type='trade',
                    bid_price=None,
                    ask_price=None,
                    bid_size=None,
                    ask_size=None,
                    trade_price=event['trade_price'],
                    trade_size=event['trade_size'],
                    aggressor_side=int(event['aggressor_side']) if pd.notna(event['aggressor_side']) else None
                )
                sign = trade_signer.sign_trade(trade_event)
                if sign is not None:
                    signed_volume += sign * event['trade_size']
                    trade_count += 1
                total_volume += event['trade_size']
        
        # Normalize signed volume
        if total_volume > 0:
            signed_volume_imbalance = signed_volume / total_volume
        else:
            signed_volume_imbalance = 0.0
        
        # Store canonical boundary timestamp (exactly T, not max event time)
        # Also store the actual quote timestamp used for midpoint calculation
        summary = {
            'timestamp': boundary_time.isoformat(),
            'date': boundary_time.astimezone(self.ny_tz).strftime("%Y-%m-%d"),
            'time': boundary_time.astimezone(self.ny_tz).strftime("%H:%M"),
            'queue_imbalance': queue_imbalance,
            'signed_volume_imbalance': signed_volume_imbalance,
            'relative_spread': relative_spread,
            'midpoint_return': 0.0,  # Will be calculated in sequence builder
            'midpoint': midpoint,
            'trade_count': trade_count,
            'has_valid_quote': True,
            'is_target_only': is_target_only,  # Mark target-only rows
            'quote_timestamp': last_quote['timestamp_dt'].isoformat(),  # Actual quote timestamp
            'bid': last_quote['bid_price_1'],
            'ask': last_quote['ask_price_1']
        }
        
        return summary