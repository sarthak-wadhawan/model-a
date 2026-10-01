"""Load minute summaries from cache and build five-point sequences."""

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Optional
import pytz

from .sequences import SequenceExample


class CacheLoader:
    """Load minute summaries from cache and build sequences."""
    
    def __init__(self, cache_path: str):
        self.cache_path = Path(cache_path)
        self.ny_tz = pytz.timezone("America/New_York")
    
    def load_summaries(self) -> Dict[str, List[Dict]]:
        """Load minute summaries from cache, grouped by date."""
        summaries_by_date: Dict[str, List[Dict]] = {}
        
        with open(self.cache_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                date = row['date']
                if date not in summaries_by_date:
                    summaries_by_date[date] = []
                
                summary = {
                    'timestamp': datetime.fromisoformat(row['timestamp']),
                    'date': row['date'],
                    'time': row['time'],
                    'queue_imbalance': float(row['queue_imbalance']),
                    'signed_volume_imbalance': float(row['signed_volume_imbalance']),
                    'relative_spread': float(row['relative_spread']),
                    'midpoint_return': float(row['midpoint_return']),
                    'midpoint': float(row['midpoint']),
                    'trade_count': int(row['trade_count']),
                    'has_valid_quote': row['has_valid_quote'] == 'True',
                    'is_target_only': row.get('is_target_only', 'False') == 'True',
                    'quote_timestamp': datetime.fromisoformat(row['quote_timestamp']),
                    'bid': float(row['bid']),
                    'ask': float(row['ask'])
                }
                summaries_by_date[date].append(summary)
        
        # Sort summaries by timestamp within each date
        for date in summaries_by_date:
            summaries_by_date[date].sort(key=lambda x: x['timestamp'])
        
        return summaries_by_date
    
    def build_sequences(self, summaries_by_date: Dict[str, List[Dict]]) -> List[SequenceExample]:
        """Build five-point sequences from minute summaries."""
        sequences = []
        
        for date in sorted(summaries_by_date.keys()):
            summaries = summaries_by_date[date]
            
            # Filter out target-only rows for model inputs
            input_summaries = [s for s in summaries if not s['is_target_only']]
            
            # Recalculate midpoint returns for this date
            previous_midpoint = None
            for i, summary in enumerate(input_summaries):
                if previous_midpoint is not None:
                    summary['midpoint_return'] = (summary['midpoint'] - previous_midpoint) / previous_midpoint
                else:
                    summary['midpoint_return'] = 0.0
                previous_midpoint = summary['midpoint']
            
            # Need at least 5 consecutive minutes
            for i in range(len(input_summaries) - 4):
                # Get 5 consecutive minute summaries
                five_summaries = input_summaries[i:i+5]
                
                # Build feature vector (5 minutes × 4 features = 20)
                features = []
                input_timestamps = []
                
                for summary in five_summaries:
                    features.extend([
                        summary['queue_imbalance'],
                        summary['signed_volume_imbalance'],
                        summary['relative_spread'],
                        summary['midpoint_return']
                    ])
                    input_timestamps.append(summary['timestamp'])
                
                # Target: 5 minutes forward
                decision_time = five_summaries[-1]['timestamp']
                target_time = decision_time + timedelta(minutes=5)
                
                # Find the midpoint at target time using target's trailing minute
                target_summary = self._find_target_summary(summaries_by_date, target_time)
                target_midpoint = target_summary['midpoint'] if target_summary else None
                decision_midpoint = five_summaries[-1]['midpoint']
                
                if target_midpoint is None:
                    continue  # Skip if no valid target
                
                # Determine target direction
                if target_midpoint > decision_midpoint:
                    target = 1  # UP
                    actual_direction = 1
                else:
                    target = 0  # DOWN
                    actual_direction = 0
                
                sequence = SequenceExample(
                    input_timestamps=input_timestamps,
                    features=features,
                    target_timestamp=target_time,
                    target=target,
                    actual_direction=actual_direction,
                    decision_midpoint=decision_midpoint,
                    target_midpoint=target_midpoint,
                    decision_quote_timestamp=five_summaries[-1]['quote_timestamp'],
                    decision_bid=five_summaries[-1]['bid'],
                    decision_ask=five_summaries[-1]['ask'],
                    target_quote_timestamp=target_summary['quote_timestamp'] if target_summary else None,
                    target_bid=target_summary['bid'] if target_summary else None,
                    target_ask=target_summary['ask'] if target_summary else None
                )
                
                sequences.append(sequence)
        
        return sequences
    
    def _find_midpoint_at_time(self, summaries_by_date: Dict[str, List[Dict]], target_time: datetime) -> float:
        """Find the midpoint at or before target_time (legacy method)."""
        target_date = target_time.astimezone(self.ny_tz).strftime("%Y-%m-%d")
        
        if target_date not in summaries_by_date:
            return None
        
        summaries = summaries_by_date[target_date]
        
        # Find the last summary with timestamp <= target_time
        for summary in reversed(summaries):
            if summary['timestamp'] <= target_time:
                return summary['midpoint']
        
        return None
    
    def _find_target_summary(self, summaries_by_date: Dict[str, List[Dict]], target_time: datetime) -> Optional[Dict]:
        """Find the summary at the target's trailing minute (T+4, T+5]."""
        target_date = target_time.astimezone(self.ny_tz).strftime("%Y-%m-%d")
        
        if target_date not in summaries_by_date:
            return None
        
        summaries = summaries_by_date[target_date]
        
        # Find the summary with timestamp exactly equal to target_time
        # This represents the trailing minute (T+4, T+5]
        for summary in summaries:
            if summary['timestamp'] == target_time:
                return summary
        
        return None
    
    def split_by_date(self, sequences: List[SequenceExample], 
                     train_dates: List[str], val_date: str, demo_date: str):
        """Split sequences by date."""
        train_seqs = []
        val_seqs = []
        demo_seqs = []
        
        for seq in sequences:
            # Use the last input timestamp to determine date
            date = seq.input_timestamps[-1].astimezone(self.ny_tz).strftime("%Y-%m-%d")
            
            if date in train_dates:
                train_seqs.append(seq)
            elif date == val_date:
                val_seqs.append(seq)
            elif date == demo_date:
                demo_seqs.append(seq)
        
        return train_seqs, val_seqs, demo_seqs