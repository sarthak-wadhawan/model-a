"""Five-point sequence construction and target generation."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import pytz

from .data_loader import Event, EventStream
from .features import MinuteSummary, FeatureCalculator, create_minute_boundaries


@dataclass
class SequenceExample:
    """A single training example with inputs and target."""
    input_timestamps: List[datetime]  # 5 consecutive decision times
    features: List[float]  # 20 flattened features (5 x 4)
    target_timestamp: datetime  # T+5 minutes
    target: int  # 1 for UP, 0 for DOWN
    actual_direction: Optional[int]  # Actual outcome if available
    decision_midpoint: Optional[float]  # Midpoint at decision time
    target_midpoint: Optional[float]  # Midpoint at target time
    decision_quote_timestamp: Optional[datetime] = None  # Actual quote timestamp for decision
    decision_bid: Optional[float] = None  # Bid at decision
    decision_ask: Optional[float] = None  # Ask at decision
    target_quote_timestamp: Optional[datetime] = None  # Actual quote timestamp for target
    target_bid: Optional[float] = None  # Bid at target
    target_ask: Optional[float] = None  # Ask at target
    
    def to_feature_vector(self) -> List[float]:
        """Return the 20-element feature vector."""
        return self.features


class SequenceBuilder:
    """Builds five-point sequences with five-minute targets."""
    
    def __init__(self, events_csv_path: str):
        self.events_csv_path = events_csv_path
        self.ny_tz = pytz.timezone("America/New_York")
        self.calculator = FeatureCalculator()
    
    def build_sequences(self, 
                       start_date: str,
                       end_date: str,
                       start_hour: int = 9,
                       start_minute: int = 35,
                       end_hour: int = 12,
                       end_minute: int = 0) -> List[SequenceExample]:
        """Build all valid five-point sequences for date range.
        
        Args:
            start_date: YYYY-MM-DD start date
            end_date: YYYY-MM-DD end date
            start_hour: Start hour in ET
            start_minute: Start minute in ET
            end_hour: End hour in ET
            end_minute: End minute in ET
            
        Returns:
            List of valid sequence examples
        """
        print(f"Loading events from {self.events_csv_path}...")
        print(f"Date range: {start_date} to {end_date}")
        
        # Create minute boundaries
        boundaries = create_minute_boundaries(
            start_date, end_date, start_hour, start_minute, end_hour, end_minute)
        
        print(f"Created {len(boundaries)} minute boundaries")
        
        # Load events for the entire range (with buffer for first minute)
        event_stream = EventStream(self.events_csv_path)
        
        # Load events with extended range for causal calculations
        from .data_loader import filter_by_session
        
        # Get buffer start time (earlier date for causal calculations)
        buffer_minutes = 10
        ny_tz = pytz.timezone("America/New_York")
        start_dt = ny_tz.localize(datetime.strptime(start_date, "%Y-%m-%d").replace(
            hour=start_hour, minute=start_minute, second=0, microsecond=0))
        buffer_dt = start_dt - timedelta(minutes=buffer_minutes)
        buffer_date = buffer_dt.strftime("%Y-%m-%d")
        
        # Load events from buffer date to end date only
        print(f"Loading events from {buffer_date} to {end_date}...")
        all_events = []
        event_count = 0
        for event in event_stream.events(start_date=buffer_date, end_date=end_date):
            event_count += 1
            if event_count % 500000 == 0:
                print(f"Loaded {event_count} events...")
            all_events.append(event)
        
        print(f"Total events loaded: {len(all_events)}")
        
        # Filter to observation window
        filtered_events = list(filter_by_session(
            iter(all_events), start_date, end_date, start_hour, start_minute, 
            end_hour, end_minute))
        
        print(f"Events in observation window: {len(filtered_events)}")
        
        # Build minute summaries
        print("Building minute summaries...")
        minute_summaries = self._build_minute_summaries(filtered_events, boundaries)
        
        valid_summaries = [s for s in minute_summaries if s is not None]
        print(f"Valid minute summaries: {len(valid_summaries)}")
        
        # Build five-point sequences
        print("Building five-point sequences...")
        sequences = self._build_five_point_sequences(minute_summaries)
        
        print(f"Five-point sequences: {len(sequences)}")
        
        # Add targets
        print("Adding targets...")
        sequences_with_targets = self._add_targets(sequences, all_events)
        
        print(f"Final sequences with targets: {len(sequences_with_targets)}")
        
        return sequences_with_targets
    
    def _get_buffer_start(self, date: str, hour: int, minute: int) -> str:
        """Get buffer start time for loading preceding data."""
        # Load 5 minutes before first observation for causal calculations
        buffer_minutes = 5
        ny_tz = pytz.timezone("America/New_York")
        start_dt = ny_tz.localize(
            datetime.strptime(date, "%Y-%m-%d").replace(
                hour=hour, minute=minute, second=0, microsecond=0))
        buffer_dt = start_dt - timedelta(minutes=buffer_minutes)
        return buffer_dt.strftime("%Y-%m-%d %H:%M:%S")
    
    def _build_minute_summaries(self, 
                                events: List[Event],
                                boundaries: List[datetime]) -> List[Optional[MinuteSummary]]:
        """Build minute summaries for each boundary time.
        
        Args:
            events: List of events in the observation window
            boundaries: List of decision times
            
        Returns:
            List of minute summaries (None for invalid minutes)
        """
        summaries = []
        
        for i, boundary in enumerate(boundaries):
            # Get events in the trailing window (boundary - 1min, boundary]
            window_start = boundary - timedelta(minutes=1)
            window_events = [
                e for e in events 
                if window_start < e.timestamp <= boundary
            ]
            
            # Calculate summary
            summary = self.calculator.calculate_minute_summary(window_events, boundary)
            summaries.append(summary)
        
        return summaries
    
    def _build_five_point_sequences(self, 
                                   summaries: List[Optional[MinuteSummary]]) -> List[Tuple[List[MinuteSummary], datetime]]:
        """Build five-point consecutive sequences.
        
        Args:
            summaries: List of minute summaries
            
        Returns:
            List of (5 summaries, decision_time) tuples
        """
        sequences = []
        
        for i in range(len(summaries) - 4):
            # Check if we have 5 consecutive valid summaries
            window = summaries[i:i+5]
            if all(s is not None for s in window):
                decision_time = window[-1].timestamp
                sequences.append((window, decision_time))
        
        return sequences
    
    def _add_targets(self, 
                    sequences: List[Tuple[List[MinuteSummary], datetime]],
                    all_events: List[Event]) -> List[SequenceExample]:
        """Add five-minute targets to sequences.
        
        Args:
            sequences: List of (5 summaries, decision_time) tuples
            all_events: All events for finding target midpoints
            
        Returns:
            List of complete sequence examples with targets
        """
        examples = []
        
        for summaries, decision_time in sequences:
            target_time = decision_time + timedelta(minutes=5)
            
            # Find midpoint at decision time (from last summary)
            decision_midpoint = self._get_midpoint_from_summary(summaries[-1])
            
            # Find midpoint at target time
            target_midpoint = self._find_midpoint_at_time(all_events, target_time)
            
            if target_midpoint is None:
                # Cannot determine target, skip this example
                continue
            
            # Determine target (UP if target > decision, else DOWN)
            if target_midpoint > decision_midpoint:
                target = 1  # UP
                actual_direction = 1
            else:
                target = 0  # DOWN (includes flat)
                actual_direction = 0
            
            # Flatten features: 5 minutes x 4 features = 20 inputs
            features = []
            input_timestamps = []
            for summary in summaries:
                features.extend(summary.to_feature_vector())
                input_timestamps.append(summary.timestamp)
            
            examples.append(SequenceExample(
                input_timestamps=input_timestamps,
                features=features,
                target_timestamp=target_time,
                target=target,
                actual_direction=actual_direction,
                decision_midpoint=decision_midpoint,
                target_midpoint=target_midpoint
            ))
        
        return examples
    
    def _get_midpoint_from_summary(self, summary: MinuteSummary) -> float:
        """Extract midpoint from a minute summary."""
        return summary.midpoint
    
    def _find_midpoint_at_time(self, 
                               events: List[Event],
                               target_time: datetime) -> Optional[float]:
        """Find the most recent valid quote at or before target_time.
        
        Uses only events with timestamp <= target_time (causal constraint).
        """
        # Filter events to those <= target_time
        causal_events = [e for e in events if e.timestamp <= target_time]
        
        # Find the last valid quote
        for event in reversed(causal_events):
            if event.is_quote() and event.has_valid_quote():
                return (event.bid_price + event.ask_price) / 2.0
        
        return None


def split_by_date(sequences: List[SequenceExample],
                  train_dates: List[str],
                  val_date: str,
                  demo_date: str) -> Tuple[List[SequenceExample], List[SequenceExample], List[SequenceExample]]:
    """Split sequences chronologically by date.
    
    Args:
        sequences: List of all sequences
        train_dates: List of training dates (YYYY-MM-DD)
        val_date: Validation date (YYYY-MM-DD)
        demo_date: Demonstration date (YYYY-MM-DD)
        
    Returns:
        (train_sequences, val_sequences, demo_sequences)
    """
    train = []
    val = []
    demo = []
    
    for seq in sequences:
        decision_date = seq.input_timestamps[-1].strftime("%Y-%m-%d")
        
        if decision_date in train_dates:
            train.append(seq)
        elif decision_date == val_date:
            val.append(seq)
        elif decision_date == demo_date:
            demo.append(seq)
    
    return train, val, demo
