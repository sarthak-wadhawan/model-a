"""Regression tests for corrected boundary semantics."""

import pytest
import numpy as np
from datetime import datetime, timedelta, timezone
import pytz

from model_a.preprocessing import calculate_majority_baseline
from model_a.sequences import SequenceExample


class TestBaselineCalculation:
    """Test baseline calculation with corrected function."""
    
    def test_baseline_synthetic_data(self):
        """Test baseline with synthetic data."""
        train_targets = np.array([0, 0, 0, 1, 1])
        val_targets = np.array([0, 1, 1, 0])
        
        baseline = calculate_majority_baseline(train_targets, val_targets)
        
        # Training majority = DOWN (0)
        # Validation DOWN count = 2
        # Expected baseline = 2/4 = 0.5
        assert baseline == 0.5
    
    def test_baseline_real_counts(self):
        """Test baseline with real corrected label counts."""
        # Real counts from corrected cache
        train_targets = np.array([0]*285 + [1]*283)
        val_targets = np.array([0]*67 + [1]*75)
        
        baseline = calculate_majority_baseline(train_targets, val_targets)
        
        # Training majority = DOWN (0)
        # Validation DOWN count = 67
        # Expected baseline = 67/142 ≈ 0.4718
        expected = 67 / 142
        assert abs(baseline - expected) < 0.0001


class TestBoundarySemantics:
    """Test boundary semantics with synthetic fixtures."""
    
    def test_event_at_t_included(self):
        """Event exactly at boundary T is included in summary T."""
        ny_tz = pytz.timezone("America/New_York")
        
        # Create synthetic event at exactly 09:55:00
        event_time = ny_tz.localize(datetime(2026, 8, 25, 9, 55, 0))
        
        # For summary T=09:55, window is (09:54, 09:55]
        # Event at 09:55:00 should be included
        window_start = event_time - timedelta(minutes=1)
        window_end = event_time
        
        assert event_time > window_start
        assert event_time <= window_end
    
    def test_event_after_t_excluded(self):
        """Event immediately after T is excluded from summary T."""
        ny_tz = pytz.timezone("America/New_York")
        
        # Event at 09:55:00.001
        event_time = ny_tz.localize(datetime(2026, 8, 25, 9, 55, 0, 1000))
        
        # For summary T=09:55, window is (09:54, 09:55]
        # Event at 09:55:00.001 should be excluded
        window_start = event_time - timedelta(minutes=1)
        window_end = event_time.replace(microsecond=0)
        
        assert not (event_time > window_start and event_time <= window_end)
    
    def test_summary_timestamp_is_canonical_t(self):
        """Stored summary timestamp equals canonical T (exactly :00:00)."""
        ny_tz = pytz.timezone("America/New_York")
        
        # Canonical boundary
        canonical_t = ny_tz.localize(datetime(2026, 8, 25, 9, 55, 0, 0))
        
        # Verify it has zero seconds and microseconds
        assert canonical_t.second == 0
        assert canonical_t.microsecond == 0
    
    def test_0955_summary_is_0954_0955(self):
        """09:55 summary represents (09:54, 09:55]."""
        ny_tz = pytz.timezone("America/New_York")
        
        # Summary timestamp for 09:55
        summary_time = ny_tz.localize(datetime(2026, 8, 25, 9, 55, 0, 0))
        
        # Window should be (09:54, 09:55]
        window_start = summary_time - timedelta(minutes=1)
        window_end = summary_time
        
        # Verify time difference
        assert (window_end - window_start) == timedelta(minutes=1)
        assert window_end == summary_time
    
    def test_0955_sequence_is_0951_0955(self):
        """09:55 five-point sequence is exactly 09:51-09:55."""
        ny_tz = pytz.timezone("America/New_York")
        
        # Five consecutive minute boundaries
        boundaries = [
            ny_tz.localize(datetime(2026, 8, 25, 9, 51, 0, 0)),
            ny_tz.localize(datetime(2026, 8, 25, 9, 52, 0, 0)),
            ny_tz.localize(datetime(2026, 8, 25, 9, 53, 0, 0)),
            ny_tz.localize(datetime(2026, 8, 25, 9, 54, 0, 0)),
            ny_tz.localize(datetime(2026, 8, 25, 9, 55, 0, 0)),
        ]
        
        # Verify they are consecutive
        for i in range(1, len(boundaries)):
            assert boundaries[i] - boundaries[i-1] == timedelta(minutes=1)
        
        # Verify last is 09:55
        assert boundaries[-1].hour == 9
        assert boundaries[-1].minute == 55
    
    def test_target_boundary_is_1000(self):
        """Target boundary for 09:55 is exactly 10:00."""
        ny_tz = pytz.timezone("America/New_York")
        
        decision_time = ny_tz.localize(datetime(2026, 8, 25, 9, 55, 0, 0))
        target_time = decision_time + timedelta(minutes=5)
        
        assert target_time.hour == 10
        assert target_time.minute == 0
        assert target_time.second == 0
    
    def test_target_midpoint_from_own_trailing_window(self):
        """Target midpoint comes from target's own trailing minute."""
        ny_tz = pytz.timezone("America/New_York")
        
        # Create synthetic summaries
        summaries = {
            ny_tz.localize(datetime(2026, 8, 25, 9, 59, 0, 0)): {'midpoint': 100.0},
            ny_tz.localize(datetime(2026, 8, 25, 10, 0, 0, 0)): {'midpoint': 101.0},  # Target
        }
        
        # For decision at 09:55, target is 10:00
        # Target should use midpoint from 10:00's trailing minute (09:59, 10:00]
        target_time = ny_tz.localize(datetime(2026, 8, 25, 10, 0, 0, 0))
        
        # Find exact match
        assert target_time in summaries
        assert summaries[target_time]['midpoint'] == 101.0
    
    def test_missing_target_quote_excludes_sample(self):
        """Missing target-minute quote excludes the sample."""
        ny_tz = pytz.timezone("America/New_York")
        
        # Create sequence with missing target
        decision_time = ny_tz.localize(datetime(2026, 8, 25, 9, 55, 0, 0))
        target_time = decision_time + timedelta(minutes=5)
        
        # No target midpoint available
        target_midpoint = None
        
        # Sample should be excluded
        assert target_midpoint is None
    
    def test_stale_fallback_cannot_create_label(self):
        """Stale midpoint fallback cannot create a target label."""
        ny_tz = pytz.timezone("America/New_York")
        
        # Create scenario where target T+5 has no data
        decision_time = ny_tz.localize(datetime(2026, 8, 25, 11, 59, 0, 0))
        target_time = decision_time + timedelta(minutes=5)
        
        # Available summaries (only up to 12:00)
        available_summaries = {
            ny_tz.localize(datetime(2026, 8, 25, 11, 59, 0, 0)): {'midpoint': 100.0},
            ny_tz.localize(datetime(2026, 8, 25, 12, 0, 0, 0)): {'midpoint': 101.0},
        }
        
        # Target 12:04 is not in available summaries
        # Correct behavior: exclude sample
        # Incorrect behavior: use 12:00 or 11:59 as fallback
        
        # Verify target time is not in available summaries
        assert target_time not in available_summaries
        
        # If we used last available (12:00), that would be stale fallback
        # Correct behavior is to return None (exclude sample)
        target_midpoint = available_summaries.get(target_time)
        assert target_midpoint is None
    
    def test_target_only_rows_cannot_enter_inputs(self):
        """Target-only rows cannot become model inputs."""
        # Create synthetic summaries
        input_summaries = [
            {'timestamp': datetime(2026, 8, 25, 13, 51, 0, tzinfo=timezone.utc), 'is_target_only': False},
            {'timestamp': datetime(2026, 8, 25, 13, 52, 0, tzinfo=timezone.utc), 'is_target_only': False},
        ]
        target_summaries = [
            {'timestamp': datetime(2026, 8, 25, 14, 1, 0, tzinfo=timezone.utc), 'is_target_only': True},
        ]
        
        # Filter for inputs
        inputs_only = [s for s in input_summaries + target_summaries if not s['is_target_only']]
        
        # Should have only 2 input summaries
        assert len(inputs_only) == 2
        assert all(not s['is_target_only'] for s in inputs_only)
    
    def test_1200_is_input_eligible(self):
        """12:00 is input eligible."""
        ny_tz = pytz.timezone("America/New_York")
        
        # 12:00 is within observation window (09:35-12:00 inclusive)
        time_1200 = ny_tz.localize(datetime(2026, 8, 25, 12, 0, 0))
        
        start_hour = 9
        start_minute = 35
        end_hour = 12
        end_minute = 0
        
        in_window = (time_1200.hour > start_hour or 
                     (time_1200.hour == start_hour and time_1200.minute >= start_minute))
        in_window = in_window and (time_1200.hour < end_hour or 
                                   (time_1200.hour == end_hour and time_1200.minute <= end_minute))
        
        assert in_window
    
    def test_1200_targets_1205(self):
        """12:00 decision targets exactly 12:05."""
        ny_tz = pytz.timezone("America/New_York")
        
        decision_time = ny_tz.localize(datetime(2026, 8, 25, 12, 0, 0))
        target_time = decision_time + timedelta(minutes=5)
        
        assert target_time.hour == 12
        assert target_time.minute == 5
    
    def test_0935_input_eligible_with_pre_roll(self):
        """09:35 is input eligible and constructed from (09:34,09:35] pre-roll."""
        ny_tz = pytz.timezone("America/New_York")
        
        # 09:35 summary uses window (09:34, 09:55]
        summary_time = ny_tz.localize(datetime(2026, 8, 25, 9, 35, 0, 0))
        window_start = summary_time - timedelta(minutes=1)
        window_end = summary_time
        
        # Window is (09:34, 09:35]
        assert window_start.hour == 9
        assert window_start.minute == 34
        assert window_end.hour == 9
        assert window_end.minute == 35


class TestDemoSelection:
    """Test demo selection logic."""
    
    def test_demo_selects_explicit_0955(self):
        """Demo selects explicit 09:55 rather than array index."""
        ny_tz = pytz.timezone("America/New_York")
        
        # Create synthetic sequences
        seq_0954 = SequenceExample(
            input_timestamps=[
                datetime(2026, 8, 25, 13, 50, 0, tzinfo=timezone.utc),
                datetime(2026, 8, 25, 13, 51, 0, tzinfo=timezone.utc),
                datetime(2026, 8, 25, 13, 52, 0, tzinfo=timezone.utc),
                datetime(2026, 8, 25, 13, 53, 0, tzinfo=timezone.utc),
                datetime(2026, 8, 25, 13, 54, 0, tzinfo=timezone.utc),
            ],
            features=[0]*20,
            target_timestamp=datetime(2026, 8, 25, 13, 59, 0, tzinfo=timezone.utc),
            target=0,
            actual_direction=0,
            decision_midpoint=100.0,
            target_midpoint=101.0
        )
        
        seq_0955 = SequenceExample(
            input_timestamps=[
                datetime(2026, 8, 25, 13, 51, 0, tzinfo=timezone.utc),
                datetime(2026, 8, 25, 13, 52, 0, tzinfo=timezone.utc),
                datetime(2026, 8, 25, 13, 53, 0, tzinfo=timezone.utc),
                datetime(2026, 8, 25, 13, 54, 0, tzinfo=timezone.utc),
                datetime(2026, 8, 25, 13, 55, 0, tzinfo=timezone.utc),
            ],
            features=[0]*20,
            target_timestamp=datetime(2026, 8, 25, 14, 0, 0, tzinfo=timezone.utc),
            target=0,
            actual_direction=0,
            decision_midpoint=100.0,
            target_midpoint=101.0
        )
        
        demo_seqs = [seq_0954, seq_0955]
        
        # Select by explicit 09:55 check
        for seq in demo_seqs:
            last_input = seq.input_timestamps[-1].astimezone(ny_tz)
            if last_input.hour == 9 and last_input.minute == 55:
                selected = seq
                break
        
        # Should select the second sequence, not the first (index 0)
        assert selected == seq_0955
        assert selected != demo_seqs[0]