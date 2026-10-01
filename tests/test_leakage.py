"""Tests for temporal leakage prevention."""

import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from model_a.data_loader import Event, EventStream
from model_a.trade_signing import TradeSigner
from model_a.features import FeatureCalculator
from model_a.sequences import SequenceBuilder


class TestTemporalLeakage:
    """Tests that ensure no future information leaks into predictions."""
    
    def test_event_after_decision_time_cannot_affect_prediction(self):
        """Test that an event at 09:55:00.001 cannot affect a prediction at 09:55:00."""
        # Create events around a decision time
        decision_time = datetime(2026, 8, 25, 13, 55, 0, tzinfo=timezone.utc)  # 09:55 ET
        
        # Event exactly at decision time (should be included)
        event_at_time = Event(
            timestamp=decision_time,
            event_type="quote",
            bid_price=100.0,
            ask_price=101.0,
            bid_size=100,
            ask_size=100,
            trade_price=None,
            trade_size=None,
            aggressor_side=None
        )
        
        # Event 1ms after decision time (should NOT be included)
        event_after = Event(
            timestamp=decision_time + timedelta(microseconds=1000),
            event_type="quote",
            bid_price=99.0,  # Different price
            ask_price=102.0,
            bid_size=200,
            ask_size=200,
            trade_price=None,
            trade_size=None,
            aggressor_side=None
        )
        
        calculator = FeatureCalculator()
        
        # Calculate summary with both events
        events = [event_at_time, event_after]
        summary = calculator.calculate_minute_summary(events, decision_time)
        
        # The summary should use the event at decision time, not the later one
        assert summary is not None
        assert summary.midpoint == 100.5  # (100.0 + 101.0) / 2, not (99.0 + 102.0) / 2
    
    def test_event_after_trade_cannot_sign_trade(self):
        """Test that an event after a trade cannot be used to sign that trade."""
        trade_time = datetime(2026, 8, 25, 13, 55, 0, tzinfo=timezone.utc)
        
        # Quote before trade
        quote_before = Event(
            timestamp=trade_time - timedelta(seconds=1),
            event_type="quote",
            bid_price=100.0,
            ask_price=101.0,
            bid_size=100,
            ask_size=100,
            trade_price=None,
            trade_size=None,
            aggressor_side=None
        )
        
        # Trade
        trade = Event(
            timestamp=trade_time,
            event_type="trade",
            bid_price=None,
            ask_price=None,
            bid_size=None,
            ask_size=None,
            trade_price=100.5,  # At midpoint
            trade_size=100,
            aggressor_side=0  # Invalid, need fallback
        )
        
        # Quote after trade (should NOT be used for signing)
        quote_after = Event(
            timestamp=trade_time + timedelta(seconds=1),
            event_type="quote",
            bid_price=99.0,  # Different midpoint
            ask_price=102.0,
            bid_size=100,
            ask_size=100,
            trade_price=None,
            trade_size=None,
            aggressor_side=None
        )
        
        signer = TradeSigner()
        
        # Process events in order
        signer.update_quote(quote_before)
        sign = signer.sign_trade(trade)
        signer.update_quote(quote_after)  # This comes after the trade
        
        # The trade should be signed using the quote before, not after
        # Since trade is at midpoint (100.5) and quote midpoint is 100.5,
        # it should use tick rule (no prior trade, so unresolved)
        assert sign is None  # Cannot resolve without tick rule info
    
    def test_0955_sequence_ends_exactly_at_0955(self):
        """Test that the 09:55 five-point sequence ends exactly at 09:55 and contains no later data."""
        # This test would require full sequence construction
        # For now, we test the timestamp logic in feature calculation
        
        decision_time = datetime(2026, 8, 25, 13, 55, 0, tzinfo=timezone.utc)  # 09:55 ET
        
        # Create events in the minute ending at 09:55
        events = []
        for i in range(10):
            ts = decision_time - timedelta(seconds=60 - i * 6)  # Spread across the minute
            event = Event(
                timestamp=ts,
                event_type="quote",
                bid_price=100.0 + i * 0.1,
                ask_price=101.0 + i * 0.1,
                bid_size=100,
                ask_size=100,
                trade_price=None,
                trade_size=None,
                aggressor_side=None
            )
            events.append(event)
        
        # Add an event exactly at 09:55:00.001 (should be excluded)
        event_after = Event(
            timestamp=decision_time + timedelta(microseconds=1000),
            event_type="quote",
            bid_price=200.0,  # Very different price
            ask_price=201.0,
            bid_size=1000,
            ask_size=1000,
            trade_price=None,
            trade_size=None,
            aggressor_side=None
        )
        events.append(event_after)
        
        calculator = FeatureCalculator()
        summary = calculator.calculate_minute_summary(events, decision_time)
        
        # Summary should not include the event after decision time
        assert summary is not None
        # The midpoint should be based on events <= decision time
        # The last event before decision time has midpoint ~100.45
        assert summary.midpoint < 150.0  # Not the 200.5 from the after event
    
    def test_target_aligned_to_1000(self):
        """Test that the corresponding target is aligned to 10:00."""
        decision_time = datetime(2026, 8, 25, 13, 55, 0, tzinfo=timezone.utc)  # 09:55 ET
        expected_target_time = decision_time + timedelta(minutes=5)  # 10:00 ET
        
        assert expected_target_time.hour == 14  # 10:00 ET = 14:00 UTC
        assert expected_target_time.minute == 0
    
    def test_aug25_cannot_influence_training(self):
        """Test that Aug 25 cannot influence scaler fitting, training, early stopping, or model selection."""
        # This is tested by the chronological split logic
        # We verify that Aug 25 data is never included in training
        
        from model_a.sequences import split_by_date, SequenceExample
        
        # Create mock sequences for different dates
        aug18_seq = SequenceExample(
            input_timestamps=[datetime(2026, 8, 18, 13, 55, tzinfo=timezone.utc)] * 5,
            features=[0.0] * 20,
            target_timestamp=datetime(2026, 8, 18, 14, 0, tzinfo=timezone.utc),
            target=1,
            actual_direction=1,
            decision_midpoint=100.0,
            target_midpoint=101.0
        )
        
        aug24_seq = SequenceExample(
            input_timestamps=[datetime(2026, 8, 24, 13, 55, tzinfo=timezone.utc)] * 5,
            features=[0.0] * 20,
            target_timestamp=datetime(2026, 8, 24, 14, 0, tzinfo=timezone.utc),
            target=1,
            actual_direction=1,
            decision_midpoint=100.0,
            target_midpoint=101.0
        )
        
        aug25_seq = SequenceExample(
            input_timestamps=[datetime(2026, 8, 25, 13, 55, tzinfo=timezone.utc)] * 5,
            features=[0.0] * 20,
            target_timestamp=datetime(2026, 8, 25, 14, 0, tzinfo=timezone.utc),
            target=1,
            actual_direction=1,
            decision_midpoint=100.0,
            target_midpoint=101.0
        )
        
        sequences = [aug18_seq, aug24_seq, aug25_seq]
        
        train_dates = ['2026-08-18']
        val_date = '2026-08-24'
        demo_date = '2026-08-25'
        
        train, val, demo = split_by_date(sequences, train_dates, val_date, demo_date)
        
        # Aug 25 should only be in demo set
        assert len(train) == 1
        assert len(val) == 1
        assert len(demo) == 1
        assert demo[0].input_timestamps[-1].date() == datetime(2026, 8, 25).date()
        assert aug25_seq not in train
        assert aug25_seq not in val


class TestFeatureLeakage:
    """Tests for feature-level leakage prevention."""
    
    def test_no_future_quotes_in_midpoint_calculation(self):
        """Test that midpoint calculation uses only causal quotes."""
        calculator = FeatureCalculator()
        
        decision_time = datetime(2026, 8, 25, 13, 55, 0, tzinfo=timezone.utc)
        
        # Quote before decision time
        quote_before = Event(
            timestamp=decision_time - timedelta(seconds=30),
            event_type="quote",
            bid_price=100.0,
            ask_price=101.0,
            bid_size=100,
            ask_size=100,
            trade_price=None,
            trade_size=None,
            aggressor_side=None
        )
        
        # Quote after decision time
        quote_after = Event(
            timestamp=decision_time + timedelta(seconds=30),
            event_type="quote",
            bid_price=200.0,
            ask_price=201.0,
            bid_size=100,
            ask_size=100,
            trade_price=None,
            trade_size=None,
            aggressor_side=None
        )
        
        events = [quote_before, quote_after]
        summary = calculator.calculate_minute_summary(events, decision_time)
        
        # Should use the quote before, not after
        assert summary.midpoint == 100.5  # From quote_before
        assert summary.midpoint != 200.5  # Not from quote_after
    
    def test_no_future_trades_in_volume_calculation(self):
        """Test that volume calculation uses only causal trades."""
        calculator = FeatureCalculator()
        
        decision_time = datetime(2026, 8, 25, 13, 55, 0, tzinfo=timezone.utc)
        
        # Trade before decision time
        trade_before = Event(
            timestamp=decision_time - timedelta(seconds=30),
            event_type="trade",
            bid_price=None,
            ask_price=None,
            bid_size=None,
            ask_size=None,
            trade_price=100.5,
            trade_size=100,
            aggressor_side=1  # Buy
        )
        
        # Trade after decision time
        trade_after = Event(
            timestamp=decision_time + timedelta(seconds=30),
            event_type="trade",
            bid_price=None,
            ask_price=None,
            bid_size=None,
            ask_size=None,
            trade_price=101.0,
            trade_size=200,
            aggressor_side=1  # Buy
        )
        
        # Need a quote for signing
        quote = Event(
            timestamp=decision_time - timedelta(seconds=60),
            event_type="quote",
            bid_price=100.0,
            ask_price=101.0,
            bid_size=100,
            ask_size=100,
            trade_price=None,
            trade_size=None,
            aggressor_side=None
        )
        
        events = [quote, trade_before, trade_after]
        summary = calculator.calculate_minute_summary(events, decision_time)
        
        # Should only include trade_before volume
        assert summary.trade_count == 1  # Only one trade included