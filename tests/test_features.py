"""Tests for feature calculation correctness."""

import pytest
from datetime import datetime, timedelta, timezone

from model_a.data_loader import Event
from model_a.features import FeatureCalculator, MinuteSummary


class TestFeatureCalculation:
    """Tests for correct feature calculation."""
    
    def test_queue_imbalance_calculation(self):
        """Test queue imbalance formula: (bid_size - ask_size) / (bid_size + ask_size)."""
        calculator = FeatureCalculator()
        
        # Equal sizes -> imbalance = 0
        assert calculator._calculate_queue_imbalance(100, 100) == 0.0
        
        # More bid -> positive imbalance
        result = calculator._calculate_queue_imbalance(150, 50)
        expected = (150 - 50) / (150 + 50)
        assert abs(result - expected) < 1e-6
        
        # More ask -> negative imbalance
        result = calculator._calculate_queue_imbalance(50, 150)
        expected = (50 - 150) / (50 + 150)
        assert abs(result - expected) < 1e-6
    
    def test_relative_spread_calculation(self):
        """Test relative spread formula: (ask - bid) / midpoint."""
        calculator = FeatureCalculator()
        
        bid = 100.0
        ask = 101.0
        midpoint = 100.5
        
        result = calculator._calculate_relative_spread(bid, ask, midpoint)
        expected = (ask - bid) / midpoint
        assert abs(result - expected) < 1e-6
    
    def test_minute_summary_has_four_features(self):
        """Test that minute summary produces exactly 4 features."""
        decision_time = datetime(2026, 8, 25, 13, 55, 0, tzinfo=timezone.utc)
        
        quote = Event(
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
        
        calculator = FeatureCalculator()
        summary = calculator.calculate_minute_summary([quote], decision_time)
        
        assert summary is not None
        features = summary.to_feature_vector()
        assert len(features) == 4
    
    def test_minute_without_valid_quote_is_invalid(self):
        """Test that a minute with no valid quote is invalid."""
        decision_time = datetime(2026, 8, 25, 13, 55, 0, tzinfo=timezone.utc)
        
        # Trade without quote
        trade = Event(
            timestamp=decision_time - timedelta(seconds=30),
            event_type="trade",
            bid_price=None,
            ask_price=None,
            bid_size=None,
            ask_size=None,
            trade_price=100.5,
            trade_size=100,
            aggressor_side=1
        )
        
        calculator = FeatureCalculator()
        summary = calculator.calculate_minute_summary([trade], decision_time)
        
        assert summary is None
    
    def test_minute_with_no_trades_has_zero_volume_imbalance(self):
        """Test that a minute with valid quotes but no trades has zero volume imbalance."""
        decision_time = datetime(2026, 8, 25, 13, 55, 0, tzinfo=timezone.utc)
        
        quote = Event(
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
        
        calculator = FeatureCalculator()
        summary = calculator.calculate_minute_summary([quote], decision_time)
        
        assert summary is not None
        assert summary.signed_volume_imbalance == 0.0
        assert summary.trade_count == 0


class TestTradeSigning:
    """Tests for correct trade signing."""
    
    def test_provider_aggressor_side_is_used_when_valid(self):
        """Test that provider aggressor_side is used when +1 or -1."""
        from model_a.trade_signing import TradeSigner
        
        signer = TradeSigner()
        
        # Set up quote
        quote = Event(
            timestamp=datetime(2026, 8, 25, 13, 54, 0, tzinfo=timezone.utc),
            event_type="quote",
            bid_price=100.0,
            ask_price=101.0,
            bid_size=100,
            ask_size=100,
            trade_price=None,
            trade_size=None,
            aggressor_side=None
        )
        signer.update_quote(quote)
        
        # Trade with valid aggressor_side
        trade_buy = Event(
            timestamp=datetime(2026, 8, 25, 13, 54, 30, tzinfo=timezone.utc),
            event_type="trade",
            bid_price=None,
            ask_price=None,
            bid_size=None,
            ask_size=None,
            trade_price=100.5,
            trade_size=100,
            aggressor_side=1  # Valid buy
        )
        
        sign = signer.sign_trade(trade_buy)
        assert sign == 1
        
        # Trade with sell side
        trade_sell = Event(
            timestamp=datetime(2026, 8, 25, 13, 54, 31, tzinfo=timezone.utc),
            event_type="trade",
            bid_price=None,
            ask_price=None,
            bid_size=None,
            ask_size=None,
            trade_price=100.5,
            trade_size=100,
            aggressor_side=-1  # Valid sell
        )
        
        sign = signer.sign_trade(trade_sell)
        assert sign == -1
    
    def test_aggressor_side_0_is_treated_as_unresolved(self):
        """Test that aggressor_side = 0 is treated as unresolved."""
        from model_a.trade_signing import TradeSigner
        
        signer = TradeSigner()
        
        # Set up quote
        quote = Event(
            timestamp=datetime(2026, 8, 25, 13, 54, 0, tzinfo=timezone.utc),
            event_type="quote",
            bid_price=100.0,
            ask_price=101.0,
            bid_size=100,
            ask_size=100,
            trade_price=None,
            trade_size=None,
            aggressor_side=None
        )
        signer.update_quote(quote)
        
        # Trade with aggressor_side = 0
        trade = Event(
            timestamp=datetime(2026, 8, 25, 13, 54, 30, tzinfo=timezone.utc),
            event_type="trade",
            bid_price=None,
            ask_price=None,
            bid_size=None,
            ask_size=None,
            trade_price=100.5,
            trade_size=100,
            aggressor_side=0  # Unresolved
        )
        
        sign = signer.sign_trade(trade)
        # Should use midpoint comparison fallback
        # Trade at midpoint (100.5) equals quote midpoint (100.5)
        # Should use tick rule, but no prior trade, so unresolved
        assert sign is None
    
    def test_midpoint_comparison_fallback(self):
        """Test midpoint comparison when aggressor_side is invalid."""
        from model_a.trade_signing import TradeSigner
        
        signer = TradeSigner()
        
        # Set up quote
        quote = Event(
            timestamp=datetime(2026, 8, 25, 13, 54, 0, tzinfo=timezone.utc),
            event_type="quote",
            bid_price=100.0,
            ask_price=101.0,
            bid_size=100,
            ask_size=100,
            trade_price=None,
            trade_size=None,
            aggressor_side=None
        )
        signer.update_quote(quote)
        
        # Trade above midpoint
        trade_above = Event(
            timestamp=datetime(2026, 8, 25, 13, 54, 30, tzinfo=timezone.utc),
            event_type="trade",
            bid_price=None,
            ask_price=None,
            bid_size=None,
            ask_size=None,
            trade_price=101.0,  # Above midpoint (100.5)
            trade_size=100,
            aggressor_side=0
        )
        
        sign = signer.sign_trade(trade_above)
        assert sign == 1  # Buy
        
        # Trade below midpoint
        trade_below = Event(
            timestamp=datetime(2026, 8, 25, 13, 54, 31, tzinfo=timezone.utc),
            event_type="trade",
            bid_price=None,
            ask_price=None,
            bid_size=None,
            ask_size=None,
            trade_price=100.0,  # Below midpoint (100.5)
            trade_size=100,
            aggressor_side=0
        )
        
        sign = signer.sign_trade(trade_below)
        assert sign == -1  # Sell


class TestMidpointReturn:
    """Tests for midpoint return calculation."""
    
    def test_midpoint_return_calculation(self):
        """Test midpoint return formula: (current - previous) / previous."""
        calculator = FeatureCalculator()
        
        # First minute - no previous midpoint
        decision_time_1 = datetime(2026, 8, 25, 13, 35, 0, tzinfo=timezone.utc)
        quote_1 = Event(
            timestamp=decision_time_1 - timedelta(seconds=30),
            event_type="quote",
            bid_price=100.0,
            ask_price=101.0,
            bid_size=100,
            ask_size=100,
            trade_price=None,
            trade_size=None,
            aggressor_side=None
        )
        
        summary_1 = calculator.calculate_minute_summary([quote_1], decision_time_1)
        assert summary_1.midpoint_return == 0.0  # First observation
        
        # Second minute - has previous midpoint
        decision_time_2 = datetime(2026, 8, 25, 13, 36, 0, tzinfo=timezone.utc)
        quote_2 = Event(
            timestamp=decision_time_2 - timedelta(seconds=30),
            event_type="quote",
            bid_price=101.0,
            ask_price=102.0,
            bid_size=100,
            ask_size=100,
            trade_price=None,
            trade_size=None,
            aggressor_side=None
        )
        
        summary_2 = calculator.calculate_minute_summary([quote_2], decision_time_2)
        expected_return = (101.5 - 100.5) / 100.5
        assert abs(summary_2.midpoint_return - expected_return) < 1e-6