"""Causal trade signing with per-trade quote fallback."""

from datetime import datetime
from typing import Optional, Tuple

from .data_loader import Event


class TradeSigner:
    """Signs trades using causal information only."""
    
    def __init__(self):
        self.last_quote_midpoint: Optional[float] = None
        self.last_quote_time: Optional[datetime] = None
        self.last_trade_price: Optional[float] = None
        self.last_trade_sign: Optional[int] = None  # +1 for buy, -1 for sell
    
    def update_quote(self, event: Event) -> None:
        """Update quote state from a valid quote event."""
        if event.has_valid_quote():
            self.last_quote_midpoint = (event.bid_price + event.ask_price) / 2.0
            self.last_quote_time = event.timestamp
    
    def sign_trade(self, event: Event) -> Optional[int]:
        """Sign a trade using causal information only.
        
        Args:
            event: Trade event to sign
            
        Returns:
            +1 for buy, -1 for sell, None if unresolvable
            
        Signing order:
        1. Provider aggressor_side if valid (+1 or -1)
        2. Compare trade price against latest causal quote midpoint
        3. Above midpoint = buy, below = sell
        4. Exact midpoint = causal tick rule
        5. Unresolved trades remain None
        """
        if not event.is_trade() or not event.has_valid_trade():
            return None
        
        # 1. Try provider aggressor_side if valid
        if event.aggressor_side in (1, -1):
            sign = event.aggressor_side
            self._update_trade_state(event.trade_price, sign)
            return sign
        
        # Provider side is 0 or invalid, use midpoint comparison
        if self.last_quote_midpoint is None:
            # No causal quote available, cannot sign
            return None
        
        # 2. Compare trade price against latest causal quote midpoint
        if event.trade_price > self.last_quote_midpoint:
            sign = 1  # buy
        elif event.trade_price < self.last_quote_midpoint:
            sign = -1  # sell
        else:
            # 3. Exact midpoint - use causal tick rule
            sign = self._apply_tick_rule(event.trade_price)
        
        self._update_trade_state(event.trade_price, sign)
        return sign
    
    def _apply_tick_rule(self, current_price: float) -> Optional[int]:
        """Apply causal tick rule for trades at midpoint.
        
        Tick rule: If current price > last trade price, buy.
                   If current price < last trade price, sell.
                   If equal, use last trade's sign.
        """
        if self.last_trade_price is None:
            # First trade at midpoint, cannot resolve
            return None
        
        if current_price > self.last_trade_price:
            return 1  # buy
        elif current_price < self.last_trade_price:
            return -1  # sell
        else:
            # Same price as last trade, use last sign
            return self.last_trade_sign
    
    def _update_trade_state(self, price: float, sign: Optional[int]) -> None:
        """Update trade state for future tick rule usage."""
        self.last_trade_price = price
        self.last_trade_sign = sign
    
    def get_current_midpoint(self) -> Optional[float]:
        """Get the most recent quote midpoint."""
        return self.last_quote_midpoint