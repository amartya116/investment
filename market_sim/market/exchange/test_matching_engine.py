import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pytest
from decimal import Decimal
from uuid import uuid4
from core.models.base import Order, OrderSide, OrderType, OrderStatus
from matching_engine import MatchingEngine
from core.utils.time_utils import utc_now

def test_buy_sell_matching():
    symbol = "AAPL"
    peers = set()  # no peers for this test
    engine = MatchingEngine(symbol, peers)

    # Create and add a SELL limit order (maker)
    sell_order = Order.create_limit_order(
        symbol=symbol,
        side=OrderSide.SELL,
        quantity=Decimal("10"),
        price=Decimal("150"),
        agent_id=str(uuid4())
    )
    trades = engine.process_order(sell_order)
    assert len(trades) == 0  # No immediate trade, order goes into book

    # Create a BUY market order (taker) that should fully match with the sell order
    buy_order = Order.create_market_order(
        symbol=symbol,
        side=OrderSide.BUY,
        quantity=Decimal("10"),
        agent_id=str(uuid4())
    )
    trades = engine.process_order(buy_order)

    # Verify one trade was executed
    assert len(trades) == 1
    trade = trades[0]
    assert trade.price == Decimal("150")
    assert trade.quantity == Decimal("10")
    assert trade.symbol == symbol

    # Check the order book is empty after the trade (both orders fully filled)
    bids, asks = engine.get_order_book_snapshot()
    assert bids == []
    assert asks == []

    # Verify the buy order is fully filled
    assert buy_order.remaining_quantity == 0
    assert buy_order.status == OrderStatus.FILLED

    # Verify the sell order is fully filled
    # Need to find the sell order in order book or track state outside
    # But since fully filled, it should be removed from order book
    # So just check order book empty is sufficient here

if __name__ == "__main__":
    test_buy_sell_matching()
    print("Test passed.")
