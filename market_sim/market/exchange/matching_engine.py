# matching_engine.py

import sys
import os
from typing import List, Optional, Set, Tuple
from decimal import Decimal
from datetime import datetime
import requests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.models.base import Order, Trade, OrderBook, OrderSide, OrderStatus, OrderType
from core.utils.time_utils import utc_now
from market_sim.Blockchain.marketTransactionsledger import mine_block_and_process
from market_sim.Blockchain.marketTransactionsledger import blockchain, mine_block_and_process


class MatchingEngine:
    def __init__(self, symbol: str, peers: Set[str]):
        self.order_book = OrderBook.create(symbol)
        self.trades: List[Trade] = []
        self.peers = peers

    def process_order(self, order: Order) -> List[Trade]:
        if order.type == OrderType.MARKET:
            return self._process_market_order(order)
        else:
            return self._process_limit_order(order)

    def _process_market_order(self, order: Order) -> List[Trade]:
        return self._match_order(order)

    def _process_limit_order(self, order: Order) -> List[Trade]:
        trades = self._match_order(order)
        if order.remaining_quantity > 0:
            self.order_book.add_order(order)
        return trades

    def _match_order(self, order: Order) -> List[Trade]:
        trades = []
        opposite_side = OrderSide.SELL if order.side == OrderSide.BUY else OrderSide.BUY
        opposite_book = self.order_book.asks if order.side == OrderSide.BUY else self.order_book.bids
        prices = sorted(opposite_book.keys(), reverse=(order.side == OrderSide.SELL))

        for price in prices:
            if order.remaining_quantity <= 0:
                break

            resting_orders = self.order_book.get_orders_at_price(opposite_side, price)
            for resting_order in resting_orders[:]:
                if order.remaining_quantity <= 0:
                    break

                if order.type == OrderType.LIMIT:
                    if (order.side == OrderSide.BUY and price > order.price) or \
                       (order.side == OrderSide.SELL and price < order.price):
                        break

                trade_quantity = min(order.remaining_quantity, resting_order.remaining_quantity)
                trade = self._create_trade(order, resting_order, trade_quantity, price)
                trades.append(trade)

                self._update_order_quantities(order, resting_order, trade_quantity)

                if resting_order.remaining_quantity == 0:
                    self.order_book.remove_order(resting_order)

        return trades

    def _create_trade(self, taker_order: Order, maker_order: Order,
                      quantity: Decimal, price: Decimal) -> Trade:
        # 1) create the Trade object locally
        buyer_order_id = taker_order.id if taker_order.side == OrderSide.BUY else maker_order.id
        seller_order_id = maker_order.id if taker_order.side == OrderSide.BUY else taker_order.id

        trade = Trade.create(
            symbol=self.order_book.symbol,
            price=price,
            quantity=quantity,
            buyer_order_id=buyer_order_id,
            seller_order_id=seller_order_id
        )

        # 2) pick one node as “primary” to post + mine
        primary = 'http://127.0.0.1:50139/'  # just grabs one URL
        tx_payload = {
            "symbol": trade.symbol,
            "price": float(trade.price),
            "quantity": float(trade.quantity),
            "buyer_order_id": str(trade.buyer_order_id),
            "seller_order_id": str(trade.seller_order_id),
        }

        # submit tx to primary
        tx_resp = requests.post(f"{primary}/transactions/new", json=tx_payload, timeout=3)
        if tx_resp.status_code != 200:
            print(f"Failed to submit tx to primary {primary}: {tx_resp.text}")
            return trade

        # mine on primary
        mine_resp = requests.get(f"{primary}/mine", timeout=10)
        if mine_resp.status_code != 200:
            print(f"Primary mine failed: {mine_resp.status_code} {mine_resp.text}")
            return trade

        # parse out the newly‑mined block
        block = mine_resp.json().get("block")
        if not block:
            print("No block returned by primary mine")
            return trade

        # 3) push that block to all the other peers
        for node in self.peers:
            if node == primary:
                continue
            try:
                r = requests.post(f"{node}/blocks/receive", json=block, timeout=5)
                if r.status_code != 200:
                    print(f"Peer {node} rejected block: {r.status_code} {r.text}")
            except Exception as e:
                print(f"Error pushing block to {node}: {e}")

        return trade



    def _broadcast_trade_to_blockchain(self, trade: Trade):
        payload = {
            "symbol": trade.symbol,
            "price": float(trade.price),
            "quantity": float(trade.quantity),
            "buyer_order_id": str(trade.buyer_order_id),
            "seller_order_id": str(trade.seller_order_id)
        }
        for peer in self.peers:
            try:
                requests.post(f"{peer}/transactions/new", json=payload, timeout=3)
            except Exception as e:
                print(f"Failed to broadcast trade to {peer}: {e}")

    def _update_order_quantities(self, taker_order: Order, maker_order: Order, trade_quantity: Decimal):
        taker_order.filled_quantity += trade_quantity
        taker_order.remaining_quantity -= trade_quantity
        taker_order.status = OrderStatus.FILLED if taker_order.remaining_quantity == 0 else OrderStatus.PARTIAL
        taker_order.updated_at = utc_now()

        maker_order.filled_quantity += trade_quantity
        maker_order.remaining_quantity -= trade_quantity
        maker_order.status = OrderStatus.FILLED if maker_order.remaining_quantity == 0 else OrderStatus.PARTIAL
        maker_order.updated_at = utc_now()

    def cancel_order(self, order_id: str) -> Optional[Order]:
        for orders in self.order_book.bids.values():
            for order in orders:
                if str(order.id) == order_id:
                    self.order_book.remove_order(order)
                    order.status = OrderStatus.CANCELLED
                    order.updated_at = utc_now()
                    return order

        for orders in self.order_book.asks.values():
            for order in orders:
                if str(order.id) == order_id:
                    self.order_book.remove_order(order)
                    order.status = OrderStatus.CANCELLED
                    order.updated_at = utc_now()
                    return order

        return None

    def get_order_book_snapshot(self, depth: int = 10) -> Tuple[List[Tuple[Decimal, Decimal]], List[Tuple[Decimal, Decimal]]]:
        bids = sorted(((price, sum(o.remaining_quantity for o in orders)) 
                      for price, orders in self.order_book.bids.items()),
                     reverse=True)[:depth]

        asks = sorted(((price, sum(o.remaining_quantity for o in orders))
                      for price, orders in self.order_book.asks.items()))[:depth]

        return bids, asks


# ----------------------------
# 🔍 Demo / Test (Optional)
# ----------------------------
if __name__ == "__main__":
    from uuid import uuid4

    symbol = "MSFT"
    peers = set()  # No peer sync for testing
    engine = MatchingEngine(symbol, peers)

    # Add a SELL LIMIT order
    sell_order = Order.create_limit_order(
        symbol=symbol,
        side=OrderSide.SELL,
        quantity=Decimal("16"),
        price=Decimal("156.00"),
        agent_id=str(uuid4())
    )
    engine.process_order(sell_order)

    # Add a BUY MARKET order
    buy_order = Order.create_market_order(
        symbol="MSFT",
        side=OrderSide.BUY,
        quantity=Decimal("5"),
        agent_id=str(uuid4())
    )
    trades = engine.process_order(buy_order)

    print(f"\n✅ Trades Executed: {len(trades)}")
    for t in trades:
        print(f"Trade: {t.symbol} | Qty: {t.quantity} | Price: {t.price}")

    print("\n📘 Order Book Snapshot:")
    bids, asks = engine.get_order_book_snapshot()
    print(f"Bids: {bids}")
    print(f"Asks: {asks}")
