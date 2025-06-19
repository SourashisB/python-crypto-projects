from models import Trade
from log import logger

class Exchange:
    def __init__(self):
        self.buy_orders = []
        self.sell_orders = []
        self.trades = []

    def add_order(self, order):
        if order.order_type == 'buy':
            self.buy_orders.append(order)
            self.buy_orders.sort(key=lambda o: (-o.price, o.timestamp))  # Highest price, earliest time
        else:
            self.sell_orders.append(order)
            self.sell_orders.sort(key=lambda o: (o.price, o.timestamp))  # Lowest price, earliest time
        logger.info(f"Order added: {order.order_type.upper()} {order.quantity} {order.stock} @ {order.price} by {order.trader_id}")
        self.match_orders()

    def match_orders(self):
        # Simple matching: match the top buy and top sell if price compatible
        while self.buy_orders and self.sell_orders:
            buy = self.buy_orders[0]
            sell = self.sell_orders[0]
            if buy.stock != sell.stock or buy.price < sell.price:
                break  # No match possible
            quantity = min(buy.quantity, sell.quantity)
            trade_price = sell.price  # Take the seller's price
            trade = Trade(buy.trader_id, sell.trader_id, buy.stock, quantity, trade_price)
            self.trades.append(trade)
            logger.info(f"Trade executed: {quantity} {buy.stock} @ {trade_price} between {buy.trader_id} (BUYER) and {sell.trader_id} (SELLER)")
            # Update order quantities
            buy.quantity -= quantity
            sell.quantity -= quantity
            if buy.quantity == 0:
                self.buy_orders.pop(0)
            if sell.quantity == 0:
                self.sell_orders.pop(0)