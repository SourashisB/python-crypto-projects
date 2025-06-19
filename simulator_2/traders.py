import random
from models import Order
from log import logger

class Trader:
    def __init__(self, trader_id, name):
        self.trader_id = trader_id
        self.name = name

    def __repr__(self):
        return f"{self.name}({self.trader_id})"

class Buyer(Trader):
    def place_order(self, exchange, stock, quantity, price):
        order = Order(self.trader_id, stock, quantity, price, 'buy')
        logger.info(f"{self.name} placing BUY order: {quantity} {stock} @ {price}")
        exchange.add_order(order)

class Seller(Trader):
    def place_order(self, exchange, stock, quantity, price):
        order = Order(self.trader_id, stock, quantity, price, 'sell')
        logger.info(f"{self.name} placing SELL order: {quantity} {stock} @ {price}")
        exchange.add_order(order)