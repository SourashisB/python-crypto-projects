import uuid
from datetime import datetime

class Trade:
    def __init__(self, buyer_id, seller_id, stock, quantity, price):
        self.trade_id = str(uuid.uuid4())
        self.buyer_id = buyer_id
        self.seller_id = seller_id
        self.stock = stock
        self.quantity = quantity
        self.price = price
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self):
        return {
            'trade_id': self.trade_id,
            'buyer_id': self.buyer_id,
            'seller_id': self.seller_id,
            'stock': self.stock,
            'quantity': self.quantity,
            'price': self.price,
            'timestamp': self.timestamp
        }

class Order:
    def __init__(self, trader_id, stock, quantity, price, order_type):
        self.order_id = str(uuid.uuid4())
        self.trader_id = trader_id
        self.stock = stock
        self.quantity = quantity
        self.price = price
        self.order_type = order_type  # 'buy' or 'sell'
        self.timestamp = datetime.now().isoformat()