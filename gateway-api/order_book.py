from database import Database

class OrderBook:
    def __init__(self):
        self.db = Database()

    def add_buy_order(self, order):
        order["type"] = "buy"
        self.db.add_order(order)
        print(f"New Buy Order Added: {order}")

    def add_sell_order(self, order):
        order["type"] = "sell"
        self.db.add_order(order)
        print(f"New Sell Order Added: {order}")

    def get_orders(self):
        return self.db.fetch_orders()