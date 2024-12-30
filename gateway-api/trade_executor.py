import time

class TradeExecutor:
    def __init__(self, order_book, transaction_logger):
        self.order_book = order_book
        self.transaction_logger = transaction_logger
        self.db = order_book.db

    def match_orders(self):
        while True:
            buy_orders, sell_orders = self.order_book.get_orders()

            if buy_orders and sell_orders:
                highest_buy = buy_orders[0]
                lowest_sell = sell_orders[0]

                if highest_buy[3] >= lowest_sell[3]:  # Match found
                    trade_price = (highest_buy[3] + lowest_sell[3]) / 2
                    trade_quantity = min(highest_buy[4], lowest_sell[4])

                    # Update quantities
                    remaining_buy_qty = highest_buy[4] - trade_quantity
                    remaining_sell_qty = lowest_sell[4] - trade_quantity

                    # Remove or update orders
                    if remaining_buy_qty == 0:
                        self.db.remove_order(highest_buy[0])
                    else:
                        self.db.add_order({
                            "type": "buy",
                            "account": highest_buy[2],
                            "price": highest_buy[3],
                            "quantity": remaining_buy_qty
                        })

                    if remaining_sell_qty == 0:
                        self.db.remove_order(lowest_sell[0])
                    else:
                        self.db.add_order({
                            "type": "sell",
                            "account": lowest_sell[2],
                            "price": lowest_sell[3],
                            "quantity": remaining_sell_qty
                        })

                    # Log the transaction
                    transaction = {
                        "buyer": highest_buy[2],
                        "seller": lowest_sell[2],
                        "price": trade_price,
                        "quantity": trade_quantity,
                        "timestamp": time.time(),
                    }
                    self.transaction_logger.log_transaction(transaction)
                    print(f"Trade Executed: {transaction}")
                else:
                    time.sleep(1)
            else:
                time.sleep(1)