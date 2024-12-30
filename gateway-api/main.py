import threading
import time
from order_book import OrderBook
from trade_executor import TradeExecutor
from transaction_logger import TransactionLogger
from utils import generate_random_order

def main():
    # Initialize the core components
    order_book = OrderBook()
    transaction_logger = TransactionLogger("transactions.log")
    trade_executor = TradeExecutor(order_book, transaction_logger)

    # Start the trade execution loop in a separate thread
    trade_executor_thread = threading.Thread(target=trade_executor.match_orders)
    trade_executor_thread.daemon = True
    trade_executor_thread.start()

    print("Crypto Exchange Gateway is running...")

    # Simulate random buy/sell orders
    while True:
        order = generate_random_order()
        if order["type"] == "buy":
            order_book.add_buy_order(order)
        else:
            order_book.add_sell_order(order)
        time.sleep(2)  # Generate a new order every 2 seconds

if __name__ == "__main__":
    main()