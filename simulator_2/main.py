import random
import time
import uuid

from traders import Buyer, Seller
from exchange import Exchange
from admin import Admin
from persistence import Persistence

def random_stock():
    return random.choice(["AAPL", "GOOGL", "TSLA", "AMZN", "MSFT"])

def main():
    # Setup
    exchange = Exchange()
    admin = Admin(exchange)
    persistence = Persistence(exchange)

    buyers = [Buyer(str(uuid.uuid4()), f"Buyer{i}") for i in range(3)]
    sellers = [Seller(str(uuid.uuid4()), f"Seller{i}") for i in range(3)]

    try:
        # Simulate random orders
        for _ in range(20):
            trader = random.choice(buyers + sellers)
            stock = random_stock()
            quantity = random.randint(1, 10)
            price = random.randint(90, 110)
            trader.place_order(exchange, stock, quantity, price)
            time.sleep(random.uniform(0.2, 1.0))

        # Admin access
        print("\n--- Trades Executed ---")
        admin.list_trades()
        input("\nPress Enter to exit and save trades...")

    finally:
        persistence.save_trades()
        persistence.stop()

if __name__ == "__main__":
    main()